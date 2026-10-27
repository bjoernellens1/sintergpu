"""Independent NumPy float64 camera-format oracles and pre-dispatch validation.

Portable GPU FP32 profiles allow <=1 LSB against these oracles. They do not
replace the retained HIP exact profile. No device, submission or readback here.
"""
import math
import numpy as np


class ColorSpec:
    def __init__(self, matrix='bt601', range='limited', profile='portable-fp32', chroma_siting='cosited-nearest'):
        if profile != 'portable-fp32':
            raise ValueError('HIP-exact requests require the retained HIP implementation')
        if matrix not in ('bt601', 'bt709') or range not in ('limited', 'full'):
            raise ValueError('explicit bt601/bt709 and limited/full tags required')
        if chroma_siting != 'cosited-nearest':
            raise ValueError('only explicit cosited-nearest decoding is supported')
        self.matrix, self.range, self.profile, self.chroma_siting = matrix, range, profile, chroma_siting


def _aligned(n):
    return (n+3)//4*4


def validate_request(operation, *, width, height, output_width=None, output_height=None,
                     source_pitch=None, uv_pitch=None, output_pitch=None, output_uv_pitch=None,
                     source_capacity=None, uv_capacity=None, output_capacity=None,
                     output_uv_capacity=None, calibration=None, map_capacity=None, tags=None, for_dispatch=False):
    """Validate metadata without touching borrowed GPU buffers; return shader constants.

    Capacities are bytes. Omitting a capacity means caller checks it separately;
    provided capacities must cover a 4-byte-padded allocation, not just live bytes.
    GPU callers must set for_dispatch=True to require every bound capacity.
    Map coordinates are pixel-center indices in the unrectified source image.
    """
    tags = tags or ColorSpec()
    if operation not in ('nv12_decode','nv12_encode','yuyv_decode','remap','resize','fused_remap'):
        raise ValueError('unsupported color operation')
    if not isinstance(tags,ColorSpec):
        raise ValueError('ColorSpec required')
    ow,oh = output_width if output_width is not None else width, output_height if output_height is not None else height
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) or v <= 0 or v > 65535 for v in (width,height,ow,oh)):
        raise ValueError('dimensions must be integers in [1,65535]')
    if operation in ('nv12_decode','nv12_encode','yuyv_decode') and (ow != width or oh != height):
        raise ValueError('decode/encode output extent must equal camera extent')
    if for_dispatch:
        required = [source_capacity, output_capacity]
        if operation=='nv12_decode': required.append(uv_capacity)
        if operation=='nv12_encode': required.append(output_uv_capacity)
        if operation in ('remap','fused_remap'): required.append(map_capacity)
        if any(value is None for value in required):
            raise ValueError('all borrowed capacities are required before dispatch')
    if operation in ('nv12_decode','nv12_encode') and (width%2 or height%2):
        raise ValueError('NV12 dimensions must be even')
    if operation in ('yuyv_decode','fused_remap') and width%2:
        raise ValueError('YUYV camera width must be even')
    if operation in ('remap','fused_remap') and calibration is None:
        raise ValueError('explicit calibrated map required')
    if operation in ('remap','fused_remap'):
        maps = np.asarray(calibration)
        if maps.dtype != np.float32 or maps.shape != (oh,ow,2) or not maps.flags.c_contiguous:
            raise ValueError('map must be contiguous float32[output_height,output_width,2]')
        if map_capacity is not None and map_capacity < ow*oh*8:
            raise ValueError('map capacity too small')
    source_row = width*(1 if operation=='nv12_decode' else 2 if operation in ('yuyv_decode','fused_remap') else 3)
    output_row = width if operation=='nv12_encode' else ow*3
    sp = source_row if source_pitch is None else source_pitch
    up = width if uv_pitch is None else uv_pitch
    dp = output_row if output_pitch is None else output_pitch
    oup = width if output_uv_pitch is None else output_uv_pitch
    for pitch,minimum,rows,capacity in [(sp,source_row,height,source_capacity),(dp,output_row,height if operation=='nv12_encode' else oh,output_capacity)]:
        if not isinstance(pitch,(int,np.integer)) or pitch < minimum or pitch*rows > 2**32-4:
            raise ValueError('invalid byte pitch/address extent')
        if capacity is not None and (not isinstance(capacity,(int,np.integer)) or capacity < _aligned(pitch*rows) or capacity%4):
            raise ValueError('capacity must cover 4-byte padded allocation')
    for pitch,capacity in ((up,uv_capacity),(oup,output_uv_capacity)):
        if operation.startswith('nv12'):
            if not isinstance(pitch,(int,np.integer)) or pitch < width or pitch*(height//2) > 2**32-4:
                raise ValueError('invalid UV pitch')
            if capacity is not None and (capacity < _aligned(pitch*(height//2)) or capacity%4):
                raise ValueError('UV allocation must be 4-byte padded')
    return dict(srcWidth=int(width),srcHeight=int(height),dstWidth=int(ow),dstHeight=int(oh),
                sourcePitch=int(sp),uvPitch=int(up),outputPitch=int(dp),outputUVPitch=int(oup),
                matrix709=int(tags.matrix=='bt709'),limited=int(tags.range=='limited'))


def _bytes(array):
    array=np.asarray(array)
    if array.dtype != np.uint8 or not array.flags.c_contiguous:
        raise ValueError('contiguous uint8 byte storage required')
    return array.reshape(-1)


def _output(output, pitch, height):
    if output is None:
        return np.zeros(_aligned(pitch*height),np.uint8)
    output=_bytes(output)
    if not output.flags.writeable or len(output)<_aligned(pitch*height) or len(output)%4:
        raise ValueError('output must be writable and 4-byte padded')
    return output


def _store_rgb(values, output, pitch):
    height,width,_=values.shape
    for row in range(height):
        output[row*pitch:row*pitch+width*3]=values[row].reshape(-1)
    return output


def _weights(tags):
    return (0.2126,0.0722) if tags.matrix=='bt709' else (0.299,0.114)


def _decode64(y,u,v,tags):
    kr,kb=_weights(tags); kg=1-kr-kb
    y=np.asarray(y,dtype=np.float64); u=np.asarray(u,dtype=np.float64)-128; v=np.asarray(v,dtype=np.float64)-128
    if tags.range=='limited':
        y=(y-16)*(255/219); u=u*(255/224); v=v*(255/224)
    values=np.stack([y+2*(1-kr)*v,y-2*kb*(1-kb)/kg*u-2*kr*(1-kr)/kg*v,y+2*(1-kb)*u],axis=-1)
    return np.rint(np.clip(values,0,255)).astype(np.uint8)


def nv12_to_rgb(y,uv,*,width,height,y_pitch=None,uv_pitch=None,rgb_pitch=None,tags=None,output=None):
    tags=tags or ColorSpec(); y=_bytes(y); uv=_bytes(uv)
    params=validate_request('nv12_decode',width=width,height=height,source_pitch=y_pitch,uv_pitch=uv_pitch,output_pitch=rgb_pitch,
                            source_capacity=len(y),uv_capacity=len(uv),tags=tags)
    yp,up,rp=params['sourcePitch'],params['uvPitch'],params['outputPitch']
    luma=np.stack([y[row*yp:row*yp+width] for row in range(height)])
    chroma=np.stack([uv[row*up:row*up+width] for row in range(height//2)]).reshape(height//2,width//2,2)
    chroma=chroma.repeat(2,axis=0).repeat(2,axis=1)
    return _store_rgb(_decode64(luma,chroma[:,:,0],chroma[:,:,1],tags),_output(output,rp,height),rp)


def yuyv_to_rgb(source,*,width,height,source_pitch=None,rgb_pitch=None,tags=None,output=None):
    tags=tags or ColorSpec(); source=_bytes(source)
    p=validate_request('yuyv_decode',width=width,height=height,source_pitch=source_pitch,output_pitch=rgb_pitch,source_capacity=len(source),tags=tags)
    packed=np.stack([source[row*p['sourcePitch']:row*p['sourcePitch']+width*2] for row in range(height)]).reshape(height,width//2,4)
    y=packed[:,:,[0,2]].reshape(height,width)
    u=packed[:,:,1].repeat(2,axis=1); v=packed[:,:,3].repeat(2,axis=1)
    return _store_rgb(_decode64(y,u,v,tags),_output(output,p['outputPitch'],height),p['outputPitch'])


def rgb_to_nv12(source,*,width,height,source_pitch=None,y_pitch=None,uv_pitch=None,tags=None,y_output=None,uv_output=None):
    tags=tags or ColorSpec(); source=_bytes(source)
    p=validate_request('nv12_encode',width=width,height=height,source_pitch=source_pitch,output_pitch=y_pitch,output_uv_pitch=uv_pitch,source_capacity=len(source),tags=tags)
    rgb=np.stack([source[row*p['sourcePitch']:row*p['sourcePitch']+width*3] for row in range(height)]).reshape(height,width,3).astype(np.float64)
    kr,kb=_weights(tags); kg=1-kr-kb
    y=kr*rgb[:,:,0]+kg*rgb[:,:,1]+kb*rgb[:,:,2]
    cb=(rgb[:,:,2]-y)/(2*(1-kb)); cr=(rgb[:,:,0]-y)/(2*(1-kr))
    yy=y*(219/255)+16 if tags.range=='limited' else y
    scale=224/255 if tags.range=='limited' else 1
    # Centered 2x2 box before chroma quantization; no top-left decimation.
    uv=np.stack([cb.reshape(height//2,2,width//2,2).mean(axis=(1,3))*scale+128,
                 cr.reshape(height//2,2,width//2,2).mean(axis=(1,3))*scale+128],axis=-1)
    yy=np.rint(np.clip(yy,0,255)).astype(np.uint8); uv=np.rint(np.clip(uv,0,255)).astype(np.uint8)
    yo=_output(y_output,p['outputPitch'],height); uvo=_output(uv_output,p['outputUVPitch'],height//2)
    for row in range(height): yo[row*p['outputPitch']:row*p['outputPitch']+width]=yy[row]
    for row in range(height//2): uvo[row*p['outputUVPitch']:row*p['outputUVPitch']+width]=uv[row].reshape(-1)
    return yo,uvo


def _corners(x,y,width,height):
    if not math.isfinite(float(x)) or not math.isfinite(float(y)): return None
    x=min(max(float(x),0),width-1); y=min(max(float(y),0),height-1)
    x0,y0=math.floor(x),math.floor(y)
    return x0,y0,min(x0+1,width-1),min(y0+1,height-1),x-x0,y-y0


def _blend(a,b,c,d,fx,fy):
    a,b,c,d=[np.asarray(v,dtype=np.float64) for v in (a,b,c,d)]
    top=a+(b-a)*fx; bottom=c+(d-c)*fx
    return np.rint(np.clip(top+(bottom-top)*fy,0,255)).astype(np.uint8)


def remap_rgb(source,maps,*,width,height,source_pitch=None,rgb_pitch=None,output=None):
    source=_bytes(source); maps=np.asarray(maps); oh,ow=maps.shape[:2]
    p=validate_request('remap',width=width,height=height,output_width=ow,output_height=oh,source_pitch=source_pitch,
                       output_pitch=rgb_pitch,source_capacity=len(source),calibration=maps,map_capacity=maps.nbytes)
    output=_output(output,p['outputPitch'],oh)
    def at(x,y): return source[y*p['sourcePitch']+x*3:y*p['sourcePitch']+x*3+3]
    for row in range(oh):
        for col in range(ow):
            corners=_corners(*maps[row,col],width,height)
            value=np.zeros(3,np.uint8)
            if corners:
                x0,y0,x1,y1,fx,fy=corners
                value=_blend(at(x0,y0),at(x1,y0),at(x0,y1),at(x1,y1),fx,fy)
            offset=row*p['outputPitch']+col*3; output[offset:offset+3]=value
    return output


def resize_rgb(source,*,width,height,output_width,output_height,source_pitch=None,rgb_pitch=None,output=None):
    # Standard half-pixel centers; clamp-to-edge bilinear reconstruction.
    x=(np.arange(output_width,dtype=np.float64)+.5)*width/output_width-.5
    y=(np.arange(output_height,dtype=np.float64)+.5)*height/output_height-.5
    xx,yy=np.meshgrid(x,y); maps=np.stack([xx,yy],axis=-1).astype(np.float32)
    return remap_rgb(source,maps,width=width,height=height,source_pitch=source_pitch,rgb_pitch=rgb_pitch,output=output)


def fused_yuyv_remap(source,maps,*,width,height,source_pitch=None,rgb_pitch=None,tags=None,output=None):
    """Scalar oracle samples and rounds four decoded RGB pixels before blending."""
    tags=tags or ColorSpec(); source=_bytes(source); maps=np.asarray(maps); oh,ow=maps.shape[:2]
    p=validate_request('fused_remap',width=width,height=height,output_width=ow,output_height=oh,source_pitch=source_pitch,
                       output_pitch=rgb_pitch,source_capacity=len(source),calibration=maps,map_capacity=maps.nbytes,tags=tags)
    output=_output(output,p['outputPitch'],oh)
    def at(x,y):
        base=y*p['sourcePitch']+(x//2)*4
        return _decode64(source[base+(2 if x%2 else 0)],source[base+1],source[base+3],tags)
    for row in range(oh):
        for col in range(ow):
            corners=_corners(*maps[row,col],width,height); value=np.zeros(3,np.uint8)
            if corners:
                x0,y0,x1,y1,fx,fy=corners
                value=_blend(at(x0,y0),at(x1,y0),at(x0,y1),at(x1,y1),fx,fy)
            offset=row*p['outputPitch']+col*3; output[offset:offset+3]=value
    return output
