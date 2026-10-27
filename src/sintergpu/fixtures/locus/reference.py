"""Scalar test oracle. Deliberately test-only CPU work; never camera ingestion."""
import math,struct

def f4(x=0,y=0,z=0,w=0):return [float(x),float(y),float(z),float(w)]
def valid(z):return math.isfinite(z) and z>0
def bits(z):return struct.unpack('<I',struct.pack('<f',z))[0]
def number(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def key(point,size):
    q=[math.floor(v/size) for v in point]
    if any(v < -512 or v > 511 for v in q):return None
    q=[v+512 for v in q];return (q[0]|q[1]<<10|q[2]<<20)+1
def validate(name,p,inputs,out,state):
    if len(p)!=32 or not all(math.isfinite(v) for v in p):raise ValueError('finite 128-byte parameters required')
    def integral(v,limit=8192):
        if v!=int(v) or v<=0 or v>limit:raise ValueError('invalid dimension/stride')
        return int(v)
    w,h=integral(p[0]),integral(p[1]);stride=integral(p[2]);outputstride=integral(p[3])
    if name in ('transform_points','voxel_insert','coverage_render'):
        if h!=1:raise ValueError('point kernels require height1')
    def image(buffer,width,height,pitch):
        pitch=integral(pitch)
        if pitch<width or len(buffer)<(height-1)*pitch+width:raise ValueError('binding extent')
    if name not in ('depth_register','voxel_insert'):image(out,w,h,outputstride)
    if name not in ('depth_resolve','coverage_heatmap','coverage_render','similarity','resize','normalize'):image(inputs[0],w,h,stride)
    if name in ('resize','normalize'):
        iw,ih=(p[4],p[5])if name=='resize' else(p[8],p[9]);image(inputs[0],integral(iw),integral(ih),p[6]if name=='resize'else stride)
    if name=='yuv_to_rgb':
        for plane in inputs[1:]:image(plane,(w+1)//2,(h+1)//2,p[4])
        if p[5]not in(0,1)or p[6]not in(0,1)or p[7]!=0:raise ValueError('unsupported color metadata')
    if name in('undistort','depth_register','backproject'):
        if p[28]!=1:raise ValueError('missing calibration')
        focal=p[8:10]+p[12:14]if name=='depth_register'else p[4:6]
        if any(x<=0 for x in focal):raise ValueError('invalid focal length')
    if name=='depth_register':
        cw,ch=integral(p[4]),integral(p[5]);image(state,cw,ch,p[6]);
        if p[7]<=0:raise ValueError('depth unit')
    if name=='depth_resolve':image(state,w,h,stride)
    if name in('invalid_depth','backproject')and p[4 if name=='invalid_depth'else 8]<=0:raise ValueError('depth unit')
    if name=='invalid_depth'and not 0<=p[5]<=p[6]:raise ValueError('depth range')
    if name=='bilateral'and any(x<=0 for x in p[4:6]):raise ValueError('bilateral sigma')
    if name=='metrics':image(inputs[1],w,h,p[6])
    if name=='reduce'and(w*h>1<<20 or len(state)<4):raise ValueError('quality accumulator limit')
    if name in('voxel_insert','coverage_heatmap','coverage_render'):
        if p[4]<=0:raise ValueError('voxel scale')
        if name=='voxel_insert':
            capacity=integral(p[5],1<<20);probes=integral(p[6],64)
            if probes>capacity or len(state)<2*capacity+3 or w>1<<20:raise ValueError('voxel capacity')
        elif name=='coverage_heatmap':image(state,w*2,h,stride*2)
        elif len(state)<2*w:raise ValueError('coverage extent')
    if name=='normalize'and(any(v<=0 for v in p[6:8])or any(v<=0 for v in p[16:19])):raise ValueError('crop/stddev')
    if name=='similarity':
        features=int(p[4]);pitch=int(p[5]);
        if features<0 or features>256 or features!=p[4] or pitch<max(1,features)or pitch!=p[5]:raise ValueError('feature bounds')
        image(inputs[0],w*pitch,h,stride*pitch)
        if len(inputs[1])<max(1,features):raise ValueError('query bounds')
    if name=='mask_postprocess'and p[5]not in(0,1,2):raise ValueError('morphology mode')

def execute(name,p,inputs,out,state):
    validate(name,p,inputs,out,state);a,b,c=inputs;out=[x[:]for x in out];state=state[:]
    w,h,s,os=map(int,p[:4]);
    def sample(image,x,y,width,height,stride):
        return image[int(y)*int(stride)+int(x)] if math.isfinite(x)and math.isfinite(y)and 0<=x<width and 0<=y<height else f4()
    def transform(v):return [sum(p[16+i*4+j]*v[j]for j in range(3))+p[19+i*4]for i in range(3)]
    for y in range(h):
        for x in range(w):
            i=y*s+x;o=y*os+x
            if name=='yuv_to_rgb':
                yy=a[i][0];uv=(y//2)*int(p[4])+x//2;u=b[uv][0]-128;v=c[uv][0]-128
                if p[5]==1:yy=(yy-16)*255/219;u*=255/224;v*=255/224
                q=[yy+1.5748*v,yy-.187324*u-.468124*v,yy+1.8556*u]if p[6]==1 else[yy+1.402*v,yy-.344136*u-.714136*v,yy+1.772*u]
                out[o]=[min(1,max(0,v/255))for v in q]+[1]
            elif name=='resize':out[o]=sample(a,(x+.5)*p[4]/w,(y+.5)*p[5]/h,p[4],p[5],p[6])[:]
            elif name=='undistort':
                qx=(x-p[6])/p[4];qy=(y-p[7])/p[5];rr=qx*qx+qy*qy;radial=1+p[8]*rr+p[9]*rr**2+p[10]*rr**3
                xx=(qx*radial+2*p[12]*qx*qy+p[13]*(rr+2*qx*qx))*p[4]+p[6]+.5
                yy=(qy*radial+p[12]*(rr+2*qy*qy)+2*p[13]*qx*qy)*p[5]+p[7]+.5
                out[o]=sample(a,xx,yy,w,h,s)[:]
            elif name=='invalid_depth':
                z=a[i][0]*p[4];good=valid(z)and p[5]<=z<=p[6];out[o]=f4(z if good else 0,1 if good else 0)
            elif name=='bilateral':
                z=a[i][0];total=weight=0
                if valid(z):
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            if not(0<=x+dx<w and 0<=y+dy<h):continue
                            n=a[(y+dy)*s+x+dx][0]
                            if valid(n):
                                alpha=math.exp(-(dx*dx+dy*dy)/(2*p[4]**2)-(n-z)**2/(2*p[5]**2));weight+=alpha;total+=alpha*n
                out[o]=f4(total/weight if weight else 0,1 if weight else 0)
            elif name=='depth_register':
                z=a[i][0]*p[7]
                if not valid(z):continue
                q=transform([(x-p[10])/p[8]*z,(y-p[11])/p[9]*z,z])
                if not valid(q[2]):continue
                xx=q[0]/q[2]*p[12]+p[14];yy=q[1]/q[2]*p[13]+p[15]
                if not(-.5<=xx<p[4]-.5 and -.5<=yy<p[5]-.5):continue
                index=math.floor(yy+.5)*int(p[6])+math.floor(xx+.5);state[index]=min(state[index],bits(q[2]))
            elif name=='depth_resolve':
                z=number(state[i]);out[o]=f4(z if valid(z)else 0,1 if valid(z)else 0)
            elif name=='metrics':
                lum=lambda xx,yy:sum(a[yy*s+xx][j]*v for j,v in enumerate((.2126,.7152,.0722)))
                center=lum(x,y);left=lum(max(0,x-1),y);right=lum(min(w-1,x+1),y);up=lum(x,max(0,y-1));down=lum(x,min(h-1,y+1))
                if not all(map(math.isfinite,a[i][:3])):out[o]=f4(0,0,0,valid(b[y*int(p[6])+x][0]));continue
                left,right,up,down=[v if math.isfinite(v)else center for v in(left,right,up,down)]
                out[o]=f4(abs(right-left)+abs(down-up),abs(left+right+up+down-4*center),any(v<=p[4]or v>=p[5]for v in a[i][:3]),valid(b[y*int(p[6])+x][0]))
            elif name=='reduce':
                for j in(0,1):state[j]+=int(min(4,max(0,a[i][j]if math.isfinite(a[i][j])else 0))*256+.5)
                state[2]+=a[i][2]>0;state[3]+=a[i][3]>0
            elif name=='backproject':
                z=a[i][0]*p[8];out[o]=f4((x-p[6])/p[4]*z,(y-p[7])/p[5]*z,z,1)if valid(z)else f4()
            elif name=='transform_points':
                q=transform(a[x]);out[x]=q+[1]if a[x][3]>0 and all(map(math.isfinite,q))else f4()
            elif name=='voxel_insert':
                if a[x][3]<=0 or not all(map(math.isfinite,a[x][:3])):continue
                cap=int(p[5]);k=key(a[x][:3],p[4]);success=False
                if k is not None:
                    for j in range(int(p[6])):
                        slot=(((k*2654435761)&0xffffffff)+j)&0xffffffff;slot%=cap
                        if state[2*slot]in(0,k):state[2*slot]=k;state[2*slot+1]+=1;state[2*cap]+=1;success=True;break
                        state[2*cap+2]+=1
                if not success:state[2*cap+1]+=1
            elif name=='coverage_heatmap':
                count=state[2*i+1];t=min(1,count/p[4]);out[o]=f4(1-t,t,.1,1)if count else f4(0,0,0,1)
            elif name=='coverage_render':
                k=state[2*x]
                if k:k-=1;out[x]=f4((((k&1023)-512)+.5)*p[4],((((k>>10)&1023)-512)+.5)*p[4],((((k>>20)&1023)-512)+.5)*p[4],1)
                else:out[x]=f4()
            elif name=='normalize':
                q=sample(a,p[4]+(x+.5)/w*p[6],p[5]+(y+.5)/h*p[7],p[8],p[9],s);v=[(q[j]-p[12+j])/p[16+j]for j in range(3)];out[o]=v+[q[3]]if all(map(math.isfinite,v))else f4()
            elif name=='similarity':
                aa=[a[i*int(p[5])+j][0]for j in range(int(p[4]))];bb=[b[j][0]for j in range(int(p[4]))];dot=sum(x*y for x,y in zip(aa,bb));den=sum(x*x for x in aa)*sum(x*x for x in bb);value=dot/math.sqrt(den)if den>0 and all(map(math.isfinite,aa+bb))else 0;out[o]=f4(value if math.isfinite(value)else 0,0,0,1)
            elif name=='mask_postprocess':
                radius=0 if p[5]==0 else 1;hits=[]
                for dy in range(-radius,radius+1):
                    for dx in range(-radius,radius+1):
                        if 0<=x+dx<w and 0<=y+dy<h:v=a[(y+dy)*s+x+dx][0];hits.append(math.isfinite(v)and v>=p[4])
                        else:hits.append(False)
                out[o]=f4(all(hits)if p[5]==2 else any(hits),0,0,1)
            else:raise ValueError('unknown kernel')
    return out,state
