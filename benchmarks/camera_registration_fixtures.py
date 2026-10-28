"""Deterministic eager-profile camera fixture inputs and complete output layouts."""
import numpy as np


def payloads():
    def make(case, function, kwargs):
        return {'case_id': case, 'function': function, 'profile': 'eager-f32', 'kwargs': kwargs}
    for batch in (1, 8):
        for h, w, channels in ((1,1,1), (3,4,3), (4,7,4)):
            k = np.broadcast_to(np.eye(3,dtype=np.float32), (batch,3,3)).copy()
            k[:,0,0], k[:,1,1], k[:,0,2], k[:,1,2] = 1.7, 2.1, .3, .4
            target = k.copy(); target[:,0,0] = 2.4; target[:,1,1] = 1.4
            image = np.arange(batch*channels*h*w,dtype=np.float32).reshape(batch,channels,h,w)/np.float32(13)
            coeff = np.tile(np.array([.02,-.003,.004,-.005,.0001],dtype=np.float32),(batch,1))
            transform = np.broadcast_to(np.eye(4,dtype=np.float32), (batch,4,4)).copy()
            transform[:,0,3] = .05
            depth = (np.arange(batch*h*w,dtype=np.float32).reshape(batch,1,h,w)%7+1)/np.float32(4)
            depth.reshape(-1)[0] = np.nan
            for brown in (False, True):
                source_opts = {'source_distortion_model':'brown_conrady','source_distortion_coefficients':coeff.copy()} if brown else {}
                depth_opts = {'depth_distortion_model':'brown_conrady','depth_distortion_coefficients':coeff.copy()} if brown else {}
                reg_opts = dict(depth_opts)
                if brown:
                    reg_opts.update(rgb_distortion_model='brown_conrady',rgb_distortion_coefficients=coeff.copy())
                label = f'b{batch}-{h}x{w}-c{channels}-brown{int(brown)}'
                yield make('grid-'+label, 'camera_rectify_grid', dict(source_intrinsics=k.copy(),target_intrinsics=target.copy(),source_hw=(h,w),output_hw=(3,5),debug=True,**source_opts))
                yield make('rays-'+label, 'depth_camera_rays', dict(depth_intrinsics=k.copy(),depth_hw=(h,w),**depth_opts))
                for sampling in ('nearest','bilinear'):
                    yield make('rectify-'+sampling+'-'+label,'camera_rectify',dict(image=image.copy(),source_intrinsics=k.copy(),target_intrinsics=target.copy(),output_hw=(3,5),sampling=sampling,debug=True,**source_opts))
                yield make('register-'+label,'depth_register',dict(depth=depth.copy(),depth_intrinsics=k.copy(),rgb_intrinsics=target.copy(),transform_rgb_depth=transform.copy(),output_hw=(3,5),debug=True,**reg_opts))
    image = np.arange(8,dtype=np.float32).reshape(1,1,2,4)
    uv = np.array([[[[np.nan,np.inf],[-.5,0],[.5,0],[1.5,0],[3.5,1],[2.25,.5],[-np.inf,np.nan]]]],dtype=np.float32)
    valid = np.array([[[[True,True,True,True,True,True,False]]]],dtype=bool)
    for sampling in ('nearest','bilinear'):
        yield make('cached-nonfinite-'+sampling,'camera_rectify_from_grid',dict(image=image.copy(),source_uv=uv.copy(),valid=valid.copy(),sampling=sampling))
    rng = np.random.default_rng(6743)
    n = 512
    cancellation_uv = rng.uniform(.1,.9,(n,1,1,2)).astype(np.float32)
    cancellation_image = rng.uniform(-10000,10000,(n,1,2,2)).astype(np.float32)
    u = cancellation_uv[:,0,0,0].astype(float); v = cancellation_uv[:,0,0,1].astype(float)
    cancellation_image[:,0,1,1] = -((cancellation_image[:,0,0,0]*(1-u)*(1-v)+cancellation_image[:,0,0,1]*u*(1-v)+cancellation_image[:,0,1,0]*(1-u)*v)/(u*v)).astype(np.float32)
    yield make('cached-cancellation','camera_rectify_from_grid',dict(image=cancellation_image,source_uv=cancellation_uv,valid=np.ones((n,1,1,1),dtype=bool),sampling='bilinear'))
    k = np.eye(3,dtype=np.float32)[None]
    for name, values in [('radial', [.1,-.02,0,0,.001]),('tangential',[0,0,.01,-.02,0])]:
        coeff = np.array([values],dtype=np.float32)
        yield make('grid-'+name,'camera_rectify_grid',dict(source_intrinsics=k.copy(),target_intrinsics=k.copy(),source_hw=(3,4),output_hw=(3,4),source_distortion_model='brown_conrady',source_distortion_coefficients=coeff,debug=False))
        yield make('rays-'+name,'depth_camera_rays',dict(depth_intrinsics=k.copy(),depth_hw=(3,4),depth_distortion_model='brown_conrady',depth_distortion_coefficients=coeff))
    yield make('rectify-nodebug','camera_rectify',dict(image=image.copy(),source_intrinsics=k.copy(),target_intrinsics=k.copy(),output_hw=(2,4),debug=False))
    transform = np.eye(4,dtype=np.float32)[None]
    rays = np.zeros((1,1,5,2),dtype=np.float32)
    yield make('register-nodebug','depth_register',dict(depth=np.ones((1,1,1,5),dtype=np.float32),depth_intrinsics=k.copy(),rgb_intrinsics=k.copy(),transform_rgb_depth=transform.copy(),output_hw=(1,1),depth_rays=rays.copy(),debug=False))
    rotation = np.eye(4,dtype=np.float32)[None]
    rotation[0,:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    rotation[0,:3,3] = [.3,.2,.1]
    reg_depth = np.array([[[[.5,1,2],[3,4,np.nan]]]],dtype=np.float32)
    coefficients = np.array([[.02,-.003,.004,-.005,.0001]],dtype=np.float32)
    for depth_brown in (False, True):
        for rgb_brown in (False, True):
            options = {}
            if depth_brown:
                options.update(depth_distortion_model='brown_conrady',depth_distortion_coefficients=coefficients.copy())
            if rgb_brown:
                options.update(rgb_distortion_model='brown_conrady',rgb_distortion_coefficients=coefficients.copy())
            yield make(f'rotated-depth{int(depth_brown)}-rgb{int(rgb_brown)}','depth_register',dict(depth=reg_depth.copy(),depth_intrinsics=k.copy(),rgb_intrinsics=k.copy(),transform_rgb_depth=rotation.copy(),output_hw=(4,5),debug=True,**options))
    eps = np.float32(np.finfo(np.float32).eps)
    boundary_depth = np.array([np.nextafter(eps,np.float32(0)),eps,np.nextafter(eps,np.float32(np.inf)),1,2,np.nextafter(np.float32(2),np.float32(np.inf))],dtype=np.float32).reshape(1,1,1,6)
    boundary_rays = np.zeros((1,1,6,2),dtype=np.float32)
    boundary_rays[0,0,:,0] = [-.5,.5,1.5,2.5,3.5,4.5]
    yield make('depth-interval-epsilon-half-ties','depth_register',dict(depth=boundary_depth,depth_intrinsics=k.copy(),rgb_intrinsics=k.copy(),transform_rgb_depth=transform.copy(),output_hw=(1,5),depth_rays=boundary_rays,min_depth=float(eps/2),max_depth=2.,debug=True))
    interval_depth = np.array([[[[.5,1,2,np.nextafter(np.float32(2),np.float32(np.inf))]]]],dtype=np.float32)
    yield make('inclusive-min-max','depth_register',dict(depth=interval_depth,depth_intrinsics=k.copy(),rgb_intrinsics=k.copy(),transform_rgb_depth=transform.copy(),output_hw=(1,4),min_depth=1.,max_depth=2.,debug=True))
    nf_image = np.array([[[[np.nan,np.inf],[-np.inf,1.]]]],dtype=np.float32)
    nf_uv = np.array([[[[0.,0.],[.5,.5],[1.,1.]]]],dtype=np.float32)
    nf_valid = np.array([[[[True,True,False]]]],dtype=bool)
    for sampling in ('nearest','bilinear'):
        yield make('nonfinite-image-'+sampling,'camera_rectify_from_grid',dict(image=nf_image.copy(),source_uv=nf_uv.copy(),valid=nf_valid.copy(),sampling=sampling))
    for name, values in [('tie',[2,1,1,np.nan,0]),('ulp',[1,np.nextafter(np.float32(1),np.float32(0)),1,np.nan,0]),('empty',[np.nan,0,-1,np.inf,-np.inf])]:
        yield make('collisions-'+name,'depth_register',dict(depth=np.array(values,dtype=np.float32).reshape(1,1,1,5),depth_intrinsics=k.copy(),rgb_intrinsics=k.copy(),transform_rgb_depth=transform.copy(),output_hw=(1,1),depth_rays=rays.copy(),debug=True))


def output_schema(payload):
    kw=payload['kwargs']; func=payload['function']
    f=np.dtype('float32').str; b=np.dtype('bool').str; i=np.dtype('int64').str
    if func == 'depth_camera_rays':
        n=kw['depth_intrinsics'].shape[0]; h,w=kw['depth_hw']
        return {'result': (f,(n,h,w,2))}
    if func == 'depth_register':
        n,_,h,w=kw['depth'].shape; th,tw=kw['output_hw']; shape=(n,1,th,tw)
        out={'depth':(f,shape),'valid':(b,shape)}
        if kw.get('debug',False):
            out.update(winner_source_index=(i,shape),collision_count=(i,shape),source_uv=(f,(n,h*w,2)),projected_z=(f,(n,h*w)),source_valid=(b,(n,h*w)))
        return out
    if func == 'camera_rectify_from_grid':
        n,c,_,_=kw['image'].shape; h,w=kw['source_uv'].shape[1:3]
    elif func == 'camera_rectify':
        n,c,_,_=kw['image'].shape; h,w=kw['output_hw']
    elif func == 'camera_rectify_grid':
        n=kw['source_intrinsics'].shape[0]; h,w=kw['output_hw']
    else:
        raise ValueError('unknown fixture function')
    if func == 'camera_rectify_grid':
        out={'source_uv':(f,(n,h,w,2)), 'valid':(b,(n,1,h,w))}
    else:
        out={'image':(f,(n,c,h,w)), 'valid':(b,(n,1,h,w))}
    if kw.get('debug',False):
        out.update(source_uv=(f,(n,h,w,2)), target_ray=(f,(n,h,w,3)), displacement=(f,(n,1,h,w)))
    return out
