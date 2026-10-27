from .reference import f4,bits,key

def cases():
    result=[]
    def add(name,label,w=3,h=3,stride=4,params=None,a=None,b=None,c=None,state=None,output=None):
        p=[0.]*32;p[:4]=[w,h,stride,stride];p[16:28]=[1,0,0,0,0,1,0,0,0,0,1,0];p[28]=1
        if params:
            for i,v in params.items():p[i]=v
        data=[a or [f4(.25,.5,.75,1)for _ in range(stride*h)],b or [f4()for _ in range(stride*h)],c or [f4()for _ in range(stride*h)]]
        result.append({'name':name,'label':label,'width':w,'height':h,'params':p,'inputs':data,'output':output or [f4()for _ in range(stride*h)],'state':state if state is not None else[0]*4})
    y=[f4(v)for v in [16,235,81,999,145,100,200,999,16,235,150,999]];uv=[f4(128),f4(90),f4(999),f4(128),f4(240),f4(999)]
    for matrix in(0,1):
        for limited in(0,1):add('yuv_to_rgb',f'yuv-matrix{matrix}-range{limited}',params={4:3,5:limited,6:matrix},a=y,b=uv,c=uv)
    add('resize','resize-odd-pitched',params={4:3,5:3,6:4})
    add('undistort','identity-undistort',params={4:2,5:2,6:1,7:1})
    add('undistort','distortion-outside',params={4:2,5:2,6:1,7:1,8:10})
    depths=[f4(v)for v in [0,2,float('nan'),99,1,2,3,99,float('inf'),-1,4,99]]
    add('invalid_depth','invalid-depth-nan',params={4:1,5:.1,6:3},a=depths)
    add('bilateral','bilateral-invalid',params={4:1,5:.1},a=depths)
    add('depth_register','identity-registration',params={4:3,5:3,6:4,7:1,8:2,9:2,10:1,11:1,12:2,13:2,14:1,15:1},a=depths,state=[bits(float('inf'))]*12)
    add('depth_register','nearest-depth-collision',w=2,h=1,stride=2,params={4:1,5:1,6:1,7:1,8:100,9:100,10:0,11:0,12:1,13:1,14:0,15:0},a=[f4(2),f4(1)],state=[bits(float('inf'))])
    add('depth_resolve','resolve-invalid',state=[bits(v)for v in [float('inf'),2,0,99,1,3,float('nan'),99,4,0,1,99]])
    add('metrics','quality-metrics',params={4:.05,5:.95,6:4},a=[f4(x/2,y/2,.5,1)for y in range(3)for x in range(4)],b=depths)
    add('metrics','quality-nan',params={4:.05,5:.95,6:4},a=[f4(float('nan'),0,0,1)]+[f4(.5,.5,.5,1)for _ in range(11)],b=depths)
    add('reduce','quality-reduction',a=[f4(.5,1,1,1)for _ in range(12)])
    add('backproject','optical-backproject',params={4:2,5:2,6:1,7:1,8:1},a=depths)
    add('transform_points','transform-validity',w=3,h=1,stride=3,params={19:10,23:-2,27:1},a=[f4(0,0,2,1),f4(1,2,3,1),f4(float('nan'),0,0,1)])
    add('voxel_insert','voxel-contention',w=10,h=1,stride=10,params={4:1,5:8,6:8},a=[f4(0,0,2,1)]*10,state=[0]*19)
    add('voxel_insert','voxel-capacity-overflow',w=2,h=1,stride=2,params={4:1,5:1,6:1},a=[f4(1,0,2,1),f4(999,0,2,1)],state=[key([0,0,2],1),1,0,0,0])
    coverage=[]
    for i in range(12):coverage.extend([key([i,0,2],1)if i%2 else 0,i%5 if i%2 else 0])
    add('coverage_heatmap','coverage-pitched',params={4:4},state=coverage)
    add('coverage_render','coverage-point-preview',w=3,h=1,stride=3,params={4:1},state=coverage)
    add('normalize','crop-normalize',params={4:0,5:0,6:3,7:3,8:3,9:3,12:.5,13:.5,14:.5,16:.5,17:.5,18:.5})
    features=[]
    for i in range(12):features.extend([f4(1),f4(2),f4(3),f4(999)])
    for count in(0,3):add('similarity','similarity-count'+str(count),params={4:count,5:4},a=features,b=[f4(1),f4(2),f4(3)])
    features[0]=f4(float('nan'));features[4:7]=[f4()]*3
    add('similarity','similarity-nan-zero',params={4:3,5:4},a=features,b=[f4(1),f4(2),f4(3)])
    for mode in(0,1,2):add('mask_postprocess','mask-mode'+str(mode),params={4:.5,5:mode},a=[f4(v)for v in[.5,0,float('nan'),99,0,1,0,99,.7,.2,.6,99]])
    return result
