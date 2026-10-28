// SPDX-License-Identifier: Apache-2.0
// VKL eager Brown-Conrady coefficients: k1,k2,p1,p2,k3.
float2 brownForward(float2 xy, StructuredBuffer<float> coefficients, uint base)
{
    float k1=coefficients[base], k2=coefficients[base+1], p1=coefficients[base+2];
    float p2=coefficients[base+3], k3=coefficients[base+4];
    float x=xy.x, y=xy.y, r=x*x+y*y;
    float radial=((1.0+k1*r)+k2*(r*r))+k3*(r*r*r);
    return float2((x*radial+((2.0*p1)*x)*y)+p2*(r+(2.0*x)*x),
                  (y*radial+p1*(r+(2.0*y)*y))+((2.0*p2)*x)*y);
}
float2 brownInverse(float2 distorted, StructuredBuffer<float> coefficients, uint base)
{
    float2 xy=distorted;
    [unroll] for(uint iteration=0; iteration<5; iteration++)
    {
        float2 projected=brownForward(xy,coefficients,base);
        xy=(xy+distorted)-projected;
    }
    return xy;
}
