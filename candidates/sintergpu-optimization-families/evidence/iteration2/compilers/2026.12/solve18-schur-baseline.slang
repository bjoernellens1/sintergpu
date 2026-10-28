
StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> delta;
RWStructuredBuffer<uint> status;
struct Config {uint terms;uint activeFrames;uint dofs;float damping;};ConstantBuffer<Config> cfg;
[shader("compute")][numthreads(1,1,1)]
void main() {
 float L[324],y[18],x[18];bool valid=inputTerms[cfg.terms-1]>=float(6*(cfg.activeFrames-1));
 for(uint k=0;k<324;k++)L[k]=0;
 for(uint k=0;k<18;k++){y[k]=0;x[k]=0;}
 for(uint i=0;i<cfg.dofs;i++)for(uint j=0;j<=i;j++) {
  float value=inputTerms[i*cfg.dofs+j]+(i==j?cfg.damping:0);
  for(uint k=0;k<j;k++)value-=L[i*18+k]*L[j*18+k];
  if(i==j){if(!(value>0)||!isfinite(value))valid=false;L[i*18+j]=valid?sqrt(value):1;}
  else{L[i*18+j]=value/L[j*18+j];if(!isfinite(L[i*18+j]))valid=false;}
 }
 for(uint i=0;i<cfg.dofs;i++) {
  float value=-inputTerms[cfg.dofs*cfg.dofs+i];
  for(uint k=0;k<i;k++)value-=L[i*18+k]*y[k];
  y[i]=value/L[i*18+i];if(!isfinite(y[i]))valid=false;
 }
 for(int i=int(cfg.dofs)-1;i>=0;i--) {
  float value=y[i];for(uint k=uint(i)+1;k<cfg.dofs;k++)value-=L[k*18+uint(i)]*x[k];
  x[i]=value/L[uint(i)*18+uint(i)];if(!isfinite(x[i]))valid=false;
 }
 status[0]=valid?0:1;for(uint k=0;k<cfg.dofs;k++)delta[k]=valid?x[k]:0;
}