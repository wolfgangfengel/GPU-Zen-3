//cloud city by eiffie
#define size iResolution 
#define time iTime 
 
float focalDistance=1.0,aperture=0.01,shadowCone=0.3; 
 
float Rect(in vec3 z, vec3 r){return max(abs(z.x)-r.x,max(abs(z.y)-r.y,abs(z.z)-r.z));} 
#define TAO 6.283 
void Kaleido(inout vec2 v,float power){float a=floor(.5+atan(v.x,-v.y)*power/TAO)*TAO/power;v=cos(a)*v+sin(a)*vec2(v.y,-v.x);} 
 
// See https://iquilezles.org/articles/morenoise 
float hash(float n) {return fract(sin(n) * 4378.54533);} 
float noyz(vec3 x) { 
 vec3 p=floor(x),j=fract(x); 
 const float tw=7.0,tx=13.0; 
 float n=p.x+p.y*tw+p.z*tx; 
 float a=hash(n),b=hash(n+1.0),c=hash(n+tw),d=hash(n+tw+1.0); 
 float e=hash(n+tx),f=hash(n+1.0+tx),g=hash(n+tw+tx),h=hash(n+1.0+tw+tx); 
 vec3 u=j*j*(3.0-2.0*j); 
 return mix(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,e+(f-e)*u.x+(g-e)*u.y+(e-f-g+h)*u.x*u.y,u.z); 
} 
 
float fbm(vec3 p) { 
 float h=noyz(p); 
 h+=0.5*noyz(p*=2.3); 
 return h+0.25*noyz(p*2.3);
} 
 
const float scl=0.08; 
 
float DE(vec3 z0, inout vec4 mcol){ 
 float dW=100.0,dD=100.0; 
 float dC=fbm(z0*0.25+vec3(100.0))*0.5+sin(z0.y)*0.1+sin(z0.z*0.4)*0.1+min(z0.y*0.04+0.1,0.1); 
 vec2 v=floor(vec2(z0.x,abs(z0.z))*0.5+0.5); 
 z0.xz=clamp(z0.xz,-2.0,2.0)*2.0-z0.xz; 
 float r=length(z0.xz); 
 float dS=r-0.6; 
 if(r<1.0){ 
  float shape=0.285-v.x*0.02;//0.21-0.36 
  z0.y+=v.y*0.2; 
  vec3 z=z0*10.0; 
  dS=max(z0.y-2.5,r-max(0.11-z0.y*0.1,0.01)); 
  float y2=max(abs(abs(mod(z.y+0.5,2.0)-1.0)-0.5)-0.05,abs(z.y-7.1)-8.3); 
  float y=sin(clamp(floor(z.y)*shape,-0.4,3.4))*40.0; 
  Kaleido(z.xz,8.0+floor(y));
  dW=Rect(z,vec3(0.9+y*0.1,22.0,0.9+y*0.1))*scl; 
  dD=max(z0.y-1.37,max(y2,r*10.0-1.75-sin(clamp((z.y-0.5)*shape,-0.05,3.49))*4.0))*scl; 
  dS=min(dS,min(dW,dD)); 
 } 
 dS=min(dS,dC); 
 if(dS==dW)mcol+=vec4(0.8,0.9,0.9,1.0);//+a for reflection 
 else if(dS==dD)mcol+=vec4(0.6,0.4,0.3,0.0); 
 else if(dS==dC)mcol+=vec4(1.0,1.0,1.0,-1.0);//-a for clouds 
 else mcol+=vec4(0.7+sin(z0.y*100.0)*0.3,1.0,0.8,0.0); 
 return dS; 
} 
 
float pixelSize; 
float CircleOfConfusion(float t){//calculates the radius of the circle of confusion at length t 
 return max(abs(focalDistance-t)*aperture,pixelSize*(1.0+t)); 
} 
mat3 lookat(vec3 fw,vec3 up){ 
 fw=normalize(fw);vec3 rt=normalize(cross(fw,normalize(up)));return mat3(rt,cross(rt,fw),fw); 
} 
float linstep(float a, float b, float t){return clamp((t-a)/(b-a),0.,1.);}// i got this from knighty and/or darkbeam 
//random seed and generator 
float randStep(inout float randSeed){//a simple pseudo random number generator based on iq's hash 
 ++randSeed;
 return (0.8+0.2*fract(sin(randSeed)*4375.54531)); 
} 
 
float FuzzyShadow(vec3 ro, vec3 rd, float coneGrad, float rCoC, inout vec4 mcol, inout float randSeed){ 
 float t=rCoC*2.0,d=1.0,s=1.0; 
 for(int i=0;i<6;i++){ 
  if(s<0.1)continue; 
  float r=rCoC+t*coneGrad;//radius of cone 
  d=DE(ro+rd*t, mcol)+r*0.4; 
  s*=linstep(-r,r,d); 
  t+=abs(d)*randStep(randSeed); 
 } 
 return clamp(s*0.75+0.25,0.0,1.0); 
} 
 
void mainImage(out vec4 O, in vec2 U) {
 vec4 mcol=vec4(0.0); 
 float randSeed;
 randSeed=fract(sin(time+dot(U,vec2(9.123,13.431)))*473.719245);
 pixelSize=2.0/size.y; 
 float tim=time*0.25; 
 vec3 ro=vec3(cos(tim),sin(tim*0.7)*0.5+0.3,sin(tim))*(1.8+.5*sin(tim*.41)); 
 vec3 rd=lookat(vec3(0.0,0.6,sin(tim*2.3))-ro,vec3(0.1,1.0,0.0))*normalize(vec3((2.0*U.xy-size.xy)/size.y,2.0)); 
 vec3 L=normalize(vec3(0.5,0.75,-0.5)); 
 vec4 col=vec4(0.0);//color accumulator 
 float t=DE(ro, mcol)*randSeed*.8;//tep();//distance traveled 
 ro+=rd*t;
 for(int i=0;i<72;i++){//march loop 
  if(col.w>0.9 || t>20.0)continue;//bail if we hit a surface or go out of bounds 
  float rCoC=CircleOfConfusion(t);//calc the radius of CoC 
  float d=DE(ro, mcol); 
  float fClouds=max(0.0,-mcol.a); 
  if(d<max(rCoC,fClouds*0.5)){//if we are inside add its contribution 
   vec3 p=ro; 
   if(fClouds<0.1)p-=rd*abs(d-rCoC);//back up to border of CoC 
   vec2 v=vec2(rCoC*0.333,0.0);//use normal deltas based on CoC radius 
   vec3 N=normalize(vec3(-DE(p-v.xyy, mcol)+DE(p+v.xyy, mcol),-DE(p-v.yxy, mcol)+DE(p+v.yxy, mcol),-DE(p-v.yyx, mcol)+DE(p+v.yyx, mcol))); 
   //if(N!=N)N=-rd; 
   mcol*=0.143; 
   vec3 scol; 
   float alpha; 
   if(fClouds>0.1){//clouds 
    float dn=clamp(0.5-d,0.0,1.0);dn=dn*2.0;dn*=dn;//density 
    alpha=(1.0-col.w)*dn; 
    scol=vec3(1.0)*(0.6+dn*dot(N,L)*0.4); 
    scol+=dn*max(0.0,dot(reflect(rd,N),L))*vec3(1.0,0.5,0.0); 
     
   }else{ 
    scol=mcol.rgb*(0.2+0.4*(1.0+dot(N,L))); 
    scol+=0.5*pow(max(0.0,dot(reflect(rd,N),L)),32.0)*vec3(1.0,0.5,0.0); 
    if(d<rCoC*0.25 && mcol.a>0.9){//reflect the ray if we hit a bulb "directly enough" 
     rd=reflect(rd,N);d=-rCoC*0.25;ro=p;t+=1.0; 
    } 
    scol*=FuzzyShadow(p,L,shadowCone,rCoC,mcol,randSeed); 
    alpha=(1.0-col.w)*linstep(-rCoC,rCoC,-d-0.5*rCoC);//calculate the mix like cloud density 
   } 
   col+=vec4(scol*alpha,alpha);//blend in the new color 
  } 
  mcol=vec4(0.0);//clear the color trap 
  d=abs(d+0.33*rCoC)*randStep(randSeed);//add in noise to reduce banding and create fuzz 
  ro+=d*rd;//march 
  t+=d; 
 }//mix in background color 
 vec3 scol=vec3(0.4,0.5,0.6)+rd*0.05+pow(max(0.0,dot(rd,L)),100.0)*vec3(1.0,0.75,0.5); 
 col.rgb+=scol*(1.0-clamp(col.w,0.0,1.0)); 
 
 O = vec4(clamp(col.rgb,0.0,1.0),1.0); 
} 