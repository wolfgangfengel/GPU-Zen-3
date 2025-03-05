
#define ITR 10
#define FAR 30.
#define time iTime

float map(vec3 p)
{
    float d = p.y+0.5;
    return d*5.5;
}

float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.02;
  float h=precis*5.0;
  float d = 0.;
  for( int i=0; i<ITR; i++ )
  {
    d = ( abs(h)<precis || d>FAR ) ? d : d + h;
    float res = map(ro+rd*d);
    h = res;
  }
	return d;
}

float tri(in float x){return abs(fract(x)-.5);}
vec3 tri3(in vec3 p){return vec3( tri(p.z+tri(p.y*1.)), tri(p.z+tri(p.x*1.)), tri(p.y+tri(p.x*1.)));}
                                 
mat2 m2 = mat2(0.970,  0.242, -0.242,  0.970);

float triNoise3d(in vec3 p, in float spd)
{
  float z=1.4;
	float rz = 0.;
  vec3 bp = p;
	for (int i=0; i<3; ++i )
	{
    vec3 dg = tri3(bp*2.);
    p += (dg+time*spd);

    bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
    p.xz *= m2;
    
    rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
    bp += 0.14;
	}
	return rz;
}

float fogmap(in vec3 p, in float d)
{
  p.x += time*1.5;
  p.z += sin(p.x*.5);
  return triNoise3d(p*2.2/(d+20.),0.2)*(1.-smoothstep(0.,.7,p.y));
}

vec3 fog(in vec3 col, in vec3 ro, in vec3 rd, in float mt)
{
  float d = .5;
  for(int i=0; i<7; i++)
  {
    vec3  pos = ro + rd*d;
    float rz = fogmap(pos, d);
    float grd =  clamp((rz - fogmap(pos+.8-float(i)*0.1,d))*3., 0.1, 1. );
    vec3 col2 = (vec3(.1,0.8,.5)*.5 + .5*vec3(.5, .8, 1.)*(1.7-grd))*0.55;
    col = d>mt ? col : mix(col,col2,clamp(rz*smoothstep(d-0.4,d+2.+d*.75,mt),0.,1.) );
    d *= 1.5+0.3;
  }
  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{	
	vec2 p = fragCoord.xy/iResolution.xy-0.5;
  vec2 q = fragCoord.xy/iResolution.xy;
	p.x*=iResolution.x/iResolution.y;
  vec2 mo = iMouse.xy / iResolution.xy-.5;
	mo.x *= iResolution.x/iResolution.y;
	
	vec3 ro = vec3(0.0,0.0,0.0);
  vec3 eyedir = normalize(vec3(cos(mo.x),mo.y,sin(mo.x)));
  vec3 rightdir = normalize(vec3(cos(mo.x+1.5708),0.,sin(mo.x+1.5708)));
  vec3 updir = normalize(cross(rightdir,eyedir));
	vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
	float rz = march(ro,rd);
	
  vec3 col = vec3(1.0,1.0,1.0);
  
  col = fog(col, ro, rd, rz);
        
	fragColor = vec4( col, 1.0 );
}
