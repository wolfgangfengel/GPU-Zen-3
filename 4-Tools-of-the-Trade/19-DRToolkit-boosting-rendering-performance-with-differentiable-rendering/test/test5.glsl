// Xyptonjtroz by nimitz (twitter: @stormoid)
// https://www.shadertoy.com/view/4ts3z2
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
// Contact the author for other licensing options

//Audio by Dave_Hoskins

#define ITR 30
#define FAR 30.
#define time iTime

/*
	Believable animated volumetric dust storm in 7 samples,
	blending each layer in based on geometry distance allows to
	render it without visible seams. 3d Triangle noise is 
	used for the dust volume.

	Also included is procedural bump mapping and glow based on
	curvature*fresnel. (see: https://www.shadertoy.com/view/Xts3WM)


	Further explanation of the dust generation (per Dave's request):
		
	The basic idea is to have layers of gradient shaded volumetric
	animated noise. The problem is when geometry is intersected
	before the ray reaches the far plane. A way to smoothly blend
	the low sampled noise is needed.  So I am blending (smoothstep)
	each dust layer based on current ray distance and the solid 
	interesction distance. I am also scaling the noise taps	as a 
	function of the current distance so that the distant dust doesn't
	appear too noisy and as a function of current height to get some
	"ground hugging" effect.
	
*/

mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}

float height(in vec2 p)
{
  p *= 0.2;
  return sin(p.y)*0.4 + sin(p.x)*0.4;
}

//smooth min (https://iquilezles.org/articles/smin)
float smin( float a, float b)
{
	float h = clamp(0.5 + 0.5*(b-a)/0.7, 0.0, 1.0);
	return mix(b, a, h) - 0.7*h*(1.0-h);
}


vec2 nmzHash22(vec2 q)
{
  uvec2 p = uvec2(ivec2(q));
  p = p*uvec2(3266489917U, 668265263U) + p.yx;
  p = p*(p.yx^(p >> 15U));
  return vec2(p^(p >> 16U))*(1.0/vec2(float(0xffffffffU)));
}

float vine(vec3 p, in float c, in float h)
{
  p.y += sin(p.z*0.2625)*2.5;
  p.x += cos(p.z*0.1575)*3.;
  vec2 q = vec2(mod(p.x, c)-c/2., p.y);
  return length(q) - h -sin(p.z*2.+sin(p.x*7.)*0.5+time*0.5)*0.13;
}

float map(vec3 p)
{
  p.y += height(p.zx);
  
  vec3 bp = p;
  vec2 hs = nmzHash22(floor(p.zx/4.));
  p.zx = mod(p.zx,4.)-2.;
  
  float d = p.y+0.5;
  p.y -= hs.x*0.4-0.15;
  p.zx += hs*1.3;
  d = smin(d, length(p)-hs.x*0.4);
  
  d = smin(d, vine(bp+vec3(1.8,0.,0),15.,.8) );
  d = smin(d, vine(bp.zyx+vec3(0.,0,17.),20.,0.75) );
  
  return d*1.1;
}

float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.002;
  float h=precis*2.0;
  float d = 0.;
  for( int i=0; i<ITR; i++ )
  {
    if( abs(h)<precis || d>FAR ) break;
    d += h;
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
  float z = 1.0;
  //float z = 1.4;
	float rz = 0.;
  vec3 bp = p;
	for (int i=0; i<=3; i++ )
	{
    vec3 dg = tri3(bp*2.);
    p += (dg+time*spd);

    bp *= 0.8;
		z *= 1.5;
		p *= 1.2;
    //bp *= 1.8;
		//z *= 1.5;
		//p *= 1.2;
    //p.xz*= m2;
        
    rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
    bp += 0.12;
    //bp += 0.14;
	}
	return rz;
}

float fogmap(in vec3 p, in float d)
{
  p.x += time*0.5;
  p.z += sin(p.x*.5);
  return triNoise3d(p*2.2/(d+20.),0.05)*(1.-smoothstep(0.,.7,p.y));
}

float rayIntersectSphere(vec3 ro, vec3 rd, vec3 center, float radius, bool reverse)
{
  float r = 0.1;
  vec3 oc = center - ro;
  float prj = dot(rd, oc);
  float oc2 = dot(oc, oc);
  float prj2 = prj * prj;
  float radius2 = radius * radius;

  float dist2 = oc2 - prj2;
  if (dist2 <= radius2) {
    float discriminant2 = radius2 - dist2;
    if (discriminant2 < 0.00001) {
      r = prj;
    }
    else {
      float discriminant = sqrt(discriminant2);
      r = reverse ? prj - discriminant : prj + discriminant;
    }
  }
  return r;
}
vec3 fog(in vec3 col, in vec3 ro, in vec3 rd, in float mt)
{
  vec3 center = vec3(0.0, 0.0, 1.0) * 20.0 + vec3(0.0, 0.0, 0.0);
  float d = 0.5;
  for(int i=0; i<3; i++)
  {
    float dist = rayIntersectSphere(ro, rd, center, 15.0 + 5.0 * float(i), false);
    vec3 pos = ro + rd * dist;
    float rz = fogmap(pos*0.3, d);
    float grd = 0.2;//clamp((rz - fogmap(pos+.8-float(i)*0.1,d))*3., 0.1, 1. );
    vec3 col2 = (vec3(.1,0.8,.5)*.5 + .5*vec3(.5, .8, 1.)*(1.7+grd))*0.55;
    col = mix(col,col2,clamp(rz*smoothstep(d-0.4,d+2.+d*.75,mt),0.,1.) );
    d *= 2.8;
    if (d>mt)break;
  }
  return col;
}

vec3 normal(in vec3 p)
{  
  vec2 e = vec2(-1., 1.)*0.005;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
    e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

float bnoise(in vec3 p)
{
  float n = sin(triNoise3d(p*.3,0.0)*11.)*0.6+0.4;
  n += sin(triNoise3d(p*1.,0.05)*40.)*0.1+0.9;
  return (n*n)*0.003;
}

vec3 bump(in vec3 p, in vec3 n, in float ds)
{
  vec2 e = vec2(.005,0);
  float n0 = bnoise(p);
  vec3 d = vec3(bnoise(p+e.xyy)-n0, bnoise(p+e.yxy)-n0, bnoise(p+e.yyx)-n0)/e.x;
  n = normalize(n-d*2.5/sqrt(ds));
  return n;
}

float shadow(in vec3 ro, in vec3 rd, in float mint, in float tmax)
{
	float res = 1.0;
  float t = mint;
  for( int i=0; i<10; i++ )
  {
  float h = map(ro + rd*t);
      res = min( res, 4.*h/t );
      t += clamp( h, 0.05, .5 );
      if(h<0.001 || t>tmax) break;
  }
  return clamp( res, 0.0, 1.0 );

}

vec3 calcLighting(in vec3 ro, in vec3 rd, in float rz, in vec3 normal, in vec3 lightPos, in vec3 lightColor, in vec3 objColor)
{
  vec3 fPos = ro + rd*rz;
  // ambient
  float ambientStrength = 0.5;
  vec3 ambient = ambientStrength * lightColor;
  
  // diffuse 
  vec3 norm = normalize(normal);
  vec3 lightDir = normalize(lightPos - fPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor;
  
  // specular
  float specularStrength = 0.5;
  vec3 viewDir = normalize(ro - fPos);
  vec3 reflectDir = reflect(-lightDir, norm);  
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  vec3 specular = specularStrength * spec * lightColor;  
      
  vec3 result = (ambient + diffuse + specular) * objColor;
  return result;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{	
	vec2 p = fragCoord.xy/iResolution.xy-0.5;
  vec2 q = fragCoord.xy/iResolution.xy;
	p.x*=iResolution.x/iResolution.y;
  vec2 mo = iMouse.xy / iResolution.xy-.5;
  mo = (mo==vec2(-.5)) ? vec2(-0.1,0.07) : mo;
	mo.x *= iResolution.x/iResolution.y;
	
	vec3 ro = vec3(0.0, 0.0, 3.6);
  ro.y -= height(ro.zx)+0.05;
  mo.x += smoothstep(0.6,1.,sin(time*.6)*0.5+0.5)-1.5;
  vec3 eyedir = normalize(vec3(cos(mo.x),mo.y*2.-0.2+sin(time*0.45*1.57)*0.1,sin(mo.x)));
  vec3 rightdir = normalize(vec3(cos(mo.x+1.5708),0.,sin(mo.x+1.5708)));
  vec3 updir = normalize(cross(rightdir,eyedir));
	vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
  vec3 ligt = normalize( vec3(.5, .05, -.2) );
  vec3 ligt2 = normalize( vec3(.5, -.1, -.2) );
    
	float rz = march(ro,rd);
	
  vec3 fogb = mix(vec3(.7,.8,.8	)*0.3, vec3(1.,1.,.77)*.95, pow(dot(rd,ligt2)+1.2, 2.5)*.25);
  fogb *= clamp(rd.y*.5+.6, 0., 1.);
  vec3 col = fogb;
  
  if ( rz < FAR )
  {
    vec3 lightPos = vec3(800.0, 1200.0, 500.0);
    vec3 lightCol = vec3(1.0, 1.0, 1.0);

    vec3 pos = ro+rz*rd;
    vec3 nor= normal( pos );
    float d = distance(pos,ro);
    nor = bump(pos,nor,d);
    float shd = shadow(pos,ligt,0.1,3.);
    float dif = clamp( dot( nor, ligt ), 0.0, 1.0 )*shd;
    vec3 brdf = vec3(0.10,0.11,0.13);
    brdf += 1.5*dif*vec3(1.00,0.90,0.7);
    col = calcLighting(ro, rd, rz, nor, lightPos, lightCol, brdf);
  }
  
  //ordinary distance fog first
  col = mix(col, fogb, smoothstep(FAR-7.,FAR,rz));
  
  //then volumetric fog
  col = fog(col, ro, rd, rz);
  
  //post
  col = pow(col,vec3(0.8));
  col *= 1.-smoothstep(0.1,2.,length(p));
    
	fragColor = vec4( col, 1.0 );
}
