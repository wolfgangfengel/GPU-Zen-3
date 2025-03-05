float random2d(vec2 Seed)
{
  float randomno = fract(sin(dot(Seed, vec2(12.9898, 78.233))) * 43758.5453);
  return randomno - 0.5;
}
bool intersectBox(vec3 ro, vec3 rd, out float tnear, out float tfar)
{
  vec3 boxmin = vec3(0.f, 0.f, 50.f);
  vec3 boxmax = vec3(160.f, 120.f, 100.f);

  vec3 invR = 1.0f / rd;
  vec3 tbot = invR * (boxmin - ro);
  vec3 ttop = invR * (boxmax - ro);

  vec3 tmin = vec3(100000.0f);
  vec3 tmax = vec3(0.0f);

  tmin = min(tmin, min(ttop, tbot));
  tmax = max(tmax, max(ttop, tbot));

  tnear = max(max(tmin.x, tmin.y), tmin.z);
  tfar = min(min(tmax.x, tmax.y), tmax.z);

  if(tfar > tnear)
    return true;
  return tfar > tnear;
}
float march(in vec3 ro, in vec3 rd)
{
	float d = 50.0 + sin(1.0*(rd.x+rd.y)) * 100.0;
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
	for (int i=0; i<=3; i++ )
	{
    vec3 dg = tri3(bp*2.);
    p += (dg+iTime*spd);

    bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
    //p.xz*= m2;
    
    rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
    bp += 0.14;
	}
	return rz;
}
float sample3d(vec3 p, float power, float alpha)
{
  float n3D = texture(iChannel2, fract(p)).x * 0.8;
  float n = power * (texture(iChannel0, fract(p.xz * alpha)).x - 0.5);
  float noise = n + n3D;
  noise = triNoise3d(p, 0.2);
  return noise;
}
float fbm3d(in vec3 st, in float base, in float amp, in float ampGain, in float freq, in float freqScale, in vec3 period, in float alpha)
{
  float angle = 0.0;
  float velocity = 3.0;
  float cosv = cos(angle * 3.1415 / 180.0);
  float sinv = sin(angle * 3.1415 / 180.0);
  st.x += iTime * velocity * cosv;
  st.z += iTime * velocity * sinv;
  
  // Initial values
  float value = base;
  float amplitude = amp;
  float frequency = freq;
  //
  // Loop of octaves
  value += amplitude * sample3d(st * frequency * 0.05 + vec3(period.x, 0.0, period.z), period.y, alpha) * 0.9;
  frequency *= freqScale;
  amplitude *= ampGain;
  
  value += amplitude * sample3d(st * frequency * 0.05 + vec3(period.x, 0.0, period.z), period.y, alpha) * 0.9;
  frequency *= freqScale;
  amplitude *= ampGain;
  
  value += amplitude * sample3d(st * frequency * 0.05 + vec3(period.x, 0.0, period.z), period.y, alpha) * 0.9;
  frequency *= freqScale;
  amplitude *= ampGain;
  return value;
}


const vec2 vp = vec2(320.0, 240.0);
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 p = fragCoord.xy/iResolution.xy-0.5;
  vec2 q = fragCoord.xy/iResolution.xy;
	p.x *= iResolution.x/iResolution.y;
  vec2 mo = iMouse.xy / iResolution.xy;
  mo.y = 1.0 - mo.y;
	mo.x *= iResolution.x / iResolution.y;
    
  //vec3 eyedir = normalize(vec3(cos(mo.x),mo.y*2.-0.2,sin(mo.x)));
  //vec3 rightdir = normalize(vec3(cos(mo.x+1.5708),0.,sin(mo.x+1.5708)));
  //vec3 updir = normalize(cross(rightdir,eyedir));
	//vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
  vec3 boxmin = vec3(0.f, 0.f, 50.f);
  vec3 boxmax = vec3(160.f, 120.f, 100.f);
  vec3 center = (boxmin + boxmax) / 2.0;
  vec3 size = boxmax - boxmin;
    
	vec3 ro = vec3(80.0, 60.0*sin(iTime), 60.0*sin(iTime));
  ro = vec3(8.0, -100.0 + 200.0*sin(iTime), -100.0+200.0*sin(iTime));
  vec3 eyedir = normalize(vec3(cos(mo.x),mo.y,sin(mo.x)));
  vec3 rightdir = normalize(cross(eyedir, vec3(0.0, 1.0, 0.0)));
  vec3 updir = normalize(cross(rightdir, eyedir));
  rightdir = normalize(cross(eyedir, updir));
	vec3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
	
	float rz = march(ro,rd);
  float mind, maxD;
  bool r = intersectBox(ro, rd, mind, maxD);
  
  float cd = length(center-ro);
  float radius = min(size.x, min(size.y, size.z))/2.0;
  
  mind = cd - radius;
  float maxd = cd + radius;
  mind = mind < 0.5 ? 0.5 : mind;
  
  float centd = (mind + maxd) / 2.0;
	vec3 pt1 = ro + rd * mind;
  vec3 pt2 = ro + rd * centd;
  vec3 pt3 = ro + rd * maxd;    
  
  float scale = 0.1;
  float base = 0.3;
  float amp = 0.2;
  float ampGain = 0.3;
  float freq = 1.0;
  float freqScale = 1.9;
  vec3 period = vec3(1.0, 0.2, 1.0);
  float alpha = 1.0;
  
  float v = fbm3d(pt1 * scale, base, amp, ampGain, freq, freqScale, period, alpha);
  v += fbm3d(pt2 * scale, base, amp, ampGain, freq, freqScale, period, alpha);
  v += fbm3d(pt3 * scale, base, amp, ampGain, freq, freqScale, period, alpha);
  v /= 2.0;
  fragColor = r ? vec4(v, v, v, 1.0) : vec4(0.5, 0.5, 0.5, 1.0);
}