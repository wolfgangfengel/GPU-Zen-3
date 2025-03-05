#include "shaderlib/glsl.h"
static SamplerState s_linear_clamp_sampler;
typedef float td_float_x4[4];
td_float_x4 glsl_float_x4_ctor(float, float, float, float);
typedef vec3 td_vec3_x4[4];
td_vec3_x4 glsl_vec3_x4_ctor(vec3, vec3, vec3, vec3);
static vec3 iResolution = vec3(1.0, 1.0, 1.0 );
static float iTime = 0.0;
static float iTimeDelta = 0.0;
static float iFrameRate = 10.0;
static int iFrame = 0;
static float iChannelTime[4 ] = glsl_float_x4_ctor(0.0, 0.0, 0.0, 0.0 );
static vec3 iChannelResolution[4 ] = glsl_vec3_x4_ctor(vec3(1.0, 1.0, 1.0 ), vec3(1.0, 1.0, 1.0 ), vec3(1.0, 1.0, 1.0 ), vec3(1.0, 1.0, 1.0 ) );
static vec4 iMouse = vec4(0.0, 0.0, 0.0, 0.0 );
static vec4 iDate = vec4(0.0, 0.0, 0.0, 0.0 );
static float iSampleRate = 44100.0;
static sampler2D iChannel0;
static sampler2D iChannel1;
static sampler2D iChannel2;
static sampler2D iChannel3;
float noise(const in vec2 x )
{
	vec2 p = floor(x );
	vec2 f = fract(x );
	f = f * f * (3.0 - 2.0 * f );
	vec2 uv = (p.xy  ) + f.xy ;
	return(iChannel0.SampleLevel(s_linear_clamp_sampler, (uv + 0.5 ) / 256.0, 0.0 ).x  );
}
float noise(const in vec3 x )
{
	vec3 p = floor(x );
	vec3 f = fract(x );
	f = f * f * (3.0 - 2.0 * f );
	vec2 uv = (p.xy  + vec2(37.0, 17.0 ) * p.z  ) + f.xy ;
	vec2 rg = iChannel0.SampleLevel(s_linear_clamp_sampler, (uv + 0.5 ) / 256.0, 0.0 ).yx ;
	return(mix(rg.x , rg.y , f.z  ) );
}
mat2 rot(const in float a )
{
	return(mat2(cos(a ), sin(a ), - sin(a ), cos(a ) ) );
}
static const mat2 m2 = mat2(0.60, -0.80, 0.80, 0.60 );
static const mat3 m3 = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64 );
float fbm(in vec3 p )
{
	float f = 0.0;
	f += 0.5000 * noise(p );
	p = glsl_mul(m3, p ) * 2.02;
	f += 0.2500 * noise(p );
	p = glsl_mul(m3, p ) * 2.03;
	f += 0.1250 * noise(p );
	p = glsl_mul(m3, p ) * 2.01;
	f += 0.0625 * noise(p );
	return(f / 0.9375 );
}
float hash(in float n )
{
	return(fract(sin(n ) * 43758.5453 ) );
}
bool intersectPlane(const in vec3 ro, const in vec3 rd, const in float height, inout float dist )
{
	if(rd.y  == 0.0 )
	{
		return(false );
	} 
	float d = - (ro.y  - height ) / rd.y ;
	d = min(100000.0, d );
	if(d > 0. && d < dist )
	{
		dist = d;
		return(true );
	}
	else
	{
		return(false );
	} 
}
static vec3 lig = normalize(vec3(0.3, 0.5, 0.6 ) );
vec3 bgColor(const in vec3 rd )
{
	float sun = clamp(dot(lig, rd ), 0.0, 1.0 );
	vec3 col = vec3(0.5, 0.52, 0.55 ) - rd.y  * 0.2 * vec3(1.0, 0.8, 1.0 ) + 0.15 * 0.75;
	col += vec3(1.0, .6, 0.1 ) * pow(sun, 8.0 );
	col *= 0.95;
	return(col );
}
float cloudMap(const in vec3 p, const in float ani )
{
	vec3 r = p / (500. / (64. * 0.03 ) );
	float den = -1.8 + cos(r.y  * 5. - 4.3 );
	float f;
	vec3 q = 2.5 * r * vec3(0.75, 1.0, 0.75 ) + vec3(1.0, 2.0, 1.0 ) * ani * 0.15;
	f = 0.50000 * noise(q );
	q = q * 2.02 - vec3(-1.0, 1.0, -1.0 ) * ani * 0.15;
	f += 0.25000 * noise(q );
	q = q * 2.03 + vec3(1.0, -1.0, 1.0 ) * ani * 0.15;
	f += 0.12500 * noise(q );
	q = q * 2.01 - vec3(1.0, 1.0, -1.0 ) * ani * 0.15;
	f += 0.06250 * noise(q );
	q = q * 2.02 + vec3(1.0, 1.0, 1.0 ) * ani * 0.15;
	f += 0.03125 * noise(q );
	return(0.065 * clamp(den + 4.4 * f, 0.0, 1.0 ) );
}
vec3 raymarchClouds(const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in vec3 fgc, const in float startdist, const in float maxdist, const in float ani )
{
	float t = startdist + (500. / (64. * 0.03 ) ) * 0.02 * hash(rd.x  + 35.6987221 * rd.y  + (iTime + 285. ) );
	vec4 sum = glsl_vec4(0.0 );
	for(int i = 0; i < 32; ++ i )
	{
		if(sum.a  > 0.99 || t > maxdist ) 
		continue;
		vec3 pos = ro + t * rd;
		float a = cloudMap(pos, ani );
		float dif = clamp(0.1 + 0.8 * (a - cloudMap(pos + lig * 0.15 * (500. / (64. * 0.03 ) ), ani ) ), 0., 0.5 );
		vec4 col = vec4((1. + dif ) * fgc, a );
		col.rgb  *= col.a ;
		sum = sum + col * (1.0 - sum.a  );
		t += (0.03 * (500. / (64. * 0.03 ) ) ) + t * 0.012;
	} 
	sum.xyz  = mix(bgc, sum.xyz  / (sum.w  + 0.0001 ), sum.w  );
	return(clamp(sum.xyz , 0.0, 1.0 ) );
}
float terrainMap(const in vec3 p )
{
	return((iChannel1.SampleLevel(s_linear_clamp_sampler, (glsl_mul(- p.zx , m2 ) ) * 0.000046, 0. ).x  * 600. ) * smoothstep(820., 1000., length(p.xz  ) ) - 2. + noise(p.xz  * 0.5 ) * 15. );
}
vec3 raymarchTerrain(const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in float startdist, inout float dist )
{
	float t = startdist;
	vec4 sum = glsl_vec4(0.0 );
	bool hit = false;
	vec3 col = bgc;
	for(int i = 0; i < 20; ++ i )
	{
		if(hit ) 
		break;
		t += 8. + t / 300.;
		vec3 pos = ro + t * rd;
		if(pos.y  < terrainMap(pos ) )
		{
			hit = true;
		} 
	} 
	if(hit )
	{
		float dt = 4. + t / 400.;
		t -= dt;
		vec3 pos = ro + t * rd;
		t += (0.5 - step(pos.y , terrainMap(pos ) ) ) * dt;
		for(int j = 0; j < 2; ++ j )
		{
			pos = ro + t * rd;
			dt *= 0.5;
			t += (0.5 - step(pos.y , terrainMap(pos ) ) ) * dt;
		} 
		pos = ro + t * rd;
		vec3 dx = vec3(100. * 0.1, 0., 0. );
		vec3 dz = vec3(0., 0., 100. * 0.1 );
		vec3 normal = vec3(0., 0., 0. );
		normal.x  = (terrainMap(pos + dx ) - terrainMap(pos - dx ) ) / (200. * 0.1 );
		normal.z  = (terrainMap(pos + dz ) - terrainMap(pos - dz ) ) / (200. * 0.1 );
		normal.y  = 1.;
		normal = normalize(normal );
		col = glsl_vec3(0.2 ) + 0.7 * iChannel2.Sample(s_linear_clamp_sampler, pos.xz  * 0.01 ).xyz  * vec3(1., .9, 0.6 );
		float veg = 0.3 * fbm(pos * 0.2 ) + normal.y ;
		if(veg > 0.75 )
		{
			col = vec3(0.45, 0.6, 0.3 ) * (0.5 + 0.5 * fbm(pos * 0.5 ) ) * 0.6;
		}
		else 
		if(veg > 0.66 )
		{
			col = col * 0.6 + vec3(0.4, 0.5, 0.3 ) * (0.5 + 0.5 * fbm(pos * 0.25 ) ) * 0.3;
		} 
		col *= vec3(0.5, 0.52, 0.65 ) * vec3(1., .9, 0.8 );
		vec3 brdf = col;
		float diff = clamp(dot(normal, - lig ), 0., 1. );
		col = brdf * diff * vec3(1.0, .6, 0.1 );
		col += brdf * clamp(dot(normal, lig ), 0., 1. ) * vec3(0.8, .6, 0.5 ) * 0.8;
		col += brdf * clamp(dot(normal, vec3(0., 1., 0. ) ), 0., 1. ) * vec3(0.8, .8, 1. ) * 0.2;
		dist = t;
		t -= pos.y  * 3.5;
		col = mix(col, bgc, 1.0 - exp(-0.0000005 * t * t ) );
	} 
	return(col );
}
float waterMap(vec2 pos )
{
	vec2 posm = glsl_mul(pos, m2 );
	return(abs(fbm(vec3(8. * posm, (iTime + 285. ) ) ) - 0.5 ) * 0.1 );
}
void mainImage(out vec4 fragColor, in vec2 fragCoord )
{
	vec2 q = fragCoord.xy  / iResolution.xy ;
	vec2 p = -1.0 + 2.0 * q;
	p.x  *= iResolution.x  / iResolution.y ;
	vec3 ro = vec3(0.0, 0.5, 0.0 );
	vec3 ta = vec3(0.0, 0.45, 1.0 );
	if(iMouse.z  >= 1. )
	{
		ta.xz   =  glsl_mul(ta.xz , rot((iMouse.x  / iResolution.x  - .5 ) * 7. ) );
	} 
	ta.xz   =  glsl_mul(ta.xz , rot(mod(iTime * 0.05, 6.2831852 ) ) );
	vec3 ww = normalize(ta - ro );
	vec3 uu = normalize(cross(vec3(0.0, 1.0, 0.0 ), ww ) );
	vec3 vv = normalize(cross(ww, uu ) );
	vec3 rd = normalize(p.x  * uu + p.y  * vv + 2.5 * ww );
	float fresnel,
	refldist = 5000.,
	maxdist = 5000.;
	bool reflected = false;
	vec3 normal,
	col = bgColor(rd );
	vec3 roo = ro,
	rdo = rd,
	bgc = col;
	if(intersectPlane(ro, rd, 0., refldist ) && refldist < 200. )
	{
		ro += refldist * rd;
		vec2 coord = ro.xz ;
		float bumpfactor = 0.1 * (1. - smoothstep(0., 60., refldist ) );
		vec2 dx = vec2(0.1, 0. );
		vec2 dz = vec2(0., 0.1 );
		normal = vec3(0., 1., 0. );
		normal.x  = - bumpfactor * (waterMap(coord + dx ) - waterMap(coord - dx ) ) / (2. * 0.1 );
		normal.z  = - bumpfactor * (waterMap(coord + dz ) - waterMap(coord - dz ) ) / (2. * 0.1 );
		normal = normalize(normal );
		float ndotr = dot(normal, rd );
		fresnel = pow(1.0 - abs(ndotr ), 5. );
		rd = reflect(rd, normal );
		reflected = true;
		bgc = col = bgColor(rd );
	} 
	col = raymarchTerrain(ro, rd, col, reflected ? (800. - refldist) : (800.), maxdist );
	col = raymarchClouds(ro, rd, col, bgc, reflected ? (max(0.,min(150.,(150. - refldist)))) : (150.), maxdist, (iTime + 285. ) * 0.05 );
	if(reflected )
	{
		col = mix(col.xyz , bgc, 1.0 - exp(-0.0000005 * refldist * refldist ) );
		col *= fresnel * 0.9;
		vec3 refr = refract(rdo, normal, 1. / 1.3330 );
		intersectPlane(ro, refr, -2., refldist );
		col += mix(iChannel2.Sample(s_linear_clamp_sampler, (roo + refldist * refr ).xz  * 1.3 ).xyz  * vec3(1., .9, 0.6 ), vec3(1., .9, 0.8 ) * 0.5, clamp(refldist / 3., 0., 1. ) ) * (1. - fresnel ) * 0.125;
	} 
	col = pow(col, glsl_vec3(0.7 ) );
	col = col * col * (3.0 - 2.0 * col );
	col = mix(col, glsl_vec3(dot(col, glsl_vec3(0.33 ) ) ), -0.5 );
	col *= 0.25 + 0.75 * pow(16.0 * q.x  * q.y  * (1.0 - q.x  ) * (1.0 - q.y  ), 0.1 );
	fragColor = vec4(col, 1.0 );
} 
td_float_x4 glsl_float_x4_ctor(float v0, float v1, float v2, float v3) {
	float __arr_tmp[4] = {v0, v1, v2, v3};
	return __arr_tmp;
}
td_vec3_x4 glsl_vec3_x4_ctor(vec3 v0, vec3 v1, vec3 v2, vec3 v3) {
	vec3 __arr_tmp[4] = {v0, v1, v2, v3};
	return __arr_tmp;
}
