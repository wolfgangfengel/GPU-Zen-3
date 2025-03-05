// Nintendo Switch by jackdavenport
// All code is free to use with credit! :)
// Created 2019

//------------------------------------------------------------------------------------------------//
// Signed Distance Fields
// Source: https://iquilezles.org/articles/distfunctions
float dstPlane(vec3 p, vec4 plane) {
    return dot(p,plane.xyz) - plane.w;
}
float dstBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q,0.)) + min(max(q.x,max(q.y,q.z)),0.);
}
float dstRoundBox(vec3 p, vec3 b, float r) {
    return dstBox(p, b) - r;
}
float dstCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float dstBox2D(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q,0.)) + min(max(q.x,q.y),0.);
}

//------------------------------------------------------------------------------------------------//
// Helpful directive functions
#define rgb(r,g,b) (vec3(r,g,b)*0.00392156862) /* the number is 1/255 */
#define saturate(x) clamp(x,0.,1.)

//------------------------------------------------------------------------------------------------//
// Distance Functions/Booleans
// Some of these are from iq's website
// Source: https://iquilezles.org/articles/distfunctions
vec2 dstUnion(vec2 a, float bt, float bid) {
    if(a.x < bt) return a;
    return vec2(bt,bid);
}
float dstSubtract(float a, float b) {
    return max(a,-b);
}
float dstIntersect(float a, float b) {
    return max(a,b);
}
vec3 dstElongate(vec3 p, vec3 h) {
    return p - clamp(p, -h, h);
}

//------------------------------------------------------------------------------------------------//
// Materials/Lighting
struct Material {
    vec3 albedo;
    vec3 specular;
    vec3 emission;
    float shininess;
    float reflectivity;
};
void getMaterial(inout Material mat, in vec3 p, in vec3 n, in vec2 t);

// Thanks knarkowicz!
// Source: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

//------------------------------------------------------------------------------------------------//
// Math Functions
vec2 rot2D(vec2 p, float a) {
    float s = sin(a), c = cos(a);
    return mat2(c,s,-s,c) * p;
}
float expFog(float dist, float density) {
    return 1. - exp(-dist*density);
}

//------------------------------------------------------------------------------------------------//
// Camera Functions
void cameraPos(inout vec3 ro, in float time, in vec3 mouse, in vec2 res) {
    if(mouse.z < .5) {
    	float theta = 3.14159 * time;
    	float s = sin(theta), c = cos(theta);
    	ro.x = s * 2.;
    	ro.z = -c * 2.;
    } else {
        float yaw = 3.14159 * 2. * (mouse.x / res.x);
        float pitch = max(3.14159 * .5 * (mouse.y / res.y), .4);
        
        float sy = sin(yaw), cy = cos(yaw);
        float sp = sin(pitch), cp = cos(pitch);
        
        ro.x = sy * cp * 2.;
        ro.y = sp * 2.;
        ro.z = -cy * cp * 2.;
    }
}
void lookAt(in vec3 focalPoint, in vec3 eyePos, in vec3 upDir, inout vec3 rd) {
    vec3 f = normalize(focalPoint - eyePos);
    vec3 u = normalize(cross(f, upDir));
    vec3 v = normalize(cross(u, f));
    rd = mat3(u, v, f) * rd;
}

// Twisting Pylon by jackdavenport
// All code is free to use with credit! :)
// Created 2016
// Link to original: https://www.shadertoy.com/view/XstXW7

#define MAX_ITERATIONS 16
#define MIN_DISTANCE .001

#define LIGHT_COL vec3(1.)
#define LIGHT_DIR normalize(vec3(90.,80.,-45.))

struct Ray { vec3 ori;  vec3 dir; };
struct Dst { float dst; int id;   };
struct Hit { vec3 p;    int id;   };
    
    
Dst dstPillar(vec3 p, vec3 pos, vec3 box) {
        
    p.xz = rot2D(p.xz, (iTime + p.y) * 0.785398163);
    
    vec3    d = abs(pos - p) - box;
    float dst = min(max(d.x,max(d.y,d.z)), 0.) + length(max(d, 0.));
    
    return Dst(dst, 0);
    
}

Dst dstFloor(vec3 p, float y) {
 
    return Dst(p.y - y, 1);
    
}

Dst dstMin(Dst a, Dst b) {
 
    if(b.dst < a.dst) return b;
    return a;
    
}

Dst dstScene0(vec3 p) {
 
    Dst dst = dstPillar(p, vec3(0.), vec3(.5,2.,.5));
    dst = dstMin(dst, dstFloor(p, -2.));
    
    return dst;
    
}

Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dst scn = dstScene0(p);
        p += ray.dir * scn.dst * .75;
        
        if(scn.dst < MIN_DISTANCE) {
         
            id = scn.id;
            break;
            
        }
        
    }
    
    return Hit(p,id);
    
}
  
// Shadow code from the incredible iq
// Source: https://www.shadertoy.com/view/Xds3zN
float softShadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<24; i++ )
    {
		float h = dstScene0( ro + rd*t ).dst;
        res = min( res,32.0*h/t );
        t += clamp( h, 0.05, 0.50 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

vec3 calcNormal(vec3 p) {
 
    vec2 eps = vec2(.001,0.);
    vec3   n = vec3(dstScene0(p + eps.xyy).dst - dstScene0(p - eps.xyy).dst,
                   	dstScene0(p + eps.yxy).dst - dstScene0(p - eps.yxy).dst,
                   	dstScene0(p + eps.yyx).dst - dstScene0(p - eps.yyx).dst);
    return normalize(n);
    
}

vec3 calcLighting(vec3 n, float s, Hit scn) {
 
    float d = max(dot(LIGHT_DIR,n), 0.);
	d *= s;
    
    return LIGHT_COL * d;
    
}

vec3 getPylonDiffuse(vec3 n, float s, Hit scn) {
 
    return calcLighting(n, s, scn);
    
}

vec3 getFloorDiffuse(Hit scn) {
 
    vec2 uv = mod(scn.p.xz / 3.5, 1.);
    float s = softShadow(scn.p, LIGHT_DIR, .0015, 10.);
    
    return texture(bufferB_iChannel1, uv).xyz * calcLighting(vec3(0., 1., 0.), s, scn);
    
}

vec3 shade(Ray ray) {
 
    Hit scn  = raymarch(ray);
    vec3 col = texture(bufferB_iChannel0, ray.dir).xyz;
    
    if(scn.id == 0) {
     
        vec3 n = calcNormal(scn.p);
        vec3 r = reflect(ray.dir, n);
        
        Ray rr = Ray(scn.p + r * .001, r);
        Hit rh = raymarch(rr);
        
        float sh = softShadow(scn.p, LIGHT_DIR, .0015, 10.);
        vec3  dc = getPylonDiffuse(n, sh, scn);
        vec3 rc  = 
            rh.id == 0 ? getPylonDiffuse(calcNormal(rh.p),softShadow(scn.p, LIGHT_DIR, .0015, 10.),rh) : 
            rh.id == 1 ? getFloorDiffuse(rh) :
            texture(bufferB_iChannel0, rr.dir).xyz;
        
        vec3 s = LIGHT_COL * pow(max(dot(LIGHT_DIR,r),0.), 30.) * softShadow(scn.p, LIGHT_DIR, .0015, 10.);
        float f = mix(.1, .9, 1. - max(pow(-dot(ray.dir,n), .1), 0.));
        return mix(dc, rc, f) + s;
        
        
    } else if(scn.id == 1) {
    
        col = getFloorDiffuse(scn);
    
    }
    
    col = clamp(col,0.,1.); // make sure colours are clamped for texturing
    return col;
    
}

void fillChannel1( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;
    
    vec3 ori = vec3(0.,0.,-5.5);
    vec3 dir = vec3(uv, 1.);
    
    vec3 col = shade(Ray(ori,dir));
	fragColor = vec4(col,1.);
}

// Nintendo Switch by jackdavenport
// All code is free to use with credit! :)
// Created 2019

//------------------------------------------------------------------------------------------------//
#define MAX_ITER 64
#define MIN_DIST .001
#define MAX_DIST 30.
#define REFL_COUNT 3

//#define DEBUG_NO_LIGHT      // uncomment to disable shading
//#define DEBUG_SHOW_NORMALS  // uncomment to display normals

//------------------------------------------------------------------------------------------------//
const vec3 lightPos1 = vec3(3.,1.5,-2.);
const vec3 lightPos2 = vec3(-3.,3.5,2.);
const float lightIntensity = 1.05;
const vec3 ambientLight = vec3(.2,.2,.1);
const float consoleRot = .15;

//------------------------------------------------------------------------------------------------//
float dstJoystick(vec3 p) {
    return min(
        dstCappedCylinder(p-vec3(0.,.03,0.),.04,.06),
    	dstCappedCylinder(p-vec3(0.,.11,0.),.04,.0005)-.02);
}
float dstButtonGrid(vec3 p) {
    float d = dstCappedCylinder(p,.03,.005);
    p.x -= .12; d = min(d, dstCappedCylinder(p,.03,.005));
    p.x += .06;
    p.z -= .08; d = min(d, dstCappedCylinder(p,.03,.005));
    p.z += .15; d = min(d, dstCappedCylinder(p,.03,.005));
    return d;
}
vec2 dstScene(vec3 p) {
    vec2 dst;
    // ground
    dst = vec2(dstPlane(p, vec4(0.,1.,0.,-.05)), 0.);
    // console body
    p.y -= .04;
    p.yz = rot2D(p.yz, consoleRot);
    vec3 baseBox = vec3(.85, .025, .5);
    if(abs(p.x) <= 1.) {
    	float base = dstRoundBox(p, baseBox, .025);
        if(p.z < -.45) {
    		base = dstSubtract(base, dstRoundBox(p-vec3(0.,.015,-.5), vec3(.007,.001,.003), .025));
    		base = dstSubtract(base, dstRoundBox(p-vec3(-.09,.015,-.5), vec3(.002), .025));
    		base = dstSubtract(base, dstRoundBox(p-vec3(.09,.015,-.5), vec3(.002), .025));
        } else if(p.z > .45) {
            if(p.x > -.5 && p.x < -.16) {
                vec3 q = p-vec3(0.,0.,.53);
                q.x = mod(q.x,.07);
            	base = dstSubtract(base, dstBox(q, vec3(.045,.025,.025)));
            }
            base = dstSubtract(base, dstRoundBox(p-vec3(-.74,-.025,.53), vec3(.06,.03,.03), .01));
            base = min(base, dstRoundBox(p-vec3(-.74,-.02,.505), vec3(.06,.02,.005), .01));
            base = min(base, dstRoundBox(p-vec3(.65,.0,.52), vec3(.05,.01,.01), .01));
        }
    	dst = dstUnion(dst, base, 1.);
    }
    // joycons
    if(abs(p.x) > .8) {
        // base
    	vec3 s = vec3(.93,1.,.55);
    	float cutout = dstBox(p, baseBox * vec3(1.,3.5,1.2));
    	cutout = dstSubtract(dstCappedCylinder(dstElongate(p/s, vec3(.7,.0005,.47)), .5, .025)-.035, cutout);
    	dst = dstUnion(dst, cutout, 2.);
    	// buttons/joysticks
        float intShape = dstRoundBox(p-vec3(0.,.07,-.02), baseBox * vec3(1.,1.,.5), .3);
        if(p.x > .8) { // left controls
    		dst = dstUnion(dst, dstJoystick(p-vec3(.98,0.,.25)), 3.);  // left stick
        	dst = dstUnion(dst, dstButtonGrid(p-vec3(.93,.068,0.)), 3.);  // left buttons
            dst = dstUnion(dst, dstBox(p-vec3(.94,.043,-.22),vec3(.025)), 3.); // capture button
            dst = dstUnion(dst, dstBox(p-vec3(.94,.044,.42),vec3(.025,.025,.005)), 3.); // - button
            dst = dstUnion(dst, dstIntersect(dstBox(p-vec3(1.,0.,.44),vec3(.13,.025,.1)), intShape), 3.); // r button
        	vec3 q = p-vec3(.97,-.07,.37);
            q.xz = rot2D(q.xz, .5);
            dst = dstUnion(dst, dstIntersect(dstRoundBox(q,vec3(.1,.04,.04),.042), intShape), 3.); // zl button
        } else if(p.x < -.8) { // right controls
    		dst = dstUnion(dst, dstJoystick(p-vec3(-.99,0.,-.1)), 3.); // right stick
        	dst = dstUnion(dst, dstButtonGrid(p-vec3(-1.05,.065,.2)), 3.);  // right buttons
            dst = dstUnion(dst, dstCappedCylinder(p-vec3(-.92,.065,-.22),.03,.005), 4.); // home button
            dst = dstUnion(dst, dstBox(p-vec3(-.92,.044,.42),vec3(.025,.025,.005)), 3.); // + button
            dst = dstUnion(dst, dstBox(p-vec3(-.92,.044,.42),vec3(.005,.025,.025)), 3.);
            dst = dstUnion(dst, dstIntersect(dstBox(p-vec3(-1.,0.,.44),vec3(.13,.025,.1)), intShape), 3.); // r button
            vec3 q = p-vec3(-.97,-.07,.37);
            q.xz = rot2D(q.xz, -.5);
            dst = dstUnion(dst, dstIntersect(dstRoundBox(q,vec3(.1,.04,.04),.042), intShape), 3.); // zr button
            q = p-vec3(-.97,.006,-.42);
            q.xz = rot2D(q.xz, .525);
            dst = dstUnion(dst, dstIntersect(dstRoundBox(q,vec3(.05,.005,.04),.022), intShape), 4.); // ir sensor
        }
    }
    // end scene
    return dst;
}

vec2 raymarch(vec3 ro, vec3 rd, in float maxDist) {
    vec2 t = vec2(0.,-1.);
    for(int i = 0; i < MAX_ITER; i++) {
        vec2 d = dstScene(ro+rd*t.x);
        if(d.x < MIN_DIST || t.x >= maxDist) {
            t.y = d.y;
            break;
        }
        // multiplied to reduce visual artefacts
        // if anyone knows a way to avoid doing this, let me know :)
        t.x += d.x * .5;
    }
    return t;
}

// source: https://iquilezles.org/articles/rmshadows
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    hlsl_attr([unroll(64)])
    for( float t=mint; t < maxt; )
    {
        float h = dstScene(ro + rd*t).x;
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

vec3 calcNormal(vec3 p, float t) {
    vec2 e = vec2(MIN_DIST*t,0.);
    vec3 n = vec3(dstScene(p+e.xyy).x-dstScene(p-e.xyy).x,
                  dstScene(p+e.yxy).x-dstScene(p-e.yxy).x,
                  dstScene(p+e.yyx).x-dstScene(p-e.yyx).x);
    return normalize(n);
}

void calcLighting(vec3 p, vec3 n, vec3 rd, Material mat, out vec3 col) {
#ifndef DEBUG_NO_LIGHT
    vec3 diff = vec3(ambientLight);
    vec3 spec = vec3(0.);
    
    for(int i = 0; i < 2; i++) {
        // calc light vector and distance
    	vec3 lv = (i == 0 ? lightPos1 : lightPos2) - p;
    	float ld = length(lv);
    	lv /= ld;
    
    	// calculate shadows
    	float shadow = softshadow(p, lv, .01, ld, 8.);
    
    	// calculate lighting
    	float ndotl = max(dot(n,lv),0.);
    	diff += ndotl * shadow * lightIntensity;
    	if(dot(mat.specular,mat.specular) > 0.) {
        	vec3 h = normalize(lv - rd);
        	float ndoth = max(dot(n,h),0.);
        	spec += mat.specular * pow(ndoth, mat.shininess) * shadow * lightIntensity;
    	}
    }
        
    // output final colour
    col = mat.albedo * diff + spec + mat.emission;
#else
    col = mat.albedo + mat.emission;
#endif
}

vec3 shade(vec3 ro, vec3 rd) {
    vec3 col = vec3(0.);
    float coeff = 1.;
    
    for(int i = 0; i < REFL_COUNT; i++) {
    	vec2 scn = raymarch(ro, rd, MAX_DIST);
    
    	if(scn.y > -1. && scn.x < MAX_DIST) {
        	vec3 p = ro + rd * scn.x;
        	vec3 n = calcNormal(p, scn.x);
            
#ifndef DEBUG_SHOW_NORMALS
            vec3 a = vec3(0.);
            
        	Material mat;
        	getMaterial(mat, p, n, scn);
        	calcLighting(p, n, rd, mat, a);
        
            if(i == 0) {
                coeff *= 1.-saturate((scn.x-5.) / 7.5);
            }
            
        	if(mat.reflectivity > 0.) {
            	float fres = 1.-clamp(pow(max(-dot(rd,n),0.),.4),0.,1.);
            	fres *= mat.reflectivity;
                
                col += a * coeff * (1. - fres);
                coeff *= fres;
                
                vec3 r = normalize(reflect(rd,n));
                ro = p + r * .01;
                rd = r;
            } else {
                col += a * coeff;
                break;
            }
#else
        	col = n * .5 + .5;
        	break;
#endif
        } else if(i > 0) {
            col += texture(iChannel2, rd).xyz * coeff;
            break;
        } else {
            break;
        }
    }
    
    
    // post processing
    //col = pow(col, vec3(1.));
    col = ACESFilm(col);
    return col;
}

hlsl_attr([numthreads(8,8,1)])
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;
    vec3 ro = vec3(0.,1.2,0.);
    vec3 rd = vec3(uv, 1.);
  
    cameraPos(ro, iTime * .06, iMouse, iResolution.xy);
    lookAt(vec3(0.,0.,0.), ro, vec3(0.,1.,0.), rd);
    
    fragColor.xyz = shade(ro, normalize(rd));
    fragColor.w   = 1.;
}

//------------------------------------------------------------------------------------------------//

void getMaterial(inout Material mat, in vec3 p, in vec3 n, in vec2 t) {
    if(t.y == 0.) { // ground material
        vec4 tex = texture(iChannel0, p.xz / 3.5);
        mat.albedo = tex.xyz;
        mat.specular = vec3(.2 + .7 * tex.x);
        mat.shininess = 10. + 50. * tex.y;
        mat.reflectivity = .8;
    } else if(t.y == 1.) { // switch body
        p.yz = rot2D(p.yz, consoleRot);
        float screen = step(0.,dstBox2D(p.xz, vec2(.73,.43))-.03);
        if(screen < .5 && p.y > .05) {
        	float innerScreen = step(0.,dstBox2D(p.xz, vec2(.67,.38)));
            mat.albedo   = mix(vec3(0.), vec3(.1), innerScreen);
        	mat.emission = mix(texture(iChannel1,(p.xz*vec2(.7,1.3))+vec2(.5,.5)).xyz, vec3(0.), innerScreen);
        } else {
            mat.albedo = rgb(38,38,38);
        }
        mat.specular = vec3(mix(1.,.4,screen));
        mat.shininess = mix(60.,30.,screen);
        mat.reflectivity = 1. - .9 * screen;
    } else if(t.y == 2.) { // joycons base
        mat.albedo = mix(rgb(247, 57, 47), rgb(46, 182, 255), step(0., p.x));
        mat.specular = vec3(.2);
        mat.shininess = 20.;
        mat.reflectivity = .2;
    } else if(t.y == 3.) { // joysticks/buttons
        mat.albedo = rgb(38,38,38);
        mat.specular = vec3(.5);
        mat.shininess = 8.;
        mat.reflectivity = .1;
    } else if(t.y == 4.) { // home button
        mat.albedo = rgb(54, 53, 52);
        mat.specular = vec3(.5);
        mat.shininess = 8.;
        mat.reflectivity = .1;
    } else if(t.y == 5.) { // ir sensor
        mat.albedo = vec3(.05);
        mat.specular = vec3(1.);
        mat.shininess = 80.;
        mat.reflectivity = 1.;
    } else { // default material
        mat.albedo = vec3(1.,0.,1.);
        mat.specular = vec3(0.);
        mat.shininess = 0.;
        mat.reflectivity = 0.;
    }
    int c = 1;
    int d = 2;
    switch(c){
        case 1:
            d=1;
            break;
        case 2:
        default:
            d=2;
            break;
        case 3:
            d=3;
    }
}