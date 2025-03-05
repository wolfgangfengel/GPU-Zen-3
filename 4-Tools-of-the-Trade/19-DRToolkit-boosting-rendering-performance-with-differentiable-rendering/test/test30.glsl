#define SHADOW
#define AMBIENT_OCCLUSION

#define T iTime
#define PI 3.141592653
#define TAU (PI*2.)
#define D2R (PI/180.)

#define P 0.001  // Precision
#define D 20.    // Max distance
#define S 256     // Marching steps
#define R 1.     // Marching substeps
#define K 32.    // Shadow softness
#define A 4.     // AO steps

mat2 rot(float a){float c=cos(a),s=sin(a);return mat2(c,s,-s,c);}
void rotX(inout vec3 p,vec3 o,mat2 m){p-=o;p.yz*=m;p+=o;}
void rotY(inout vec3 p,vec3 o,mat2 m){p-=o;p.xz*=m;p+=o;}
void rotZ(inout vec3 p,vec3 o,mat2 m){p-=o;p.xy*=m;p+=o;}

struct RotRange
{
    vec2 x,y,z;
};

/* =============== */
/* === Rigging === */
/* =============== */

// joint position (based on: http://image.shutterstock.com/z/stock-vector-female-body-from-three-angles-276920123.jpg)
vec3 _p_head      = vec3(   0, 1300, 0) / 1400.;
vec3 _p_shoulderL = vec3(-126, 1150, 0) / 1400.;
vec3 _p_shoulderR = vec3( 126, 1150, 0) / 1400.;
vec3 _p_elbowL    = vec3(-143,  915, 0) / 1400.;
vec3 _p_elbowR    = vec3( 143,  915, 0) / 1400.;
vec3 _p_wristL    = vec3(-143,  680, 0) / 1400.;
vec3 _p_wristR    = vec3( 143,  680, 0) / 1400.;
vec3 _p_chest     = vec3(   0, 1045, 0) / 1400.;
vec3 _p_pelvis    = vec3(   0,  750, 0) / 1400.;
vec3 _p_legL      = vec3( -75,  750, 0) / 1400.;
vec3 _p_legR      = vec3(  75,  750, 0) / 1400.;
vec3 _p_kneeL     = vec3( -60,  400, 0) / 1400.;
vec3 _p_kneeR     = vec3(  60,  400, 0) / 1400.;
vec3 _p_footL     = vec3( -50,    0, 0) / 1400.;
vec3 _p_footR     = vec3(  50,    0, 0) / 1400.;

// joint rotation angle restraints
const RotRange _rr_shoulder = RotRange(vec2( -45, 165)*D2R, vec2( -90, 180)*D2R, vec2(  10, 120)*D2R);
const RotRange _rr_chest    = RotRange(vec2(   0,   0)*D2R, vec2( -45,  45)*D2R, vec2(   0,   0)*D2R);
const RotRange _rr_leg      = RotRange(vec2( -45, 160)*D2R, vec2(   0,   0)*D2R, vec2(   0,  90)*D2R);
const RotRange _rr_pelvis   = RotRange(vec2( -10, 160)*D2R, vec2( -20,  20)*D2R, vec2(   0,   0)*D2R);
const RotRange _rr_elbow    = RotRange(vec2(   0, 170)*D2R, vec2(   0,   0)*D2R, vec2(   0,   0)*D2R);
const RotRange _rr_wrist    = RotRange(vec2(   0,   0)*D2R, vec2(   0,   0)*D2R, vec2(   0,   0)*D2R);
const RotRange _rr_knee     = RotRange(vec2(-130,   0)*D2R, vec2(   0,   0)*D2R, vec2(   0,   0)*D2R);

// joint rotation angles
vec3 _r_shoulderL = vec3(0,0,0);
vec3 _r_shoulderR = vec3(0,0,0);
vec3 _r_chest     = vec3(0,0,0);
vec3 _r_legL      = vec3(0,0,0);
vec3 _r_legR      = vec3(0,0,0);
vec3 _r_pelvis    = vec3(0,0,0);
vec3 _r_elbowL    = vec3(0,0,0);
vec3 _r_elbowR    = vec3(0,0,0);
vec3 _r_wristL    = vec3(0,0,0);
vec3 _r_wristR    = vec3(0,0,0);
vec3 _r_kneeL     = vec3(0,0,0);
vec3 _r_kneeR     = vec3(0,0,0);

vec3 _ground = vec3(0);

void waveHands()
{
     _r_shoulderL.y = -PI/3.;
     _r_shoulderL.z = _rr_shoulder.z.x + (sin(T*4.)+1.)/2. * (_rr_shoulder.z.y - _rr_shoulder.z.x);

     _r_shoulderR.y = -PI/3.;
     _r_shoulderR.z = _rr_shoulder.z.x + (sin(T*4.)+1.)/2. * (_rr_shoulder.z.y - _rr_shoulder.z.x);
    
     _r_elbowL.x = (sin(T*4.)+1.)/2.*PI/4.;
     _r_elbowR.x = (sin(T*4.)+1.)/2.*PI/4.;
}

void doCrazyStuff()
{
    _r_shoulderL = sin(vec3(T/01.,T/01.*2.,T/01.*3.)) * PI;
    _r_shoulderR = sin(vec3(T/02.,T/02.*2.,T/02.*3.)) * PI;
    _r_chest     = sin(vec3(T/03.,T/03.*2.,T/03.*3.)) * PI;
    _r_legL      = sin(vec3(T/04.,T/04.*2.,T/04.*3.)) * PI;
    _r_legR      = sin(vec3(T/05.,T/05.*2.,T/05.*3.)) * PI;
    _r_pelvis    = sin(vec3(T/06.,T/06.*2.,T/06.*3.)) * PI;
    _r_elbowL    = sin(vec3(T/07.,T/07.*2.,T/07.*3.)) * PI;
    _r_elbowR    = sin(vec3(T/08.,T/08.*2.,T/08.*3.)) * PI;
    _r_wristL    = sin(vec3(T/09.,T/09.*2.,T/09.*3.)) * PI;
    _r_wristR    = sin(vec3(T/10.,T/10.*2.,T/10.*3.)) * PI;
    _r_kneeL     = sin(vec3(T/11.,T/11.*2.,T/11.*3.)) * PI;
    _r_kneeR     = sin(vec3(T/12.,T/12.*2.,T/12.*3.)) * PI;
}

void walk(float s)
{
    float t1 = (sin(T*s)+1.)/2.;
    float t2 = (sin(T*s-TAU/3.5)+1.)/2.;
    
    float t3 = (sin(T*s+PI)+1.)/2.;
    float t4 = (sin(T*s+PI-TAU/3.5)+1.)/2.;
    
    _r_legL.x  = t1 * _rr_leg.x.x/2. + (1.-t1) * _rr_leg.x.y/5.;
    _r_kneeL.x = t2 * _rr_knee.x.x/2.;
    
    _r_legR.x  = t3 * _rr_leg.x.x/2. + (1.-t3) * _rr_leg.x.y/5.;
    _r_kneeR.x = t4 * _rr_knee.x.x/2.;
    
    _r_shoulderL.x = t3 * _rr_shoulder.x.x/4. + (1.-t3) * _rr_shoulder.x.y/8.;
    _r_shoulderR.x = t1 * _rr_shoulder.x.x/4. + (1.-t1) * _rr_shoulder.x.y/8.;
    
    _r_elbowL.x = t4 * _rr_elbow.x.y/8.;
    _r_elbowR.x = t2 * _rr_elbow.x.y/8.;
    
    _r_pelvis.y = (_rr_pelvis.y.x + t1 * _rr_pelvis.y.y * 2.) / 2.;
    _r_chest.y  = (_rr_chest.y.x + t3 * _rr_chest.y.y * 2.) / 4.;
    
    _r_pelvis.x = -PI/128.;
    
    _ground.z = T*s/8.;
}

void rig()
{    
    mat2 m;
    
    // Legs (x-axis-rotation)
    rotX(_p_kneeL,_p_legL,m=rot(-clamp(_r_legL.x,_rr_leg.x.x,_rr_leg.x.y)));
    rotX(_p_footL,_p_legL,m);
    rotX(_p_footL,_p_kneeL,rot(-clamp(_r_kneeL.x,_rr_knee.x.x,_rr_knee.x.y)));
    
    rotX(_p_kneeR,_p_legR,m=rot(-clamp(_r_legR.x,_rr_leg.x.x,_rr_leg.x.y)));
    rotX(_p_footR,_p_legR,m);
    rotX(_p_footR,_p_kneeR,rot(-clamp(_r_kneeR.x,_rr_knee.x.x,_rr_knee.x.y)));
    
    // Arms (x-axis-rotation)
    rotX(_p_elbowL,_p_shoulderL,m=rot(-clamp(_r_shoulderL.x,_rr_shoulder.x.x,_rr_shoulder.x.y)));
    rotX(_p_wristL,_p_shoulderL,m);
    rotX(_p_wristL,_p_elbowL,rot(-clamp(_r_elbowL.x,_rr_elbow.x.x,_rr_elbow.x.y)));
    
    rotX(_p_elbowR,_p_shoulderR,m=rot(-clamp(_r_shoulderR.x,_rr_shoulder.x.x,_rr_shoulder.x.y)));
    rotX(_p_wristR,_p_shoulderR,m);
    rotX(_p_wristR,_p_elbowR,rot(-clamp(_r_elbowR.x,_rr_elbow.x.x,_rr_elbow.x.y)));
    
    // Arms (y-axis-rotation)
    rotY(_p_elbowL,_p_shoulderL,m=rot(-clamp(_r_shoulderL.y,_rr_shoulder.y.x,_rr_shoulder.y.y)));
    rotY(_p_wristL,_p_shoulderL,m);
    rotY(_p_wristL,_p_elbowL,rot(-clamp(_r_elbowL.y,_rr_elbow.y.x,_rr_elbow.x.y)));
    
    rotY(_p_elbowR,_p_shoulderR,m=rot(clamp(_r_shoulderR.y,_rr_shoulder.y.x,_rr_shoulder.y.y)));
    rotY(_p_wristR,_p_shoulderR,m);
    rotY(_p_wristR,_p_elbowR,rot(clamp(_r_elbowR.y,_rr_elbow.y.x,_rr_elbow.x.y)));
    
    // Legs (z-axis-rotation)
    rotZ(_p_kneeL,_p_legL,m=rot(clamp(_r_legL.z,_rr_leg.z.x,_rr_leg.z.y)));
    rotZ(_p_footL,_p_legL,m);
    // rotX(_p_footL,_p_kneeL,rot(clamp(_r_kneeL.x,_rr_knee.x,_rr_knee.y)));
    
    rotZ(_p_kneeR,_p_legR,m=rot(-clamp(_r_legR.z,_rr_leg.z.x,_rr_leg.z.y)));
    rotZ(_p_footR,_p_legR,m);
    // rotX(_p_footL,_p_kneeL,rot(-clamp(_r_kneeL.x,_rr_knee.x,_rr_knee.y)));
    
    // Arms (z-axis-rotation)
    rotZ(_p_elbowL,_p_shoulderL,m=rot(clamp(_r_shoulderL.z,_rr_shoulder.z.x,_rr_shoulder.z.y)));
    rotZ(_p_wristL,_p_shoulderL,m);
    // rotZ(_p_wristL,_p_elbowL,rot(clamp(_r_elbowL.z,_rr_elbow.z,_rr_elbow.w)));
    
    rotZ(_p_elbowR,_p_shoulderR,m=rot(-clamp(_r_shoulderR.z,_rr_shoulder.z.x,_rr_shoulder.z.y)));
    rotZ(_p_wristR,_p_shoulderR,m);
    // rotZ(_p_wristR,_p_elbowR,rot(-clamp(_r_elbowR.z,_rr_elbow.z.x,_rr_elbow.z.y)));

    // Pelvis (x-axis-rotation)
    rotX(_p_head,_p_pelvis,m=rot(clamp(_r_pelvis.x,_rr_pelvis.x.x,_rr_pelvis.x.y)));
    rotX(_p_shoulderL,_p_pelvis,m);
    rotX(_p_elbowL,_p_pelvis,m);
    rotX(_p_wristL,_p_pelvis,m);
    rotX(_p_shoulderR,_p_pelvis,m);
    rotX(_p_elbowR,_p_pelvis,m);
    rotX(_p_wristR,_p_pelvis,m);
    
    // Pelvis (y-axis-rotation)
    rotY(_p_legL,_p_pelvis,m=rot(clamp(_r_pelvis.y,_rr_pelvis.y.x,_rr_pelvis.y.y)));
    rotY(_p_kneeL,_p_pelvis,m);
    rotY(_p_footL,_p_pelvis,m);
    rotY(_p_legR,_p_pelvis,m);
    rotY(_p_kneeR,_p_pelvis,m);
    rotY(_p_footR,_p_pelvis,m);

    // Chest (y-axis-rotation)
    rotY(_p_head,_p_chest,m=rot(clamp(_r_chest.y,_rr_chest.y.x,_rr_chest.y.y)));
    rotY(_p_shoulderL,_p_chest,m);
    rotY(_p_elbowL,_p_chest,m);
    rotY(_p_wristL,_p_chest,m);
    rotY(_p_shoulderR,_p_chest,m);
    rotY(_p_elbowR,_p_chest,m);
    rotY(_p_wristR,_p_chest,m);
}

/* ======================== */
/* === Marching Globals === */
/* ======================== */

struct Hit {
	vec3 p;
	float t;
	float d;
	float s;
};

struct Ray {
	vec3 o;
	vec3 d;
};

struct Cam {
	vec3 p;
	vec3 t;
    vec3 u;
    float f;
}, _cam;

const int _num_objects = 3;

int _ignore = -1;

bool _ambientOccMarch = false;
bool _shadowMarch = false;
bool _normalMarch = false;

/* ================= */
/* === Utilities === */
/* ================= */

// https://iquilezles.org/articles/distfunctions
float sdEll(vec3 p,vec3 r){return (length(p/r)-1.)*min(min(r.x,r.y),r.z);}
float udBox(vec3 p,vec3 s,float r){return length(max(abs(p)-s+r,0.))-r;}
float sdBox(vec3 p,vec3 s,float r){vec3 d = abs(p)-s+r;return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.))-r;}

float sdLine(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 ab = b-a, ap = p-a;
    return length(ap-ab*clamp(dot(ap,ab)/dot(ab,ab),0.,1.))-r;
}

float sdCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    vec3 ab = b-a, ap = p-a;
    float t = clamp(dot(ap,ab)/dot(ab,ab),0.,1.);
    return length(ap-ab*t)-mix(r1,r2,t);
}

float patternCheckered(vec2 p)
{
    return mod(floor(p.x)+mod(floor(p.y),2.),2.);
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

/* ============ */
/* === Scene=== */
/* ============ */

float sdJoints(vec3 p, float r)
{
    float d = 1e10;
    
    d = min(d,length(p-_p_head)-r*2.);
    d = min(d,length(p-_p_shoulderL)-r);
    d = min(d,length(p-_p_shoulderR)-r);
    d = min(d,length(p-_p_elbowL)-r);
    d = min(d,length(p-_p_elbowR)-r);
    d = min(d,length(p-_p_wristL)-r);
    d = min(d,length(p-_p_wristR)-r);
    d = min(d,length(p-_p_legL)-r);
    d = min(d,length(p-_p_legR)-r);
    d = min(d,length(p-_p_kneeL)-r);
    d = min(d,length(p-_p_kneeR)-r);
    d = min(d,length(p-_p_footL)-r);
    d = min(d,length(p-_p_footR)-r);

    return d;
}

float sdFrame(vec3 p, float r)
{
    float d = 1e10;
    
    d = min(d,sdLine(p,_p_shoulderL,_p_shoulderR,r));
    //d = min(d,sdLine(p,_p_legL,_p_legR,r));
    
    d = min(d,sdLine(p,_p_shoulderL,_p_elbowL,r));
    //d = min(d,sdLine(p,_p_shoulderL,_p_legL,r));
    d = min(d,sdLine(p,_p_elbowL,_p_wristL,r));
    d = min(d,sdLine(p,_p_legL,_p_kneeL,r));
    d = min(d,sdLine(p,_p_kneeL,_p_footL,r));
     
    d = min(d,sdLine(p,_p_shoulderR,_p_elbowR,r));
    //d = min(d,sdLine(p,_p_shoulderR,_p_legR,r));
    d = min(d,sdLine(p,_p_elbowR,_p_wristR,r));
    d = min(d,sdLine(p,_p_legR,_p_kneeR,r));
    d = min(d,sdLine(p,_p_kneeR,_p_footR,r));

    return d;
}

float sdBody(vec3 p)
{
    float d = 1e10;

    d = min(d,sdCone(p,_p_shoulderL,_p_elbowL,.03,.025));
    d = min(d,sdCone(p,_p_elbowL,_p_wristL,.025,.02));
    
    d = min(d,sdCone(p,_p_shoulderR,_p_elbowR,.03,.027));
    d = min(d,sdCone(p,_p_elbowR,_p_wristR,.027,.02));
    
    d = min(d,sdCone(p,_p_legL,_p_kneeL,.05,.03));
    d = min(d,sdCone(p,_p_kneeL,_p_footL,.03,.025));
    d = min(d,sdCone(p,_p_legR,_p_kneeR,.05,.03));
    d = min(d,sdCone(p,_p_kneeR,_p_footR,.03,.025));

    return d;
}

float scene(vec3 p, inout float _obj[_num_objects], inout float _d)
{

	vec3 q, s; float d = 1e10;

    // Floor
	d = sdBox(p*vec3(1,1,0)+vec3(0,1.1,0),vec3(.2,.1,0),.01);
    _obj[0] = d; d = 1e10;
    
    if (_ambientOccMarch == true) _obj[0] = 1e10;
    
    p.y += 1.;

    _obj[1] = min(sdFrame(p,0.01),sdJoints(p,.025));
    _obj[2] = mod(floor(T*.25),2.) == 0. ? 1e10 : sdBody(p);

	for(int i = 0; i < _num_objects; i++)
	{
		//if (_ignore == i) continue;
		d = min(d,_obj[i]);
	}

	_d = d;
	return d;
}

/* ================ */
/* === Marching === */
/* ================ */

Ray lookAt(Cam c, vec2 uv)
{
	vec3 d = normalize(c.t - c.p);
	vec3 r = normalize(cross(d,c.u));
	vec3 u = cross(r,d);

	return Ray(c.p*c.f, normalize(uv.x*r + uv.y*u + d*c.f));
}

Hit march(Ray r, inout float _obj[_num_objects], inout float _d)
{
	float t = 0., d, s;
	vec3 p;
	
	for(int i = 0; i < S; i++)
	{
		d = scene(p = r.o + r.d*t, _obj, _d);

		if (d < P || t > D)
		{
			s = float(i);
			break;
		}

		t += d/R;
	}
    
	return Hit(p, t, d, s);
}

vec3 getNormal(vec3 p, inout float _obj[_num_objects], inout float _d)
{
    _normalMarch = true;
    
	vec2 e = vec2(P,0.);

	return normalize(vec3(
		scene(p+e.xyy, _obj, _d)-scene(p-e.xyy, _obj, _d),
		scene(p+e.yxy, _obj, _d)-scene(p-e.yxy, _obj, _d),
		scene(p+e.yyx, _obj, _d)-scene(p-e.yyx, _obj, _d)
	));
}

/* =============== */
/* === Shading === */
/* =============== */

float getShadow(vec3 light, vec3 origin, inout float _obj[_num_objects], inout float _d)
{
	_shadowMarch = true;

	vec3 d = normalize(light - origin);
	float t = 0.;
	float maxt = length(light - origin)-.1;
	float s = 1.0;
    
    const int n = S/4;

	for(int i = 0; i < n; i++)
	{
		float d = scene(origin + d * t, _obj, _d);
		if (t > maxt || t > D) { break; }
		t += d; s = min(s,d/t*K);
	}

	return s;
}

float getAmbientOcclusion(Hit h, Ray _ray, inout float _obj[_num_objects], inout float _d) 
{
    _ambientOccMarch = true;
    
    float t = 0., a = 0.;
    
	for(float i = 0.; i < A; i++)
    {
        float d = scene(h.p-_ray.d*i/A*.2, _obj, _d);
        t += d;
    }

	return clamp(t/A*20.,0.,1.);
}

vec3 getColor(Hit h, Ray _ray, vec2 _uv, inout float _obj[_num_objects], inout float _d)
{
	if (h.d > P) { return vec3(_uv.y-h.s/float(S)*2.); }

	vec3 col = vec3(0);
	vec3 n = getNormal(h.p, _obj, _d);
    vec3 light = vec3(0,5,0);
    
	float diff = max(dot(n, normalize(light-h.p)),.1);
	float spec = pow(max(dot(reflect(normalize(h.p-light),n),normalize(_cam.p-h.p)),0.),100.);
	float dist = clamp(10./exp(length(h.p-light)*.3),0.,1.);
    
    if (_d == _obj[0])
    {
        vec2 c = h.p.xz;
        if (abs(n.x) > .9) c = h.p.yz;
        else if (abs(n.z) > .9) c = h.p.xy;
            
        c.y -= _ground.z;
		col = vec3(1) - texture(iChannel0,c*.2).r * .3;
    }
    else if(_d == _obj[1])
    {
		col = vec3(1,0,0);
    }
    else if(_d == _obj[2])
    {
		col = vec3(1);
    }
    
    #ifdef SHADOW
    col *= max(getShadow(light, h.p, _obj, _d),.5);
    #endif
    
    #ifdef AMBIENT_OCCLUSION
  	col *= getAmbientOcclusion(h, _ray, _obj, _d);
    #endif

    col = mix(vec3(_uv.y-h.s/float(S)*2.),col*1.5,dist);
    return col;
}

/* ============ */
/* === Main === */
/* ============ */

float hash21(vec2 p)
{
    return fract(sin(dot(p, vec2(50159.91193,49681.51239))) * 73943.1699);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    Ray _ray;
    vec2 _uv;
    float _d, _obj[_num_objects];

    vec2 uvm = iMouse.xy/iResolution.xy;
    if (iMouse.x < 10. && iMouse.y < 10.) { uvm = vec2(.5+T*.025,.5); }
    
    _cam = Cam(vec3(.5,.5,-1), vec3(0,-.2,0),vec3(0,1,0),2.);
    _cam.p.yz *= rot((uvm.y*.25+.5)*PI + PI/1.8);
    _cam.p.xz *= rot(uvm.x*TAU - PI/4.);

    vec2 coord = fragCoord.xy;
    
    _uv = (2.*coord-iResolution.xy)/iResolution.xx;
    _ray = lookAt(_cam,_uv);
    
    //doCrazyStuff();

    if(cos(T*.5)<0.)
    waveHands();
    else
    walk(4.);
    
    rig();

    float f = 1.-length((2.0*coord-iResolution.xy)/iResolution.xy)*0.5;
	fragColor = vec4(getColor(march(_ray, _obj, _d), _ray, _uv, _obj, _d)*f,1);    
}