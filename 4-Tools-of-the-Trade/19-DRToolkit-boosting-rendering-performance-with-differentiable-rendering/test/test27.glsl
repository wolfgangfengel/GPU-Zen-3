vec3 LIGHT_DIR = normalize(vec3(0.5, -1.0, -0.5));

// Animation constants. Initialized in initAnimation() because some compilers don't like
// non-constant global initializers.
float TIME;
float TORSO_LEAN;
float TORSO_TWIST;
float TORSO_BOUNCE;
float HEAD_BOB;
float L_LEG_SWING;
float R_LEG_SWING;
float L_KNEE_BEND;
float R_KNEE_BEND;
float L_ANKLE_BEND;
float R_ANKLE_BEND;
vec3 L_ARM_SWING;
vec3 R_ARM_SWING;
float L_ELBOW_BEND;
float R_ELBOW_BEND;

void initAnimation() {
    TIME = iTime * 6.2;
    TORSO_LEAN = -0.1;
    TORSO_TWIST = 0.15*sin(0.5+TIME);
    TORSO_BOUNCE = 0.9 * abs(sin(TIME + 0.4));
    HEAD_BOB = - 0.05 * (1.0 - (sin(2.0 * (TIME - 1.0))));
    L_LEG_SWING =  .6 * sin(TIME);
    R_LEG_SWING = -.6 * sin(TIME);
    L_KNEE_BEND = -0.8 * (1.0 + sin(TIME+1.7));
    R_KNEE_BEND = -0.8 * (1.0 - sin(TIME+1.7));
    L_ANKLE_BEND = 0.3 * (1.0 + sin(TIME+1.));
    R_ANKLE_BEND = 0.3 * (1.0 - sin(TIME+1.));
    L_ARM_SWING = vec3(-0.6 * sin(TIME), 0.1, -0.4);
    R_ARM_SWING = vec3( 0.6 * sin(TIME), -0.1,  0.4);
    L_ELBOW_BEND = mix(0.9, 1.5, 1.0 - (sin(TIME + 0.3) + 0.3 * sin(2.0 * (TIME + 0.3))));
    R_ELBOW_BEND = mix(0.9, 1.5, 1.0 + (sin(TIME + 0.3) + 0.3 * sin(2.0 * (TIME + 0.3))));   
}

float sdPlane(in vec3 p) {
    return p.y;
}

float sdSphere(in vec3 p, in float r) {
    return length(p)-r;
}

float sdCylinder( vec3 p, vec2 h ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdEllipsoid( in vec3 p, in vec3 r ) {
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

float udRoundBox( vec3 p, vec3 b, float r ) {
    return length(max(abs(p)-b,0.0))-r;
}

float smin( float a, float b, float k ) {
    float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

vec3 rotx(vec3 p, float rx) {
    float sinx = sin(rx);
    float cosx = cos(rx);
    return mat3(1., 0., 0., 0., cosx, sinx, 0., -sinx, cosx) * p;
}

vec3 roty(vec3 p, float ry) {
    float sinx = sin(ry);
    float cosx = cos(ry);
    return mat3(cosx, 0., -sinx, 0., 1., 0., sinx, 0., cosx) * p;
}

vec3 rotz(vec3 p, float rz) {
    float sinx = sin(rz);
    float cosx = cos(rz);
    return mat3(cosx, sinx, 0., -sinx, cosx, 0., 0., 0., 1.) * p;
}

vec3 rot(vec3 p, vec3 r) {
    return rotx(roty(rotz(p, r.z), r.y), r.x);
}

float sdLeg(vec3 p, float r, vec2 h, float legSwing, float kneeBend, float ankleBend) {    
    vec3 cylOffset = vec3(0.0, h.y, 0.0);
    p = rotx(p - 2.0 * cylOffset, legSwing) + 2.0 * cylOffset; // Swing upper leg.
    
    // Knee
    float d = sdSphere(p, r);
    
    // Thigh
	d = smin(d, sdCylinder(p - cylOffset, h), 0.4);
    
    p = rotx(p, kneeBend); // Swing lower leg.

    // Shin
    d = smin(d, sdCylinder(p + cylOffset, h), 0.4);
    
    vec3 ty = vec3(0.0, 2.0 * h.y, 0.0);
    p = rotx(p + ty, ankleBend); // Swing foot.

    // Foot
	vec3 tz = vec3(0.0, 0.0, -0.7);
    d = smin(d, sdEllipsoid(p + tz, vec3(0.25, 0.2, 0.7)), 0.4);
    
    return d;
}

float sdLegs(vec3 p) {
    vec2 legDimens = vec2(0.3, 1.9);
    vec3 legDisp = vec3(0.9, 0.0, 0.0);
    return min(
        sdLeg(p - legDisp, 0.1, legDimens, L_LEG_SWING, L_KNEE_BEND, L_ANKLE_BEND),
        sdLeg(p + legDisp, 0.1, legDimens, R_LEG_SWING, R_KNEE_BEND, R_ANKLE_BEND));
}

float sdArm(vec3 p, vec3 swing, float elbowBend) {
    p = rot(p, swing);
    
    // Shoulder
    float d = sdSphere(p, 0.3);
    
    // Upper arm
    vec2 upperArmDimens = vec2(0.3, 1.3);
    d = smin(d, sdCylinder(p + vec3(0.0, upperArmDimens.y, 0.0), upperArmDimens), 0.4);

    // Rotate at elbow
    p.y += 2.0 * upperArmDimens.y;
    p = rotx(p, elbowBend);

    // Elbow
    d = smin(d, sdSphere(p, 0.3), 0.4);
    
    // Lower arm
    vec2 lowerArmDimens = vec2(0.3, 1.2);
    d = smin(d, sdCylinder(p + vec3(0.0, lowerArmDimens.y, 0.0), lowerArmDimens), 0.4);
    
    // Hand
    p.y += 2.0 * lowerArmDimens.y;
    //d = smin(d, sdSphere(p, 0.3), 0.4);
    
    return d;
}

float sdArms(vec3 p) {    
    vec3 armDisp = vec3(1.4, 0.0, 0.0);
    return min(
        sdArm(p - armDisp, L_ARM_SWING, L_ELBOW_BEND),
        sdArm(p + armDisp, R_ARM_SWING, R_ELBOW_BEND));
}

float sdUpperBody(vec3 p) {
    float d = udRoundBox(p, vec3(0.7, 1.5, .0), 0.7);
    d = smin(d, sdArms(p - vec3(0.0, 1.8, 0.0)), 0.4);
    d = smin(d, sdCylinder(p - vec3(0.0, 2.5, 0.0), vec2(0.2, 0.4)), 0.4);
    
    p.y -= 2.5;
    p = rotx(p, HEAD_BOB);
    p.y -= 1.0;
    d = smin(d, sdSphere(p, 1.0), 0.4);

    return d;
}

vec2 argMin(in vec2 a, in vec2 b) {
    return a.x < b.x ? a : b;
}

float sdPerson(vec3 p) {
    float d = sdLegs(p);    
    p.y -= TORSO_BOUNCE + 4.9;
    p.z -= 0.2;
    p = rotx(p, TORSO_LEAN);
    p = roty(p, TORSO_TWIST);
    return smin(d, sdUpperBody(p), 0.5);
}

vec2 map(in vec3 p) {
    vec2 res = vec2(sdPlane(p), 1.0);
    return argMin(res, vec2(sdPerson(p - vec3(0.0, 3.95, 0.0)), 2.0));
}

vec3 calcNormal(in vec3 p)
{
    vec2 eps = vec2(0.001, 0.0);
    vec3 n = vec3(map(p+eps.xyy).x - map(p-eps.xyy).x,
                  map(p+eps.yxy).x - map(p-eps.yxy).x,
                  map(p+eps.yyx).x - map(p-eps.yyx).x);
    return normalize(n);
}

vec2 marchRay(in vec3 ro, in vec3 rd) {
    float t = 0.0;
    float precis = 0.02;
    float tmax = 15.0;
    for (int i=0; i<50; i++)
    {
        vec3 p = ro + t*rd;

        if (length(p) > 200.0) break; // Throw away points far from origin.

        vec2 res = map(p);

        if (res.x < precis) return vec2(t, res.y);
        if (res.x > tmax) break;

        t += res.x;
    }
    return vec2(-1.0);
}

vec3 render(in vec3 ro, in vec3 rd) {
    vec2 res = marchRay(ro, rd);
    if (res.y < -0.5) return vec3(0.6, 0.7, 0.9);
    
    float t = res.x;
    vec3 p = ro + t*rd;
    vec3 n = calcNormal(p);
    vec3 lightDir = normalize(vec3(0.5, -1.0, -0.5));
    
    vec3 col = vec3(0.1, 0.6, 0.8);
    if (res.y < 1.5) {
        col = vec3(0.3, 0.55, 0.3);
        if (abs(p.x) < 10.0) {
            col = vec3(mix(0.9, 0.8, step(mod(p.z + 5.4 *TIME, 20.0), 0.3)));
        }
    }
    
	float lambert = 0.2 * max(0.0, dot(n, normalize(vec3(1.0, 1.0, -1.0))));// fill
    if (marchRay(p - 0.1 * LIGHT_DIR, -LIGHT_DIR).y < -0.5) {
        lambert = max(0.0, dot(n, -LIGHT_DIR));// key
    }
    
    return col * vec3(0.3 + 0.7 * lambert);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    initAnimation();

    vec2 uv = fragCoord.xy/iResolution.xy;
    vec2 p = -1.0+2.0*uv;
    p.x *= iResolution.x/iResolution.y;

    // camera
    vec3 eye = vec3(25.0, 11.0, 25.0);
    if (iMouse.z > -1.0) {
        eye = rotx(eye, 0.3 *(iMouse.y - iMouse.w)/iResolution.y);
        eye = roty(eye, -10.0*((iMouse.x - iMouse.z)/iResolution.x));
    }
    vec3 look = vec3(0.0, 6.0, 0.0);
    vec3 up = vec3( 0.0, 1.0, 0.0 );
    vec3 w = normalize( look - eye );
    vec3 u = normalize( cross(w,up) );
    vec3 v = normalize( cross(u,w) );
    vec3 rd = normalize( p.x*u + p.y*v + 3.5*w );

    vec3 col = render( eye, rd );
    
    col = pow(col, vec3(.4545));
    
    fragColor=vec4( col, 1.0 );
}
