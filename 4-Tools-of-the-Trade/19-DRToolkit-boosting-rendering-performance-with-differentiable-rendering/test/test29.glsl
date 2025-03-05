// EisernSchild 3D library
// Copyright (c) 2022 by Denis Reischl
//
// SPDX-License-Identifier: MIT

// uses code with following rights :
// Copyright (c) Microsoft
// 
// SPDX-License-Identifier: MIT

// ############################ Math

#define PI 3.141592654f
#define gameTime iTime

// Orthographic projection : ba = (b.an)an
vec2 ortho_proj(vec2 vA, vec2 vB)
{
    vec2 vAn = normalize(vA);
    return dot(vB, vAn) * vAn;
}

vec3 ortho_proj(vec3 vA, vec3 vB)
{
    vec3 vAn = normalize(vA);
    return dot(vB, vAn) * vAn;
}

// rotate 2D
vec2 rotate(vec2 vV, float fA) 
{
	float fS = sin(fA);
	float fC = cos(fA);
	mat2 mR = mat2(fC, -fS, fS, fC);
	return mR * vV;
}

// ############################ Transform

// y rotation matrix
mat3 Rotate3dX(float fAng) 
{
  float fS = sin(fAng);
  float fC = cos(fAng);

  return mat3(
    1.0, 0.0, 0.0,
    0.0, fC, fS,
    0.0, -fS, fC
  );
}

// y rotation matrix
mat3 Rotate3dY(float fAng) 
{
  float fS = sin(fAng);
  float fC = cos(fAng);

  return mat3(
    fC, 0.0, -fS,
    0.0, 1.0, 0.0,
    fS, 0.0, fC
  );
}

// z rotation matrix
mat3 Rotate3dZ(float fAng) 
{
  float fS = sin(fAng);
  float fC = cos(fAng);

  return mat3(
    fC, fS, 0.f,
    -fS, fC, 0.f,
    0.f, 0.f, 1.f
  );
}

// rotate vector by y axis
vec3 RotateX(vec3 v, float fAng) { return Rotate3dX(fAng) * v; }
// rotate vector by y axis
vec3 RotateY(vec3 v, float fAng) { return Rotate3dY(fAng) * v; }
// rotate vector by z axis
vec3 RotateZ(vec3 v, float fAng) { return Rotate3dZ(fAng) * v; }

// provide a lookat matrix
mat4 LookAtLH(vec3 vCam, vec3 vTar, vec3 vUp)
{
    mat4 avLookAt;
    vec3 vZ = normalize(vTar - vCam);
    vec3 vX = normalize(cross(vUp, vZ));
    vec3 vY = cross(vZ, vX);
       
    avLookAt = 
    mat4(
        vec4(1., 0., 0., -vCam.x),
        vec4(0., 1., 0., -vCam.y),
        vec4(0., 0., 1., -vCam.z),
        vec4(0., 0., 0., 1.)
    ) *
    mat4(
        vec4(vX.x, vX.y, vX.z, 0.),
        vec4(vY.x, vY.y, vY.z, 0.),
        vec4(vZ.x, vZ.y, vZ.z, 0.),
        vec4(0., 0., 0., 1.)
    );
    
    return avLookAt;
}

// provide a perspective projection matrix
mat4 PerspectiveLH(vec2 vFov, vec2 vZnf)
{
    float fW = tan(vFov.x*0.5),
          fH = tan(vFov.y*0.5);
    
    mat4 avProj = mat4( 0.0 );
    avProj[0][0] = 2. * vZnf.x / fW;
    avProj[1][1] = 2. * vZnf.x / fH;
    avProj[2][2] = vZnf.y / (vZnf.y - vZnf.x);
    avProj[3][2] = 1.;
    avProj[2][3] = vZnf.x*vZnf.y/(vZnf.x - vZnf.y);
    
    return avProj;  
}

// transform a ray based on screen position, camera position and inverse wvp matrix - Microsoft method
void TransformRay(in uvec2 sIndex, in vec2 sScreenSz, in vec4 vCamPos, in mat4 sWVPrInv,
	out vec3 vOrigin, out vec3 vDirection)
{
	// center in the middle of the pixel, get screen position
	vec2 vXy = vec2(sIndex.xy) + 0.5f;
	vec2 vUv = vXy / sScreenSz.xy * 2.0 - 1.0;
	
	// unproject by inverse wvp
	vec4 vWorld = vec4(vUv, 0, 1) * sWVPrInv;

	vWorld.xyz /= vWorld.w;
	vOrigin = vCamPos.xyz;
	vDirection = normalize(vWorld.xyz - vOrigin);
}

// ############################ Lighting

// phong pointlight.. in Position Texel, Light, Camera, Material Color, Ray Direction, Normal
// inspired by Shane's lighting model
vec3 Phong_PointLight(vec3 vPosTex, vec3 vPosLig, vec3 vPosCam, vec3 cMat, vec3 vDir, vec3 vNor)
{
    const float fAmbient = .2f;
    const float fAttenuation = .005f;
        
    // light direction, distance
	vec3 vLDir = vPosTex - vPosLig;
	float fLDist = max(length(vLDir), .001f);
	vLDir /= fLDist;

	// ambient, diffuse, attenuation, specular
	float fDif = max(dot(-vLDir, vNor), 0.f);
	fDif = pow(fDif, 2.f) * .6f + pow(fDif, 4.f) * .2f + pow(fDif, 8.f) * .2f;
	float fAtt = 1. / (1. + fLDist * fLDist * fAttenuation);
	vec3 vRef = reflect(vLDir, vNor);
	float fSpec = pow(max(dot(-vDir, vRef), 0.0), 32.f);

	return clamp(((cMat * max(fDif, fAmbient)) + fSpec * .5f) * fAtt, 0.f, 1.f);
}

// ############################ Inverse Kinetics

// if exactness is given, the target may not be hit exactly when rotated
#define IK_EXACT 0

void IK_EndEffectorToTargetAngles(
    in vec3 vTar,     /* <= base to target local vector */
    in float fA,      /* <= length of bone 1 */
    in float fB,      /* <= length base->target (=length(vTar)) */
    in float fC,      /* <= length of bone 0 */
    out float fAlpha, /* <= local space rotation Z angle base joint */
    out float fBeta,  /* <= local space rotation Z angle mid joint */
    out float fGamma) /* <= local space rotation Y angle base joint */
{
    // triangle angles
    
    // arccos(b*b + c*c - a*a) / 2bc
    fAlpha = acos((fB * fB + fC * fC - fA * fA) / (2.f * fB * fC) );
    // arccos(a*a + c*c - b*b) / 2ac
    fBeta = acos((fA * fA + fC * fC - fB * fB) / (2.f * fA * fC) );
        
    // triangle angles to local angles
    
    // arctan(cy-dy / length(ad))
    fAlpha = fAlpha + atan(vTar.y/ length(vTar.xz));
    // PI - beta
    fBeta = abs(PI - fBeta);
    
    // Y rotation
    
    // -arctan(z / y)
    fGamma = -atan(vTar.z / vTar.x);
    fGamma = (vTar.x < 0.f) ? PI + fGamma: fGamma;    
}

// clamp end effector and mid joint between base joint and target
void IK_ClampToTarget(
                in    vec3 vJBaseP,  /*<= Base joint position */
                inout vec3 vJMidP,   /*<= Mid joint position */
                inout vec3 vJEEfP,   /*<= End effector joint position */
                in    vec2 sBoneL,   /*<= Bone lengths (x - base; y - endeff) */
                in    vec3 vTarget,  /*<= Target position */
                in   float fRotMid   /*<= rotate mid joint around target axis */
                )
{
    // get triangle lengths
    vec3 vTar = vTarget - vJBaseP;
    float fA = sBoneL.y;
    float fB = length(vTar);
    float fC = sBoneL.x;
    
    // target too far ?
    if ((fA + fC) <= fB)
    {
        vTar = normalize(vTar);
        vJMidP = vJBaseP + vTar * fC; 
        vJEEfP = vJMidP + vTar * fA;
        return;
    }
    
    // angles to local space rotations
    float fAlpha, fBeta, fGamma;
    
    // calculate angles
    IK_EndEffectorToTargetAngles(vTar, fA, fB, fC, fAlpha, fBeta, fGamma);

    // mid joint rotation ?
    if (fRotMid == 0.f)
    {
        // get joint positions
        vJMidP = vJBaseP + RotateY(RotateZ(vec3(sBoneL.x, 0.f, 0.f), fAlpha), fGamma); 
        vJEEfP = vJMidP + RotateY(RotateZ(vec3(sBoneL.y, 0.f, 0.f), fAlpha - fBeta), fGamma);
    }
    else
    {
        // get joint positions... in x direction
        vec3 vJMidP1 = RotateZ(vec3(sBoneL.x, 0.f, 0.f), fAlpha); 
        vJEEfP = vJMidP1 + RotateZ(vec3(sBoneL.y, 0.f, 0.f), fAlpha - fBeta);

        // rotate mid to triangle, rotate x, rotate back, rotate y
        float fAngBE = atan(vJEEfP.y / vJEEfP.x);
        vJMidP = vJBaseP + RotateY(RotateZ(RotateX(RotateZ(vJMidP1, -fAngBE), fRotMid), fAngBE), fGamma);
        #if (IK_EXACT == 0)
        vJEEfP = vJMidP + normalize(vTarget - vJMidP) * sBoneL.y;
        #elif (IK_EXACT == 1)
        vJEEfP = vJMidP + RotateY(RotateZ(RotateX(RotateZ(vJEEfP - vJMidP1, -fAngBE), fRotMid), fAngBE), fGamma);
        #else
        vJEEfP = vJBaseP + RotateY(vJEEfP, fGamma);
        #endif
    }
}

// end effector to target, align base to opposite direction
void IK_MidCenterRotate(
                inout vec3 vJBaseP,  /*<= Base joint position */
                in    vec3 vJMidP,   /*<= Mid joint position */
                inout vec3 vJEEfP,   /*<= End effector joint position */
                in    vec2 sBoneL,   /*<= Bone lengths (x - base; y - endeff) */
                in    vec3 vTarget   /*<= Target position */
                )
{
    vec3 vDir = normalize(vTarget - vJMidP);
    vJEEfP = vJMidP + vDir * sBoneL.y;
    vJBaseP = vJMidP - vDir * sBoneL.x;
}

// end effector to target
void IK_EndeffectorToTarget(
                in    vec3  vJBaseP,  /*<= Base joint position */
                inout vec3  vJEEfP,   /*<= End effector joint position */
                in    float fBoneL,   /*<= Bone length */
                in    vec3  vTarget   /*<= Target position */
                )
{
    vec3 vDir = normalize(vTarget - vJBaseP);
    vJEEfP = vJBaseP + vDir * fBoneL;
}

// ############################ Intersection

// intersection
struct Intersection
{
    // hit entry, exit (T + position)
    vec4 vTen;
    vec4 vTex;
    // normal entry, exit
    vec3 vNen;
    vec3 vNex;
};

Intersection iPlane(in vec3 vOri, in vec3 vDir)
{
    // ortho project up-origin/up-dir
    float fT = -vOri.y/vDir.y;
    vec3 vHPos = vOri + fT * vDir;
    
    return Intersection( vec4(fT, vHPos), 
                        vec4(-fT, vHPos),
                        vec3(0.f, 1.f, 0.f),
                        vec3(0.f, -1.f, 0.f));
}

Intersection iSphere(in vec3 vOri, in vec3 vDir, in vec3 vCen, float fRad)
{
    // get local origin
    vec3 vOriL = vOri - vCen;
    
    // ortho project local origin->direction
    float fOD = dot(vOriL, vDir);
    
    // square distance origin->center minus radius
    float fOR = dot(vOriL, vOriL) - fRad*fRad;
    
    // square hit center (!)
    float fTHitS = fOD*fOD - fOR;
    
    // no intersection
    Intersection sRet = Intersection( vec4(-1.f), vec4(-1.f), vec3(0.f), vec3(0.f));
    if( fTHitS < .0f ) return sRet;
    
    // hit distance to center disk
    fTHitS = sqrt(fTHitS);
    
    // hit vector + position
    /*
    sRet.vTen.x = -fOD - fTHitS;
    sRet.vTex.x = -fOD + fTHitS;
    sRet.vTen.yzw = vOri + vDir * sRet.vTen.x;
    sRet.vTex.yzw = vOri + vDir * sRet.vTex.x;
    */

    vec4 tmp1, tmp2;
    tmp1.x = -fOD - fTHitS;
    tmp2.x = -fOD + fTHitS;
    tmp1.yzw = vOri + vDir * tmp1.x;
    tmp2.yzw = vOri + vDir * tmp2.x;

    sRet.vTen = tmp1;
    sRet.vTex = tmp2;
    
    // normals
    sRet.vNen = normalize(sRet.vTen.yzw - vCen);
    sRet.vNex = normalize(sRet.vTex.yzw - vCen);

    // return with normals
    return sRet;
}

// Skeleton rig animated
// Copyright (c) 2022 by Denis Reischl
//
// SPDX-License-Identifier: MIT

/*
    "Skeleton rig animated"
    
    Here is the prototype of an animated skeleton rig based
    on inverse kinematics.
    
    See how he walks over a transparent obstacle, a good 
    example of how inverse kinematics can also be used when 
    climbing stairs or on rough terrain. 
    
    For the math behind Inverse Kinetics read the
    tutorial linked to :
    "Inverse Kinematics Tutorial"
    https://www.shadertoy.com/view/ctf3R4
*/

// here we have 16 joints and 15 bones
#define JOINTS_N  16
#define BONES_N 15

// Joint indices (hex)
//
//         9           0    - tail
//         2           1    - neck
//   5 4 3 1 6 7 8     2    - throat
//                     3..8 - arm l/r
//         0           9    - head
//       a   d         a..f - foot l/r
//       b   e
//       c   f

// joint sphere radii
const float afJRad[JOINTS_N] = float[JOINTS_N](
        .08f, .05f, .02f,                   // tail, neck, throat
        .07f, .05f, .06f, .07f, .05f, .06f, // arms
        .1f,                                // head
        .05f, .05f, .1f, .05f, .05f, .1f    // feet
        );
        
// bone joint indices
const int anJointIdc[BONES_N * 2] = int[BONES_N * 2](
    0x0, 0x1,            // 0    .... spine        (joint 0-1)
    0x1, 0x2, 0x2, 0x9,  // 1, 2 .... neck, head   (joint 1-2, 2-9)
    0x1, 0x3, 0x1, 0x6,  // 3, 4 .... shoulder l/r (joint 1-3, 1-6)
    0x3, 0x4, 0x4, 0x5,  // 5, 6 .... arm_l        (joint 3-4, 4-5)
    0x6, 0x7, 0x7, 0x8,  // 7, 8 .... arm_r        (joint 6-7, 7-8)
    0x0, 0xa, 0x0, 0xd,  // 9, a .... hip l/r      (joint 0-a, 0-d)
    0xa, 0xb, 0xb, 0xc,  // b, c .... foot_l       (joint a-b, b-c)
    0xd, 0xe, 0xe, 0xf); // d, e .... foot_r       (joint d-e, e-f)
    
// bone lengths
const float afBoneL[BONES_N] = float[BONES_N](
        .5f, .1f, .18f,        // spine, neck, head
        .2f, .2f,              // shoulders
        .3f, .15f, .3f, .15f,  // arms
        .15f, .15f,            // hips 
        .45f, .35f, .45f, .35f // feet
        );

// ray hit attribute
struct PosNorm
{
	vec3 vPosition;
	vec3 vNormal;
};

// infinite cylinder 
// by iq : https://iquilezles.org/articles/intersectors/
vec2 cylIntersect( in vec3 ro, in vec3 rd, in vec3 cb, in vec3 ca, float cr )
{
    vec3  oc = ro - cb;
    float card = dot(ca,rd);
    float caoc = dot(ca,oc);
    float a = 1.0 - card*card;
    float b = dot( oc, rd) - caoc*card;
    float c = dot( oc, oc) - caoc*caoc - cr*cr;
    float h = b*b - a*c;
    if( h<0.0 ) return vec2(-1.0); //no intersection
    h = sqrt(h);
    return vec2(-b-h,-b+h)/a;
}

// axis aligned box centered at the origin, with size boxSize 
// by iq : https://iquilezles.org/articles/intersectors/
vec2 boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize) //, out vec3 outNormal ) 
{
    vec3 m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m*ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    //outNormal = (tN>0.0) ? step(vec3(tN),t1) : // ro ouside the box
    //                       step(t2,vec3(tF));  // ro inside the box
    //outNormal *= -sign(rd);
    return vec2( tN, tF );
}

// simple checkers
float checkers_001(vec2 vUv, float fDist)
{
    return mix(0.3, smoothstep(0.005, 0.005 + fDist * 0.005, min(fract(vUv.x), fract(vUv.y))) * (max(mod(floor(vUv.x), 2.), mod(floor(vUv.y), 2.)) * .25 + .75),
        (1. - smoothstep(0.995 - fDist * 0.005, 0.995, max(fract(vUv.x), fract(vUv.y)))));
}

// simple floor (little knobs)
float floor_knobbed(vec2 vPos)
{
	float fH = 1.0;
	fH -= (max(sin(vPos.x * .5 + PI * .5) + cos(vPos.y * .5 - PI * 2.), 1.25) - 2.0) * 0.25;
	return fH;
}

// simple floor (little knobs)
vec3 floor_knobbed_nor(vec2 vPos)
{
    // perturbe by y axis
    const vec2 vEps = vec2(.1f , 0);
    const float fBumpF = .5f;
    vec3 vP = vec3(0.f, floor_knobbed(vPos), 0.f);
    vec3 vPX = vec3(0.f, floor_knobbed(vPos + vEps.xy), 0.f);
    vec3 vPZ = vec3(0.f, floor_knobbed(vPos + vEps.yx), 0.f);
    return normalize(vec3((vPX.y - vP.y) * fBumpF / vEps.x, 1., (vPZ.y - vP.y) * fBumpF  / vEps.x));
}

// simple hash float1<-float2
float hash12(vec2 vP)
{
	vec3 vP3  = fract(vec3(vP.xyx) * .1031);
    vP3 += dot(vP3, vP3.yzx + 33.33);
    return fract((vP3.x + vP3.y) * vP3.z);
}

// function from : https://www.shadertoy.com/view/4dsSzr
// By Morgan McGuire @morgan3d, http://graphicscodex.com
// Reuse permitted under the BSD license.
vec3 desertGradient(float t) {
	float s = sqrt(clamp(1.0 - (t - 0.4) / 0.6, 0.0, 1.0));
	vec3 sky = sqrt(mix(vec3(1, 1, 1), vec3(0, 0.8, 1.0), smoothstep(0.4, 0.9, t)) * vec3(s, s, 1.0));
	vec3 land = mix(vec3(0.7, 0.3, 0.0), vec3(0.85, 0.75 + max(0.8 - t * 20.0, 0.0), 0.5), (t / 0.4)*(t / 0.4));
	return clamp((t > 0.4) ? sky : land, 0.0, 1.0) * clamp(1.5 * (1.0 - abs(t - 0.4)), 0.0, 1.0);
}

void mainImage(out vec4 cOut, in vec2 vXY )
{
    //vec2 uv = vec2(0.3750, 0.4847);
    //vXY = uv * iResolution.xy;
    /// ------- KINEMATICS

    float fTme = gameTime * 3.f * .4f;

    // joint positions
    vec3 avJPos[JOINTS_N];
    
    // left or right foot standing ?
    bool bLR = (mod(floor(fTme), 2.f) < 1.f);
    // obstacle ?
    float fObstacle = (mod(fTme, 4.f) <= 2.f) ? min(sin(mod(fTme, 2.f) * PI * .4f), .2f) : 0.f;
    
    // left/right current (foot) target
    vec3 avTarget[2] = vec3[2](
        vec3(-.2f, .1f, floor(fTme)),
        vec3(.2f, fObstacle + .1f, floor(fTme))
    );
    avTarget[0] += (bLR) ? vec3(.0f, .0f, .5f) : 
                           vec3(.0f, sin(fract(fTme) * PI) * .2f, -.5f + fract(fTme) * 2.f); 
    avTarget[1] += (bLR) ? vec3(.0f, sin(fract(fTme) * PI) * .2f, -.5f + fract(fTme) * 2.f) : 
                           vec3(.0f, .0f, .5f); 

    // set tail position 
    avJPos[0] = vec3(0.f, .8f + sin(fract(fTme) * PI) * .1f, fTme);
    avJPos[0] += (!bLR) ? vec3(.0f, sin(fract(fTme) * PI) * fObstacle, 0.f) : vec3(0.f);
    
    // set hips (a, d joints - 9, a bones) XZ to right target
    IK_MidCenterRotate(
                avJPos[0xa], avJPos[0x0], avJPos[0xd],
                vec2(afBoneL[0x9], afBoneL[0xa]),
                mix(avJPos[0] + vec3(1.f, 0.f, 0.f), vec3(avTarget[0x1].x, avJPos[0].y, avTarget[1].z), .5f) 
                );
                
    // clamp feet to target ( a...f joints - b..e bones).. interchanged z<=>y
    IK_ClampToTarget(avJPos[0xa].xzy, avJPos[0xb].xzy, avJPos[0xc].xzy,
                    vec2(afBoneL[0xb], afBoneL[0xc]),
                    avTarget[0].xzy, 0.f);
    IK_ClampToTarget(avJPos[0xd].xzy, avJPos[0xe].xzy, avJPos[0xf].xzy,
                    vec2(afBoneL[0xd], afBoneL[0xe]),
                    avTarget[1].xzy, 0.f);
                    
    // spine, neck, head off tail towards up vector for now (clamp throat z<=>y)
    IK_EndeffectorToTarget(avJPos[0x0], avJPos[0x1], afBoneL[0], avJPos[0x0] + vec3(.0f, 1.f, .0f));
    IK_ClampToTarget(avJPos[0x1].xzy, avJPos[0x2].xzy, avJPos[0x9].xzy,
                    vec2(afBoneL[0x1], afBoneL[0x2]),
                    (avJPos[0x1].xzy + vec3(0.f, sin(fTme) * .1f, .25f)), PI);
                    
    // set shoulders viseverca direction to hips
    IK_MidCenterRotate(
                avJPos[0x3], avJPos[0x1], avJPos[0x6],
                vec2(afBoneL[0x3], afBoneL[0x4]),
                mix(avJPos[0x1] + vec3(1.f, 0.f, 0.f), vec3(avTarget[0x1].x, avJPos[0x1].y, avTarget[0].z), .5f) 
                );
        
    // swing the arms ... z<=>y
    IK_ClampToTarget(avJPos[0x3].xzy, avJPos[0x4].xzy, avJPos[0x5].xzy,
                    vec2(afBoneL[0x5], afBoneL[0x6]),
                    avJPos[0x3].xzy + vec3(-.1f, sin(fTme * PI) * .5f, -.35f), PI);
    IK_ClampToTarget(avJPos[0x6].xzy, avJPos[0x7].xzy, avJPos[0x8].xzy,
                    vec2(afBoneL[0x7], afBoneL[0x8]),
                    avJPos[0x6].xzy + vec3(.1f, sin(fTme * PI) * -.5f, -.35f), PI);
    
    /// ------- CAMERA
    
    // get current camera position and lookat matrix (to tail)
    float fCamDist = 4.6f;
    vec4 vCamPos = vec4(avJPos[0], 0.f) + vec4(sin(gameTime * .2f) * fCamDist, 3.5f + sin(gameTime * .3f) * .5f, cos(gameTime * .2f) * fCamDist, 0.f);
    vec3 vCamLAt = avJPos[0];
    mat4 avLookAt = LookAtLH(vCamPos.xyz, vCamLAt, vec3(0.f, 1.f, 0.f));
    
    // get projection matrix
    mat4 avProj = PerspectiveLH(vec2(radians(90.), radians(60.)), vec2(1., 1000.));
    
    /// ------- RAYTRACING
    
    // get ray
    vec3 vOri, vDir;
    TransformRay(uvec2(vXY), iResolution.xy, 
        vCamPos, inverse(avLookAt * avProj), vOri, vDir);
    
    // do raytracing
    bool bLeftRight = ((vXY.x/iResolution.x) < .5f);
    //Intersection asI[JOINTS_N + 1];
    vec4 vTen[JOINTS_N + 1];
    vec4 vTex[JOINTS_N + 1];
    vec3 vNen[JOINTS_N + 1];
    vec3 vNex[JOINTS_N + 1];
        
    Intersection asI = iPlane(vOri, vDir);
    vTen[0] = asI.vTen;
    vTex[0] = asI.vTex;
    vNen[0] = asI.vNen;
    vNex[0] = asI.vNex;
    for (int nIx = 0; nIx < JOINTS_N; nIx++) {
        asI = iSphere(vOri, vDir, avJPos[nIx], afJRad[nIx]);
        vTen[nIx + 1] = asI.vTen;
        vTex[nIx + 1] = asI.vTex;
        vNen[nIx + 1] = asI.vNen;
        vNex[nIx + 1] = asI.vNex;
    }
    
    // get lit primitive index
    int nTI = 0;
    float fTHit = vTen[0].x;
    for(int nI = 1; nI < JOINTS_N + 1; nI++)
    {
        if (vTen[nI].x > 0.)
        {
            if (fTHit > 0.)
            {
                if (vTen[nI].x < fTHit) nTI = nI;
            }
            else nTI = nI;
            
            fTHit = vTen[nTI].x;           
        }
    }
    
    // hit attributes
    PosNorm sAttr;
    sAttr.vPosition = vTen[nTI].yzw;
    sAttr.vNormal = vNen[nTI];
    
    /// ------- TEXTURE
    
    if (nTI == 0) // floor
    {
        if (abs(sAttr.vPosition.x) < .485f)
        {
            // floor with knobs
            cOut = mix(vec4(desertGradient(.1f), 1.f), vec4(1.f), max(.4f, smoothstep(.37f, .42f, abs(sAttr.vPosition.x))));
            sAttr.vNormal = floor_knobbed_nor(sAttr.vPosition.xz * 120.f);
        }
        else
        {
            // simple coloring (dessert gradient)
            float fTile = hash12(floor(sAttr.vPosition.xz * 1.5f));
            cOut = vec4(mix(desertGradient(fTile), vec3(1.f), .4f), 1.f);
            cOut.xyz *= checkers_001(sAttr.vPosition.xz * 2.f, fTHit);
        }
    }
    else
        cOut = vec4(desertGradient(.7f), 1.f);
    
    /// ------- LIGHTING 
    
    vec3 cLit = Phong_PointLight(sAttr.vPosition, vCamLAt + vec3(2.f, 5.f, .4f), vOri, cOut.xyz, vDir, sAttr.vNormal);

    // occlusion (distance, ground)
    float fOcc = (1. - fTHit * .01);
    float fOccD = 0.f;
    if (nTI == 0)
        for(int nI = 0; nI < JOINTS_N; nI++)
        {
            float fDXZ = length(sAttr.vPosition.xz - avJPos[nI].xz);
            if (fDXZ < afJRad[nI])
                fOccD = (avJPos[nI].y < afJRad[nI] + .32f) ? max(fOccD, ((afJRad[nI] - fDXZ) / (afJRad[nI] * .5f)) * (afJRad[nI] + .32f - avJPos[nI].y)) : fOccD;
        }
    fOcc -= fOccD;
    cLit *= fOcc;
    
    /// ------- BONE LASER BEAMS
    
    for(int nI = 0; nI < BONES_N; nI++)
    {
        // get infinite cylinder
        vec2 vTHitB = cylIntersect(vOri, vDir, avJPos[anJointIdc[nI*2]], normalize(avJPos[anJointIdc[nI*2+1]] - avJPos[anJointIdc[nI*2]]), .02f);
        if ((vTHitB.x > 0.f) && (vTHitB.x < fTHit))
        {
            // clamp cylinder between joints
            float fTHitB = (vTHitB.x + vTHitB.y) * .5f;
            vec3 vPB = vOri + vDir * fTHitB;
            float fBA = length(avJPos[anJointIdc[nI*2]] - vPB);
            float fBB = length(avJPos[anJointIdc[nI*2+1]] - vPB);
            float fAB = length(avJPos[anJointIdc[nI*2]] - avJPos[anJointIdc[nI*2+1]]) * .89f;
            if ((fBA < fAB) && (fBB < fAB))
                cLit = mix(cLit, vec3(.5f, 1.f, .7f), pow(abs(vTHitB.x - min(vTHitB.y, fTHit)) / .02f, 1.2f) * .1f);
        }
    }
    
    /// -------- OBSTACLE
    if ((mod(fTme, 4.f) < 2.f) && (mod(fTme, 4.f) > 1.f))
    {
        // get infinite cylinder
        vec3 vTar = vec3(.2f, .1f, .5f + floor(fTme));
        vec2 vTHitB = boxIntersection(vOri - vTar, vDir, vec3(.2f));
        if ((vTHitB.x > 0.f) && (vTHitB.x < fTHit))
        {
            // clamp cylinder
            float fTHitB = (vTHitB.x + vTHitB.y) * .5f;
            cLit = mix(cLit, vec3(.0f, .7f, 1.0f), pow(abs(vTHitB.x - min(vTHitB.y, fTHit)) / .2f, 1.2f) * .1f);
        }
    }
    
	cOut = vec4(clamp(cLit, 0.f, 1.f), 1.f);
}
