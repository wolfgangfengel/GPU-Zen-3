#define vec2 float2
#define vec3 float3
#define vec4 float4
#define dvec2 double2
#define dvec3 double3
#define dvec4 double4
#define ivec2 int2
#define ivec3 int3
#define ivec4 int4
#define uvec2 uint2
#define uvec3 uint3
#define uvec4 uint4
#define bvec2 bool2
#define bvec3 bool3
#define bvec4 bool4
#define mat2 float2x2
#define mat3 float3x3
#define mat4 float4x4
#define dmat2 double2x2
#define dmat3 double3x3
#define dmat4 double4x4
#define glsl_mul(x, y) mul(y, x)
#define matrixCompMult(x, y) (x * y)
#define sampler2D Texture2D
#define samplerCube TextureCube
#define sampler3D Texture3D
#define mix lerp
#define fract frac
#define inversesqrt rsqrt
#define dFdx ddx
#define dFdy ddy
#define dFdxFine ddx_fine
#define dFdyFine ddy_fine
#define dFdxCoarse ddx_coarse
#define dFdyCoarse ddy_coarse
#define fwidthFine(p) (abs(ddx_fine(p))+abs(ddy_fine(p)))
#define fwidthCoarse(p) (abs(ddx_coarse(p))+abs(ddy_coarse(p)))
//#define mod fmod
#define mod(x, y) (x - floor(x / y) * y)
#define hlsl_attr(x) x

vec2 glsl_vec2(float arg)
{
    return vec2(arg, arg);
}
vec3 glsl_vec3(float arg)
{
    return vec3(arg, arg, arg);
}
vec4 glsl_vec4(float arg)
{
    return vec4(arg, arg, arg, arg);
}
dvec2 glsl_dvec2(double arg)
{
    return dvec2(arg, arg);
}
dvec3 glsl_dvec3(double arg)
{
    return dvec3(arg, arg, arg);
}
dvec4 glsl_dvec4(double arg)
{
    return dvec4(arg, arg, arg, arg);
}
ivec2 glsl_ivec2(int arg)
{
    return ivec2(arg, arg);
}
ivec3 glsl_ivec3(int arg)
{
    return ivec3(arg, arg, arg);
}
ivec4 glsl_ivec4(int arg)
{
    return ivec4(arg, arg, arg, arg);
}
uvec2 glsl_uvec2(uint arg)
{
    return uvec2(arg, arg);
}
uvec3 glsl_uvec3(uint arg)
{
    return uvec3(arg, arg, arg);
}
uvec4 glsl_uvec4(uint arg)
{
    return uvec4(arg, arg, arg, arg);
}
bvec2 glsl_bvec2(bool arg)
{
    return bvec2(arg, arg);
}
bvec3 glsl_bvec3(bool arg)
{
    return bvec3(arg, arg, arg);
}
bvec4 glsl_bvec4(bool arg)
{
    return bvec4(arg, arg, arg, arg);
}

mat2 glsl_mat2(float arg)
{
    return mat2(
        arg, 0.0, 
        0.0, arg
        );
}
mat3 glsl_mat3(float arg)
{
    return mat3(
        arg, 0.0, 0.0, 
        0.0, arg, 0.0, 
        0.0, 0.0, arg
        );
}
mat4 glsl_mat4(float arg)
{
    return mat4(
        arg, 0.0, 0.0, 0.0,
        0.0, arg, 0.0, 0.0,
        0.0, 0.0, arg, 0.0,
        0.0, 0.0, 0.0, arg
        );
}

dmat2 glsl_dmat2(double arg)
{
    return dmat2(
        arg, 0.0, 
        0.0, arg
        );
}
dmat3 glsl_dmat3(double arg)
{
    return dmat3(
        arg, 0.0, 0.0, 
        0.0, arg, 0.0, 
        0.0, 0.0, arg
        );
}
dmat4 glsl_dmat4(double arg)
{
    return dmat4(
        arg, 0.0, 0.0, 0.0,
        0.0, arg, 0.0, 0.0,
        0.0, 0.0, arg, 0.0,
        0.0, 0.0, 0.0, arg
        );
}

mat2 glsl_inverse(mat2 m) {
    float a = m[0][0];
    float b = m[0][1];
    float c = m[1][0];
    float d = m[1][1];

    float dA = a * d - b * c;
    mat2 rm = {b, -b, -c, a};    
    rm = dA != 0.0 ? rm / dA : rm;
    return rm;
}
mat3 glsl_inverse(mat3 m) {
    float a11 = m[0][0];
    float a12 = m[0][1];
    float a13 = m[0][2];
    float a21 = m[1][0];
    float a22 = m[1][1];
    float a23 = m[1][2];
    float a31 = m[2][0];
    float a32 = m[2][1];
    float a33 = m[2][2];

    float dA = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31 - a12*a21*a33 - a11*a23*a32;
    mat3 rm = { a22*a33-a23*a32, -(a12*a33-a13*a32), a12*a23-a13*a22, -(a21*a33-a23*a31), a11*a33-a13*a31, -(a11*a23-a13*a21), a21*a32-a22*a31, -(a11*a32-a12*a31), a11*a22-a12*a21 };
    rm = dA != 0.0 ? rm / dA : rm;
    return rm;
}
mat4 glsl_inverse(mat4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    mat4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

dmat2 glsl_inverse(dmat2 m) {
    double a = m[0][0];
    double b = m[0][1];
    double c = m[1][0];
    double d = m[1][1];

    double dA = a * d - b * c;
    dmat2 rm = {b, -b, -c, a};    
    rm = dA != 0.0 ? rm / dA : rm;
    return rm;
}
dmat3 glsl_inverse(dmat3 m) {
    double a11 = m[0][0];
    double a12 = m[0][1];
    double a13 = m[0][2];
    double a21 = m[1][0];
    double a22 = m[1][1];
    double a23 = m[1][2];
    double a31 = m[2][0];
    double a32 = m[2][1];
    double a33 = m[2][2];

    double dA = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31 - a12*a21*a33 - a11*a23*a32;
    dmat3 rm = { a22*a33-a23*a32, -(a12*a33-a13*a32), a12*a23-a13*a22, -(a21*a33-a23*a31), a11*a33-a13*a31, -(a11*a23-a13*a21), a21*a32-a22*a31, -(a11*a32-a12*a31), a11*a22-a12*a21 };
    rm = dA != 0.0 ? rm / dA : rm;
    return rm;
}
dmat4 glsl_inverse(dmat4 m) {
    double n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    double n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    double n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    double n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    double t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    double t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    double t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    double t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    double det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    double idet = 1.0f / det;

    dmat4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

#include "glsl_autogen.h"
