// Rewrite unchanged result:
func(spec[nothing], float2, glsl_vec2)params(var(spec[nothing], float, arg)) {
  return <- float2(arg, arg);
};

func(spec[nothing], float3, glsl_vec3)params(var(spec[nothing], float, arg)) {
  return <- float3(arg, arg, arg);
};

func(spec[nothing], float4, glsl_vec4)params(var(spec[nothing], float, arg)) {
  return <- float4(arg, arg, arg, arg);
};

func(spec[nothing], double2, glsl_dvec2)params(var(spec[nothing], double, arg)) {
  return <- double2(arg, arg);
};

func(spec[nothing], double3, glsl_dvec3)params(var(spec[nothing], double, arg)) {
  return <- double3(arg, arg, arg);
};

func(spec[nothing], double4, glsl_dvec4)params(var(spec[nothing], double, arg)) {
  return <- double4(arg, arg, arg, arg);
};

func(spec[nothing], int2, glsl_ivec2)params(var(spec[nothing], int, arg)) {
  return <- int2(arg, arg);
};

func(spec[nothing], int3, glsl_ivec3)params(var(spec[nothing], int, arg)) {
  return <- int3(arg, arg, arg);
};

func(spec[nothing], int4, glsl_ivec4)params(var(spec[nothing], int, arg)) {
  return <- int4(arg, arg, arg, arg);
};

func(spec[nothing], uint2, glsl_uvec2)params(var(spec[nothing], uint, arg)) {
  return <- uint2(arg, arg);
};

func(spec[nothing], uint3, glsl_uvec3)params(var(spec[nothing], uint, arg)) {
  return <- uint3(arg, arg, arg);
};

func(spec[nothing], uint4, glsl_uvec4)params(var(spec[nothing], uint, arg)) {
  return <- uint4(arg, arg, arg, arg);
};

func(spec[nothing], bool2, glsl_bvec2)params(var(spec[nothing], bool, arg)) {
  return <- bool2(arg, arg);
};

func(spec[nothing], bool3, glsl_bvec3)params(var(spec[nothing], bool, arg)) {
  return <- bool3(arg, arg, arg);
};

func(spec[nothing], bool4, glsl_bvec4)params(var(spec[nothing], bool, arg)) {
  return <- bool4(arg, arg, arg, arg);
};

func(spec[nothing], float2x2, glsl_mat2)params(var(spec[nothing], float, arg)) {
  return <- float2x2(arg, 0.0, 0.0, arg);
};

func(spec[nothing], float3x3, glsl_mat3)params(var(spec[nothing], float, arg)) {
  return <- float3x3(arg, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, arg);
};

func(spec[nothing], float4x4, glsl_mat4)params(var(spec[nothing], float, arg)) {
  return <- float4x4(arg, 0.0, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, 0.0, arg);
};

func(spec[nothing], double2x2, glsl_dmat2)params(var(spec[nothing], double, arg)) {
  return <- double2x2(arg, 0.0, 0.0, arg);
};

func(spec[nothing], double3x3, glsl_dmat3)params(var(spec[nothing], double, arg)) {
  return <- double3x3(arg, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, arg);
};

func(spec[nothing], double4x4, glsl_dmat4)params(var(spec[nothing], double, arg)) {
  return <- double4x4(arg, 0.0, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, 0.0, arg, 0.0, 0.0, 0.0, 0.0, arg);
};

func(spec[nothing], float2x2, glsl_inverse)params(var(spec[nothing], float2x2, m)) {
  var(spec[nothing], float, a) = m[0][0];
  var(spec[nothing], float, b) = m[0][1];
  var(spec[nothing], float, c) = m[1][0];
  var(spec[nothing], float, d) = m[1][1];
  var(spec[nothing], float, dA) = a * d - b * c;
  var(spec[nothing], float2x2, rm) = { b, -b, -c, a };
  rm = dA != 0.0 ? (rm / dA) : (rm);
  return <- rm;
};

func(spec[nothing], float3x3, glsl_inverse)params(var(spec[nothing], float3x3, m)) {
  var(spec[nothing], float, a11) = m[0][0];
  var(spec[nothing], float, a12) = m[0][1];
  var(spec[nothing], float, a13) = m[0][2];
  var(spec[nothing], float, a21) = m[1][0];
  var(spec[nothing], float, a22) = m[1][1];
  var(spec[nothing], float, a23) = m[1][2];
  var(spec[nothing], float, a31) = m[2][0];
  var(spec[nothing], float, a32) = m[2][1];
  var(spec[nothing], float, a33) = m[2][2];
  var(spec[nothing], float, dA) = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
  var(spec[nothing], float3x3, rm) = { a22 * a33 - a23 * a32, -(a12 * a33 - a13 * a32), a12 * a23 - a13 * a22, -(a21 * a33 - a23 * a31), a11 * a33 - a13 * a31, -(a11 * a23 - a13 * a21), a21 * a32 - a22 * a31, -(a11 * a32 - a12 * a31), a11 * a22 - a12 * a21 };
  rm = dA != 0.0 ? (rm / dA) : (rm);
  return <- rm;
};

func(spec[nothing], float4x4, glsl_inverse)params(var(spec[nothing], float4x4, m)) {
  var(spec[nothing], float, n11) = m[0][0], var(spec[nothing], float, n12) = m[1][0], var(spec[nothing], float, n13) = m[2][0], var(spec[nothing], float, n14) = m[3][0];
  var(spec[nothing], float, n21) = m[0][1], var(spec[nothing], float, n22) = m[1][1], var(spec[nothing], float, n23) = m[2][1], var(spec[nothing], float, n24) = m[3][1];
  var(spec[nothing], float, n31) = m[0][2], var(spec[nothing], float, n32) = m[1][2], var(spec[nothing], float, n33) = m[2][2], var(spec[nothing], float, n34) = m[3][2];
  var(spec[nothing], float, n41) = m[0][3], var(spec[nothing], float, n42) = m[1][3], var(spec[nothing], float, n43) = m[2][3], var(spec[nothing], float, n44) = m[3][3];
  var(spec[nothing], float, t11) = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
  var(spec[nothing], float, t12) = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
  var(spec[nothing], float, t13) = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
  var(spec[nothing], float, t14) = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
  var(spec[nothing], float, det) = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
  var(spec[nothing], float, idet) = 1.0F / det;
  var(spec[nothing], float4x4, ret);
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
  return <- ret;
};

func(spec[nothing], double2x2, glsl_inverse)params(var(spec[nothing], double2x2, m)) {
  var(spec[nothing], double, a) = m[0][0];
  var(spec[nothing], double, b) = m[0][1];
  var(spec[nothing], double, c) = m[1][0];
  var(spec[nothing], double, d) = m[1][1];
  var(spec[nothing], double, dA) = a * d - b * c;
  var(spec[nothing], double2x2, rm) = { b, -b, -c, a };
  rm = dA != 0.0 ? (rm / dA) : (rm);
  return <- rm;
};

func(spec[nothing], double3x3, glsl_inverse)params(var(spec[nothing], double3x3, m)) {
  var(spec[nothing], double, a11) = m[0][0];
  var(spec[nothing], double, a12) = m[0][1];
  var(spec[nothing], double, a13) = m[0][2];
  var(spec[nothing], double, a21) = m[1][0];
  var(spec[nothing], double, a22) = m[1][1];
  var(spec[nothing], double, a23) = m[1][2];
  var(spec[nothing], double, a31) = m[2][0];
  var(spec[nothing], double, a32) = m[2][1];
  var(spec[nothing], double, a33) = m[2][2];
  var(spec[nothing], double, dA) = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
  var(spec[nothing], double3x3, rm) = { a22 * a33 - a23 * a32, -(a12 * a33 - a13 * a32), a12 * a23 - a13 * a22, -(a21 * a33 - a23 * a31), a11 * a33 - a13 * a31, -(a11 * a23 - a13 * a21), a21 * a32 - a22 * a31, -(a11 * a32 - a12 * a31), a11 * a22 - a12 * a21 };
  rm = dA != 0.0 ? (rm / dA) : (rm);
  return <- rm;
};

func(spec[nothing], double4x4, glsl_inverse)params(var(spec[nothing], double4x4, m)) {
  var(spec[nothing], double, n11) = m[0][0], var(spec[nothing], double, n12) = m[1][0], var(spec[nothing], double, n13) = m[2][0], var(spec[nothing], double, n14) = m[3][0];
  var(spec[nothing], double, n21) = m[0][1], var(spec[nothing], double, n22) = m[1][1], var(spec[nothing], double, n23) = m[2][1], var(spec[nothing], double, n24) = m[3][1];
  var(spec[nothing], double, n31) = m[0][2], var(spec[nothing], double, n32) = m[1][2], var(spec[nothing], double, n33) = m[2][2], var(spec[nothing], double, n34) = m[3][2];
  var(spec[nothing], double, n41) = m[0][3], var(spec[nothing], double, n42) = m[1][3], var(spec[nothing], double, n43) = m[2][3], var(spec[nothing], double, n44) = m[3][3];
  var(spec[nothing], double, t11) = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
  var(spec[nothing], double, t12) = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
  var(spec[nothing], double, t13) = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
  var(spec[nothing], double, t14) = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
  var(spec[nothing], double, det) = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
  var(spec[nothing], double, idet) = 1.0F / det;
  var(spec[nothing], double4x4, ret);
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
  return <- ret;
};

func(spec[nothing], float2, glsl_vec2)params(var(spec[nothing], double2, arg)) {
  return <- float2(arg);
};

func(spec[nothing], float2, glsl_vec2)params(var(spec[nothing], int2, arg)) {
  return <- float2(arg);
};

func(spec[nothing], float2, glsl_vec2)params(var(spec[nothing], uint2, arg)) {
  return <- float2(arg);
};

func(spec[nothing], float2, glsl_vec2)params(var(spec[nothing], bool2, arg)) {
  return <- float2(arg);
};

func(spec[nothing], double2, glsl_dvec2)params(var(spec[nothing], float2, arg)) {
  return <- double2(arg);
};

func(spec[nothing], double2, glsl_dvec2)params(var(spec[nothing], int2, arg)) {
  return <- double2(arg);
};

func(spec[nothing], double2, glsl_dvec2)params(var(spec[nothing], uint2, arg)) {
  return <- double2(arg);
};

func(spec[nothing], double2, glsl_dvec2)params(var(spec[nothing], bool2, arg)) {
  return <- double2(arg);
};

func(spec[nothing], int2, glsl_ivec2)params(var(spec[nothing], float2, arg)) {
  return <- int2(arg);
};

func(spec[nothing], int2, glsl_ivec2)params(var(spec[nothing], double2, arg)) {
  return <- int2(arg);
};

func(spec[nothing], int2, glsl_ivec2)params(var(spec[nothing], uint2, arg)) {
  return <- int2(arg);
};

func(spec[nothing], int2, glsl_ivec2)params(var(spec[nothing], bool2, arg)) {
  return <- int2(arg);
};

func(spec[nothing], uint2, glsl_uvec2)params(var(spec[nothing], float2, arg)) {
  return <- uint2(arg);
};

func(spec[nothing], uint2, glsl_uvec2)params(var(spec[nothing], double2, arg)) {
  return <- uint2(arg);
};

func(spec[nothing], uint2, glsl_uvec2)params(var(spec[nothing], int2, arg)) {
  return <- uint2(arg);
};

func(spec[nothing], uint2, glsl_uvec2)params(var(spec[nothing], bool2, arg)) {
  return <- uint2(arg);
};

func(spec[nothing], bool2, glsl_bvec2)params(var(spec[nothing], float2, arg)) {
  return <- bool2(arg);
};

func(spec[nothing], bool2, glsl_bvec2)params(var(spec[nothing], double2, arg)) {
  return <- bool2(arg);
};

func(spec[nothing], bool2, glsl_bvec2)params(var(spec[nothing], int2, arg)) {
  return <- bool2(arg);
};

func(spec[nothing], bool2, glsl_bvec2)params(var(spec[nothing], uint2, arg)) {
  return <- bool2(arg);
};

func(spec[nothing], float3, glsl_vec3)params(var(spec[nothing], double3, arg)) {
  return <- float3(arg);
};

func(spec[nothing], float3, glsl_vec3)params(var(spec[nothing], int3, arg)) {
  return <- float3(arg);
};

func(spec[nothing], float3, glsl_vec3)params(var(spec[nothing], uint3, arg)) {
  return <- float3(arg);
};

func(spec[nothing], float3, glsl_vec3)params(var(spec[nothing], bool3, arg)) {
  return <- float3(arg);
};

func(spec[nothing], double3, glsl_dvec3)params(var(spec[nothing], float3, arg)) {
  return <- double3(arg);
};

func(spec[nothing], double3, glsl_dvec3)params(var(spec[nothing], int3, arg)) {
  return <- double3(arg);
};

func(spec[nothing], double3, glsl_dvec3)params(var(spec[nothing], uint3, arg)) {
  return <- double3(arg);
};

func(spec[nothing], double3, glsl_dvec3)params(var(spec[nothing], bool3, arg)) {
  return <- double3(arg);
};

func(spec[nothing], int3, glsl_ivec3)params(var(spec[nothing], float3, arg)) {
  return <- int3(arg);
};

func(spec[nothing], int3, glsl_ivec3)params(var(spec[nothing], double3, arg)) {
  return <- int3(arg);
};

func(spec[nothing], int3, glsl_ivec3)params(var(spec[nothing], uint3, arg)) {
  return <- int3(arg);
};

func(spec[nothing], int3, glsl_ivec3)params(var(spec[nothing], bool3, arg)) {
  return <- int3(arg);
};

func(spec[nothing], uint3, glsl_uvec3)params(var(spec[nothing], float3, arg)) {
  return <- uint3(arg);
};

func(spec[nothing], uint3, glsl_uvec3)params(var(spec[nothing], double3, arg)) {
  return <- uint3(arg);
};

func(spec[nothing], uint3, glsl_uvec3)params(var(spec[nothing], int3, arg)) {
  return <- uint3(arg);
};

func(spec[nothing], uint3, glsl_uvec3)params(var(spec[nothing], bool3, arg)) {
  return <- uint3(arg);
};

func(spec[nothing], bool3, glsl_bvec3)params(var(spec[nothing], float3, arg)) {
  return <- bool3(arg);
};

func(spec[nothing], bool3, glsl_bvec3)params(var(spec[nothing], double3, arg)) {
  return <- bool3(arg);
};

func(spec[nothing], bool3, glsl_bvec3)params(var(spec[nothing], int3, arg)) {
  return <- bool3(arg);
};

func(spec[nothing], bool3, glsl_bvec3)params(var(spec[nothing], uint3, arg)) {
  return <- bool3(arg);
};

func(spec[nothing], float4, glsl_vec4)params(var(spec[nothing], double4, arg)) {
  return <- float4(arg);
};

func(spec[nothing], float4, glsl_vec4)params(var(spec[nothing], int4, arg)) {
  return <- float4(arg);
};

func(spec[nothing], float4, glsl_vec4)params(var(spec[nothing], uint4, arg)) {
  return <- float4(arg);
};

func(spec[nothing], float4, glsl_vec4)params(var(spec[nothing], bool4, arg)) {
  return <- float4(arg);
};

func(spec[nothing], double4, glsl_dvec4)params(var(spec[nothing], float4, arg)) {
  return <- double4(arg);
};

func(spec[nothing], double4, glsl_dvec4)params(var(spec[nothing], int4, arg)) {
  return <- double4(arg);
};

func(spec[nothing], double4, glsl_dvec4)params(var(spec[nothing], uint4, arg)) {
  return <- double4(arg);
};

func(spec[nothing], double4, glsl_dvec4)params(var(spec[nothing], bool4, arg)) {
  return <- double4(arg);
};

func(spec[nothing], int4, glsl_ivec4)params(var(spec[nothing], float4, arg)) {
  return <- int4(arg);
};

func(spec[nothing], int4, glsl_ivec4)params(var(spec[nothing], double4, arg)) {
  return <- int4(arg);
};

func(spec[nothing], int4, glsl_ivec4)params(var(spec[nothing], uint4, arg)) {
  return <- int4(arg);
};

func(spec[nothing], int4, glsl_ivec4)params(var(spec[nothing], bool4, arg)) {
  return <- int4(arg);
};

func(spec[nothing], uint4, glsl_uvec4)params(var(spec[nothing], float4, arg)) {
  return <- uint4(arg);
};

func(spec[nothing], uint4, glsl_uvec4)params(var(spec[nothing], double4, arg)) {
  return <- uint4(arg);
};

func(spec[nothing], uint4, glsl_uvec4)params(var(spec[nothing], int4, arg)) {
  return <- uint4(arg);
};

func(spec[nothing], uint4, glsl_uvec4)params(var(spec[nothing], bool4, arg)) {
  return <- uint4(arg);
};

func(spec[nothing], bool4, glsl_bvec4)params(var(spec[nothing], float4, arg)) {
  return <- bool4(arg);
};

func(spec[nothing], bool4, glsl_bvec4)params(var(spec[nothing], double4, arg)) {
  return <- bool4(arg);
};

func(spec[nothing], bool4, glsl_bvec4)params(var(spec[nothing], int4, arg)) {
  return <- bool4(arg);
};

func(spec[nothing], bool4, glsl_bvec4)params(var(spec[nothing], uint4, arg)) {
  return <- bool4(arg);
};

func(spec[nothing], int2, greaterThan)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- int2(1, 1) - int2(step(x, y));
};

func(spec[nothing], int3, greaterThan)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- int3(1, 1, 1) - int3(step(x, y));
};

func(spec[nothing], int4, greaterThan)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- int4(1, 1, 1, 1) - int4(step(x, y));
};

func(spec[nothing], int2, lessThan)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- int2(1.0, 1.0) - int2(step(y, x));
};

func(spec[nothing], int3, lessThan)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- int3(1.0, 1.0, 1.0) - int3(step(y, x));
};

func(spec[nothing], int4, lessThan)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- int4(1.0, 1.0, 1.0, 1.0) - int4(step(y, x));
};

func(spec[nothing], int2, greaterThanEqual)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- int2(step(y, x));
};

func(spec[nothing], int3, greaterThanEqual)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- int3(step(y, x));
};

func(spec[nothing], int4, greaterThanEqual)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- int4(step(y, x));
};

func(spec[nothing], int2, lessThanEqual)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- int2(step(x, y));
};

func(spec[nothing], int3, lessThanEqual)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- int3(step(x, y));
};

func(spec[nothing], int4, lessThanEqual)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- int4(step(x, y));
};

func(spec[nothing], int2, equal)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- int2(step(x, y) * step(y, x));
};

func(spec[nothing], int3, equal)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- int3(step(y, x) * step(y, x));
};

func(spec[nothing], int4, equal)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- int4(step(y, x) * step(y, x));
};

func(spec[nothing], int2, notEqual)params(var(spec[nothing], float2, x), var(spec[nothing], float2, y)) {
  return <- (int2(1, 1) - int2(step(x, y))) * (int2(1, 1) - int2(step(y, x)));
};

func(spec[nothing], int3, notEqual)params(var(spec[nothing], float3, x), var(spec[nothing], float3, y)) {
  return <- (int3(1, 1, 1) - int3(step(x, y))) * (int3(1, 1, 1) - int3(step(y, x)));
};

func(spec[nothing], int4, notEqual)params(var(spec[nothing], float4, x), var(spec[nothing], float4, y)) {
  return <- (int4(1, 1, 1, 1) - int4(step(x, y))) * (int4(1, 1, 1, 1) - int4(step(y, x)));
};

func(spec[nothing], int2, greaterThan)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- int2(1, 1) - int2(step(x, y));
};

func(spec[nothing], int3, greaterThan)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- int3(1, 1, 1) - int3(step(x, y));
};

func(spec[nothing], int4, greaterThan)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- int4(1, 1, 1, 1) - int4(step(x, y));
};

func(spec[nothing], int2, lessThan)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- int2(1.0, 1.0) - int2(step(y, x));
};

func(spec[nothing], int3, lessThan)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- int3(1.0, 1.0, 1.0) - int3(step(y, x));
};

func(spec[nothing], int4, lessThan)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- int4(1.0, 1.0, 1.0, 1.0) - int4(step(y, x));
};

func(spec[nothing], int2, greaterThanEqual)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- int2(step(y, x));
};

func(spec[nothing], int3, greaterThanEqual)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- int3(step(y, x));
};

func(spec[nothing], int4, greaterThanEqual)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- int4(step(y, x));
};

func(spec[nothing], int2, lessThanEqual)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- int2(step(x, y));
};

func(spec[nothing], int3, lessThanEqual)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- int3(step(x, y));
};

func(spec[nothing], int4, lessThanEqual)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- int4(step(x, y));
};

func(spec[nothing], int2, equal)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- int2(step(x, y) * step(y, x));
};

func(spec[nothing], int3, equal)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- int3(step(y, x) * step(y, x));
};

func(spec[nothing], int4, equal)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- int4(step(y, x) * step(y, x));
};

func(spec[nothing], int2, notEqual)params(var(spec[nothing], double2, x), var(spec[nothing], double2, y)) {
  return <- (int2(1, 1) - int2(step(x, y))) * (int2(1, 1) - int2(step(y, x)));
};

func(spec[nothing], int3, notEqual)params(var(spec[nothing], double3, x), var(spec[nothing], double3, y)) {
  return <- (int3(1, 1, 1) - int3(step(x, y))) * (int3(1, 1, 1) - int3(step(y, x)));
};

func(spec[nothing], int4, notEqual)params(var(spec[nothing], double4, x), var(spec[nothing], double4, y)) {
  return <- (int4(1, 1, 1, 1) - int4(step(x, y))) * (int4(1, 1, 1, 1) - int4(step(y, x)));
};

func(spec[nothing], int2, greaterThan)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- int2(1, 1) - int2(step(x, y));
};

func(spec[nothing], int3, greaterThan)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- int3(1, 1, 1) - int3(step(x, y));
};

func(spec[nothing], int4, greaterThan)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- int4(1, 1, 1, 1) - int4(step(x, y));
};

func(spec[nothing], int2, lessThan)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- int2(1.0, 1.0) - int2(step(y, x));
};

func(spec[nothing], int3, lessThan)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- int3(1.0, 1.0, 1.0) - int3(step(y, x));
};

func(spec[nothing], int4, lessThan)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- int4(1.0, 1.0, 1.0, 1.0) - int4(step(y, x));
};

func(spec[nothing], int2, greaterThanEqual)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- int2(step(y, x));
};

func(spec[nothing], int3, greaterThanEqual)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- int3(step(y, x));
};

func(spec[nothing], int4, greaterThanEqual)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- int4(step(y, x));
};

func(spec[nothing], int2, lessThanEqual)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- int2(step(x, y));
};

func(spec[nothing], int3, lessThanEqual)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- int3(step(x, y));
};

func(spec[nothing], int4, lessThanEqual)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- int4(step(x, y));
};

func(spec[nothing], int2, equal)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- int2(step(x, y) * step(y, x));
};

func(spec[nothing], int3, equal)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- int3(step(y, x) * step(y, x));
};

func(spec[nothing], int4, equal)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- int4(step(y, x) * step(y, x));
};

func(spec[nothing], int2, notEqual)params(var(spec[nothing], int2, x), var(spec[nothing], int2, y)) {
  return <- (int2(1, 1) - int2(step(x, y))) * (int2(1, 1) - int2(step(y, x)));
};

func(spec[nothing], int3, notEqual)params(var(spec[nothing], int3, x), var(spec[nothing], int3, y)) {
  return <- (int3(1, 1, 1) - int3(step(x, y))) * (int3(1, 1, 1) - int3(step(y, x)));
};

func(spec[nothing], int4, notEqual)params(var(spec[nothing], int4, x), var(spec[nothing], int4, y)) {
  return <- (int4(1, 1, 1, 1) - int4(step(x, y))) * (int4(1, 1, 1, 1) - int4(step(y, x)));
};

func(spec[nothing], int2, greaterThan)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- int2(1, 1) - int2(step(x, y));
};

func(spec[nothing], int3, greaterThan)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- int3(1, 1, 1) - int3(step(x, y));
};

func(spec[nothing], int4, greaterThan)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- int4(1, 1, 1, 1) - int4(step(x, y));
};

func(spec[nothing], int2, lessThan)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- int2(1.0, 1.0) - int2(step(y, x));
};

func(spec[nothing], int3, lessThan)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- int3(1.0, 1.0, 1.0) - int3(step(y, x));
};

func(spec[nothing], int4, lessThan)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- int4(1.0, 1.0, 1.0, 1.0) - int4(step(y, x));
};

func(spec[nothing], int2, greaterThanEqual)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- int2(step(y, x));
};

func(spec[nothing], int3, greaterThanEqual)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- int3(step(y, x));
};

func(spec[nothing], int4, greaterThanEqual)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- int4(step(y, x));
};

func(spec[nothing], int2, lessThanEqual)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- int2(step(x, y));
};

func(spec[nothing], int3, lessThanEqual)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- int3(step(x, y));
};

func(spec[nothing], int4, lessThanEqual)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- int4(step(x, y));
};

func(spec[nothing], int2, equal)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- int2(step(x, y) * step(y, x));
};

func(spec[nothing], int3, equal)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- int3(step(y, x) * step(y, x));
};

func(spec[nothing], int4, equal)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- int4(step(y, x) * step(y, x));
};

func(spec[nothing], int2, notEqual)params(var(spec[nothing], uint2, x), var(spec[nothing], uint2, y)) {
  return <- (int2(1, 1) - int2(step(x, y))) * (int2(1, 1) - int2(step(y, x)));
};

func(spec[nothing], int3, notEqual)params(var(spec[nothing], uint3, x), var(spec[nothing], uint3, y)) {
  return <- (int3(1, 1, 1) - int3(step(x, y))) * (int3(1, 1, 1) - int3(step(y, x)));
};

func(spec[nothing], int4, notEqual)params(var(spec[nothing], uint4, x), var(spec[nothing], uint4, y)) {
  return <- (int4(1, 1, 1, 1) - int4(step(x, y))) * (int4(1, 1, 1, 1) - int4(step(y, x)));
};

func(spec[nothing], int2, greaterThan)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- int2(1, 1) - int2(step(x, y));
};

func(spec[nothing], int3, greaterThan)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- int3(1, 1, 1) - int3(step(x, y));
};

func(spec[nothing], int4, greaterThan)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- int4(1, 1, 1, 1) - int4(step(x, y));
};

func(spec[nothing], int2, lessThan)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- int2(1.0, 1.0) - int2(step(y, x));
};

func(spec[nothing], int3, lessThan)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- int3(1.0, 1.0, 1.0) - int3(step(y, x));
};

func(spec[nothing], int4, lessThan)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- int4(1.0, 1.0, 1.0, 1.0) - int4(step(y, x));
};

func(spec[nothing], int2, greaterThanEqual)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- int2(step(y, x));
};

func(spec[nothing], int3, greaterThanEqual)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- int3(step(y, x));
};

func(spec[nothing], int4, greaterThanEqual)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- int4(step(y, x));
};

func(spec[nothing], int2, lessThanEqual)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- int2(step(x, y));
};

func(spec[nothing], int3, lessThanEqual)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- int3(step(x, y));
};

func(spec[nothing], int4, lessThanEqual)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- int4(step(x, y));
};

func(spec[nothing], int2, equal)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- int2(step(x, y) * step(y, x));
};

func(spec[nothing], int3, equal)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- int3(step(y, x) * step(y, x));
};

func(spec[nothing], int4, equal)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- int4(step(y, x) * step(y, x));
};

func(spec[nothing], int2, notEqual)params(var(spec[nothing], bool2, x), var(spec[nothing], bool2, y)) {
  return <- (int2(1, 1) - int2(step(x, y))) * (int2(1, 1) - int2(step(y, x)));
};

func(spec[nothing], int3, notEqual)params(var(spec[nothing], bool3, x), var(spec[nothing], bool3, y)) {
  return <- (int3(1, 1, 1) - int3(step(x, y))) * (int3(1, 1, 1) - int3(step(y, x)));
};

func(spec[nothing], int4, notEqual)params(var(spec[nothing], bool4, x), var(spec[nothing], bool4, y)) {
  return <- (int4(1, 1, 1, 1) - int4(step(x, y))) * (int4(1, 1, 1, 1) - int4(step(y, x)));
};

var(spec[static, nothing], SamplerState, s_linear_clamp_sampler);
typedef(float, td_float_x4[4]);
func(spec[nothing], td_float_x4, glsl_float_x4_ctor)params(var(spec[nothing], float), var(spec[nothing], float), var(spec[nothing], float), var(spec[nothing], float));
typedef(float3, td_vec3_x4[4]);
func(spec[nothing], td_vec3_x4, glsl_vec3_x4_ctor)params(var(spec[nothing], float3), var(spec[nothing], float3), var(spec[nothing], float3), var(spec[nothing], float3));
var(spec[static, nothing], float3, iResolution) = float3(1.0, 1.0, 1.0);
var(spec[static, nothing], float, iTime) = 0.0;
var(spec[static, nothing], float, iTimeDelta) = 0.0;
var(spec[static, nothing], float, iFrameRate) = 10.0;
var(spec[static, nothing], int, iFrame) = 0;
var(spec[static, nothing], float, iChannelTime[4]) = glsl_float_x4_ctor(0.0, 0.0, 0.0, 0.0);
var(spec[static, nothing], float3, iChannelResolution[4]) = glsl_vec3_x4_ctor(float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0));
var(spec[static, nothing], float4, iMouse) = float4(0.0, 0.0, 0.0, 0.0);
var(spec[static, nothing], float4, iDate) = float4(0.0, 0.0, 0.0, 0.0);
var(spec[static, nothing], float, iSampleRate) = 44100.0;
var(spec[static, nothing], Texture2D, iChannel0);
var(spec[static, nothing], Texture2D, iChannel1);
var(spec[static, nothing], Texture2D, iChannel2);
var(spec[static, nothing], Texture2D, iChannel3);
func(spec[nothing], float, noise)params(var(spec[in, nothing], const float2, x)) {
  var(spec[nothing], float2, p) = floor(x);
  var(spec[nothing], float2, f) = frac(x);
  f = f * f * (3.0 - 2.0 * f);
  var(spec[nothing], float2, uv) = (p.xy) + f.xy;
  return <- (iChannel0.SampleLevel(s_linear_clamp_sampler, (uv + 0.5) / 256.0, 0.0).x);
};

func(spec[nothing], float, noise)params(var(spec[in, nothing], const float3, x)) {
  var(spec[nothing], float3, p) = floor(x);
  var(spec[nothing], float3, f) = frac(x);
  f = f * f * (3.0 - 2.0 * f);
  var(spec[nothing], float2, uv) = (p.xy + float2(37.0, 17.0) * p.z) + f.xy;
  var(spec[nothing], float2, rg) = iChannel0.SampleLevel(s_linear_clamp_sampler, (uv + 0.5) / 256.0, 0.0).yx;
  return <- (lerp(rg.x, rg.y, f.z));
};

func(spec[nothing], float2x2, rot)params(var(spec[in, nothing], const float, a)) {
  return <- (float2x2(cos(a), sin(a), -sin(a), cos(a)));
};

var(spec[static, nothing], const float2x2, m2) = float2x2(0.59999999999999998, -0.80000000000000004, 0.80000000000000004, 0.59999999999999998);
var(spec[static, nothing], const float3x3, m3) = float3x3(0.0, 0.80000000000000004, 0.59999999999999998, -0.80000000000000004, 0.35999999999999999, -0.47999999999999998, -0.59999999999999998, -0.47999999999999998, 0.64000000000000001);
func(spec[nothing], float, fbm)params(var(spec[in, nothing], float3, p)) {
  var(spec[nothing], float, f) = 0.0;
  f += 0.5 * noise(p);
  p = mul(p, m3) * 2.02;
  f += 0.25 * noise(p);
  p = mul(p, m3) * 2.0299999999999998;
  f += 0.125 * noise(p);
  p = mul(p, m3) * 2.0099999999999998;
  f += 0.0625 * noise(p);
  return <- (f / 0.9375);
};

func(spec[nothing], float, hash)params(var(spec[in, nothing], float, n)) {
  return <- (frac(sin(n) * 43758.545299999998));
};

func(spec[nothing], bool, intersectPlane)params(var(spec[in, nothing], const float3, ro), var(spec[in, nothing], const float3, rd), var(spec[in, nothing], const float, height), var(spec[inout, nothing], float, dist)) {
  if (rd.y == 0.0) {
    return <- (false);
  };
  var(spec[nothing], float, d) = -(ro.y - height) / rd.y;
  d = min(1.0E+5, d);
  if (d > 0.0 && d < dist) {
    dist = d;
    return <- (true);
  } else {
    return <- (false);
  };
};

var(spec[static, nothing], float3, lig) = normalize(float3(0.29999999999999999, 0.5, 0.59999999999999998));
func(spec[nothing], float3, bgColor)params(var(spec[in, nothing], const float3, rd)) {
  var(spec[nothing], float, sun) = clamp(dot(lig, rd), 0.0, 1.0);
  var(spec[nothing], float3, col) = float3(0.5, 0.52000000000000002, 0.55000000000000004) - rd.y * 0.20000000000000001 * float3(1.0, 0.80000000000000004, 1.0) + 0.14999999999999999 * 0.75;
  col += float3(1.0, 0.59999999999999998, 0.10000000000000001) * pow(sun, 8.0);
  col *= 0.94999999999999996;
  return <- (col);
};

func(spec[nothing], float, cloudMap)params(var(spec[in, nothing], const float3, p), var(spec[in, nothing], const float, ani)) {
  var(spec[nothing], float3, r) = p / (500.0 / (64.0 * 0.029999999999999999));
  var(spec[nothing], float, den) = -1.8 + cos(r.y * 5.0 - 4.2999999999999998);
  var(spec[nothing], float, f);
  var(spec[nothing], float3, q) = 2.5 * r * float3(0.75, 1.0, 0.75) + float3(1.0, 2.0, 1.0) * ani * 0.14999999999999999;
  f = 0.5 * noise(q);
  q = q * 2.02 - float3(-1.0, 1.0, -1.0) * ani * 0.14999999999999999;
  f += 0.25 * noise(q);
  q = q * 2.0299999999999998 + float3(1.0, -1.0, 1.0) * ani * 0.14999999999999999;
  f += 0.125 * noise(q);
  q = q * 2.0099999999999998 - float3(1.0, 1.0, -1.0) * ani * 0.14999999999999999;
  f += 0.0625 * noise(q);
  q = q * 2.02 + float3(1.0, 1.0, 1.0) * ani * 0.14999999999999999;
  f += 0.03125 * noise(q);
  return <- (0.065000000000000002 * clamp(den + 4.4000000000000004 * f, 0.0, 1.0));
};

func(spec[nothing], float3, raymarchClouds)params(var(spec[in, nothing], const float3, ro), var(spec[in, nothing], const float3, rd), var(spec[in, nothing], const float3, bgc), var(spec[in, nothing], const float3, fgc), var(spec[in, nothing], const float, startdist), var(spec[in, nothing], const float, maxdist), var(spec[in, nothing], const float, ani)) {
  var(spec[nothing], float, t) = startdist + (500.0 / (64.0 * 0.029999999999999999)) * 0.02 * hash(rd.x + 35.698722099999998 * rd.y + (iTime + 285.0));
  var(spec[nothing], float4, sum) = glsl_vec4(0.0);
  for ([var(spec[nothing], int, i) = 0], [i < 32], [++i]) {
    if (sum.a > 0.98999999999999999 || t > maxdist) {
      continue;
    };
    var(spec[nothing], float3, pos) = ro + t * rd;
    var(spec[nothing], float, a) = cloudMap(pos, ani);
    var(spec[nothing], float, dif) = clamp(0.10000000000000001 + 0.80000000000000004 * (a - cloudMap(pos + lig * 0.14999999999999999 * (500.0 / (64.0 * 0.029999999999999999)), ani)), 0.0, 0.5);
    var(spec[nothing], float4, col) = float4((1.0 + dif) * fgc, a);
    col.rgb *= col.a;
    sum = sum + col * (1.0 - sum.a);
    t += (0.029999999999999999 * (500.0 / (64.0 * 0.029999999999999999))) + t * 0.012;
  };
  sum.xyz = lerp(bgc, sum.xyz / (sum.w + 1.0E-4), sum.w);
  return <- (clamp(sum.xyz, 0.0, 1.0));
};

func(spec[nothing], float, terrainMap)params(var(spec[in, nothing], const float3, p)) {
  return <- ((iChannel1.SampleLevel(s_linear_clamp_sampler, (mul(m2, -p.zx)) * 4.6E-5, 0.0).x * 600.0) * smoothstep(820.0, 1000.0, length(p.xz)) - 2.0 + noise(p.xz * 0.5) * 15.0);
};

func(spec[nothing], float3, raymarchTerrain)params(var(spec[in, nothing], const float3, ro), var(spec[in, nothing], const float3, rd), var(spec[in, nothing], const float3, bgc), var(spec[in, nothing], const float, startdist), var(spec[inout, nothing], float, dist)) {
  var(spec[nothing], float, t) = startdist;
  var(spec[nothing], float4, sum) = glsl_vec4(0.0);
  var(spec[nothing], bool, hit) = false;
  var(spec[nothing], float3, col) = bgc;
  for ([var(spec[nothing], int, i) = 0], [i < 20], [++i]) {
    if (hit) {
      break;
    };
    t += 8.0 + t / 300.0;
    var(spec[nothing], float3, pos) = ro + t * rd;
    if (pos.y < terrainMap(pos)) {
      hit = true;
    };
  };
  if (hit) {
    var(spec[nothing], float, dt) = 4.0 + t / 400.0;
    t -= dt;
    var(spec[nothing], float3, pos) = ro + t * rd;
    t += (0.5 - step(pos.y, terrainMap(pos))) * dt;
    for ([var(spec[nothing], int, j) = 0], [j < 2], [++j]) {
      pos = ro + t * rd;
      dt *= 0.5;
      t += (0.5 - step(pos.y, terrainMap(pos))) * dt;
    };
    pos = ro + t * rd;
    var(spec[nothing], float3, dx) = float3(100.0 * 0.10000000000000001, 0.0, 0.0);
    var(spec[nothing], float3, dz) = float3(0.0, 0.0, 100.0 * 0.10000000000000001);
    var(spec[nothing], float3, normal) = float3(0.0, 0.0, 0.0);
    normal.x = (terrainMap(pos + dx) - terrainMap(pos - dx)) / (200.0 * 0.10000000000000001);
    normal.z = (terrainMap(pos + dz) - terrainMap(pos - dz)) / (200.0 * 0.10000000000000001);
    normal.y = 1.0;
    normal = normalize(normal);
    col = glsl_vec3(0.20000000000000001) + 0.69999999999999996 * iChannel2.Sample(s_linear_clamp_sampler, pos.xz * 0.01).xyz * float3(1.0, 0.90000000000000002, 0.59999999999999998);
    var(spec[nothing], float, veg) = 0.29999999999999999 * fbm(pos * 0.20000000000000001) + normal.y;
    if (veg > 0.75) {
      col = float3(0.45000000000000001, 0.59999999999999998, 0.29999999999999999) * (0.5 + 0.5 * fbm(pos * 0.5)) * 0.59999999999999998;
    } elseif (veg > 0.66000000000000003) {
      col = col * 0.59999999999999998 + float3(0.40000000000000002, 0.5, 0.29999999999999999) * (0.5 + 0.5 * fbm(pos * 0.25)) * 0.29999999999999999;
    };
    col *= float3(0.5, 0.52000000000000002, 0.65000000000000002) * float3(1.0, 0.90000000000000002, 0.80000000000000004);
    var(spec[nothing], float3, brdf) = col;
    var(spec[nothing], float, diff) = clamp(dot(normal, -lig), 0.0, 1.0);
    col = brdf * diff * float3(1.0, 0.59999999999999998, 0.10000000000000001);
    col += brdf * clamp(dot(normal, lig), 0.0, 1.0) * float3(0.80000000000000004, 0.59999999999999998, 0.5) * 0.80000000000000004;
    col += brdf * clamp(dot(normal, float3(0.0, 1.0, 0.0)), 0.0, 1.0) * float3(0.80000000000000004, 0.80000000000000004, 1.0) * 0.20000000000000001;
    dist = t;
    t -= pos.y * 3.5;
    col = lerp(col, bgc, 1.0 - exp(-4.9999999999999998E-7 * t * t));
  };
  return <- (col);
};

func(spec[nothing], float, waterMap)params(var(spec[nothing], float2, pos)) {
  var(spec[nothing], float2, posm) = mul(m2, pos);
  return <- (abs(fbm(float3(8.0 * posm, (iTime + 285.0))) - 0.5) * 0.10000000000000001);
};

func(spec[nothing], void, mainImage)params(var(spec[out, nothing], float4, fragColor), var(spec[in, nothing], float2, fragCoord)) {
  var(spec[nothing], float2, q) = fragCoord.xy / iResolution.xy;
  var(spec[nothing], float2, p) = -1.0 + 2.0 * q;
  p.x *= iResolution.x / iResolution.y;
  var(spec[nothing], float3, ro) = float3(0.0, 0.5, 0.0);
  var(spec[nothing], float3, ta) = float3(0.0, 0.45000000000000001, 1.0);
  if (iMouse.z >= 1.0) {
    ta.xz = mul(rot((iMouse.x / iResolution.x - 0.5) * 7.0), ta.xz);
  };
  ta.xz = mul(rot((iTime * 0.050000000000000003 - floor(iTime * 0.050000000000000003 / 6.2831852000000001) * 6.2831852000000001)), ta.xz);
  var(spec[nothing], float3, ww) = normalize(ta - ro);
  var(spec[nothing], float3, uu) = normalize(cross(float3(0.0, 1.0, 0.0), ww));
  var(spec[nothing], float3, vv) = normalize(cross(ww, uu));
  var(spec[nothing], float3, rd) = normalize(p.x * uu + p.y * vv + 2.5 * ww);
  var(spec[nothing], float, fresnel), var(spec[nothing], float, refldist) = 5000.0, var(spec[nothing], float, maxdist) = 5000.0;
  var(spec[nothing], bool, reflected) = false;
  var(spec[nothing], float3, normal), var(spec[nothing], float3, col) = bgColor(rd);
  var(spec[nothing], float3, roo) = ro, var(spec[nothing], float3, rdo) = rd, var(spec[nothing], float3, bgc) = col;
  if (intersectPlane(ro, rd, 0.0, refldist) && refldist < 200.0) {
    ro += refldist * rd;
    var(spec[nothing], float2, coord) = ro.xz;
    var(spec[nothing], float, bumpfactor) = 0.10000000000000001 * (1.0 - smoothstep(0.0, 60.0, refldist));
    var(spec[nothing], float2, dx) = float2(0.10000000000000001, 0.0);
    var(spec[nothing], float2, dz) = float2(0.0, 0.10000000000000001);
    normal = float3(0.0, 1.0, 0.0);
    normal.x = -bumpfactor * (waterMap(coord + dx) - waterMap(coord - dx)) / (2.0 * 0.10000000000000001);
    normal.z = -bumpfactor * (waterMap(coord + dz) - waterMap(coord - dz)) / (2.0 * 0.10000000000000001);
    normal = normalize(normal);
    var(spec[nothing], float, ndotr) = dot(normal, rd);
    fresnel = pow(1.0 - abs(ndotr), 5.0);
    rd = reflect(rd, normal);
    reflected = true;
    bgc = col = bgColor(rd);
  };
  col = raymarchTerrain(ro, rd, col, reflected ? ((800.0 - refldist)) : ((800.0)), maxdist);
  col = raymarchClouds(ro, rd, col, bgc, reflected ? ((max(0.0, min(150.0, (150.0 - refldist))))) : ((150.0)), maxdist, (iTime + 285.0) * 0.050000000000000003);
  if (reflected) {
    col = lerp(col.xyz, bgc, 1.0 - exp(-4.9999999999999998E-7 * refldist * refldist));
    col *= fresnel * 0.90000000000000002;
    var(spec[nothing], float3, refr) = refract(rdo, normal, 1.0 / 1.333);
    intersectPlane(ro, refr, -2.0, refldist);
    col += lerp(iChannel2.Sample(s_linear_clamp_sampler, (roo + refldist * refr).xz * 1.3).xyz * float3(1.0, 0.90000000000000002, 0.59999999999999998), float3(1.0, 0.90000000000000002, 0.80000000000000004) * 0.5, clamp(refldist / 3.0, 0.0, 1.0)) * (1.0 - fresnel) * 0.125;
  };
  col = pow(col, glsl_vec3(0.69999999999999996));
  col = col * col * (3.0 - 2.0 * col);
  col = lerp(col, glsl_vec3(dot(col, glsl_vec3(0.33000000000000002))), -0.5);
  col *= 0.25 + 0.75 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.10000000000000001);
  fragColor = float4(col, 1.0);
};

func(spec[nothing], td_float_x4, glsl_float_x4_ctor)params(var(spec[nothing], float, v0), var(spec[nothing], float, v1), var(spec[nothing], float, v2), var(spec[nothing], float, v3)) {
  var(spec[nothing], float, __arr_tmp[4]) = { v0, v1, v2, v3 };
  return <- __arr_tmp;
};

func(spec[nothing], td_vec3_x4, glsl_vec3_x4_ctor)params(var(spec[nothing], float3, v0), var(spec[nothing], float3, v1), var(spec[nothing], float3, v2), var(spec[nothing], float3, v3)) {
  var(spec[nothing], float3, __arr_tmp[4]) = { v0, v1, v2, v3 };
  return <- __arr_tmp;
};



