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
func(spec[nothing], float2x2, mm2)params(var(spec[in, nothing], float, a)) {
  var(spec[nothing], float, c) = cos(a), var(spec[nothing], float, s) = sin(a);
  return <- (float2x2(c, s, -s, c));
};

func(spec[nothing], float3, objmov)params(var(spec[nothing], float3, p)) {
  p.xz = mul(mm2(-iTime * 3.3999999999999999 + sin(iTime * 1.1100000000000001)), p.xz);
  p.yz = mul(mm2(iTime * 2.7000000000000002 + cos(iTime * 2.5)), p.yz);
  return <- (p);
};

func(spec[nothing], float, tri)params(var(spec[in, nothing], float, x)) {
  return <- (abs(frac(x) - 0.5) - 0.25);
};

func(spec[nothing], float, trids)params(var(spec[in, nothing], float3, p)) {
  return <- (max(tri(p.z), min(tri(p.x), tri(p.y))));
};

func(spec[nothing], float, tri2)params(var(spec[in, nothing], float, x)) {
  return <- (abs(frac(x) - 0.5));
};

func(spec[nothing], float3, tri3)params(var(spec[in, nothing], float3, p)) {
  return <- (float3(tri(p.z + tri(p.y * 1.0)), tri(p.z + tri(p.x * 1.05)), tri(p.y + tri(p.x * 1.1000000000000001))));
};

var(spec[static, nothing], float2x2, m2) = float2x2(0.96999999999999997, 0.24199999999999999, -0.24199999999999999, 0.96999999999999997);
func(spec[nothing], float, triNoise3d)params(var(spec[in, nothing], float3, p), var(spec[in, nothing], float, spd)) {
  var(spec[nothing], float, z) = 1.45;
  var(spec[nothing], float, rz) = 0.0;
  var(spec[nothing], float3, bp) = p;
  for ([var(spec[nothing], float, i) = 0.0], [i < 4.0], [++i]) {
    var(spec[nothing], float3, dg) = tri3(bp);
    p += (dg + iTime * spd + 10.1);
    bp *= 1.6499999999999999;
    z *= 1.5;
    p *= 0.90000000000000002;
    p.xz = mul(m2, p.xz);
    rz += (tri2(p.z + tri2(p.x + tri2(p.y)))) / z;
    bp += 0.90000000000000002;
  };
  return <- (rz);
};

func(spec[nothing], float, map)params(var(spec[nothing], float3, p)) {
  p *= 1.5;
  p = objmov(p);
  var(spec[nothing], float, d) = length(p) - 1.0;
  d -= trids(p * 1.2) * 0.69999999999999996;
  return <- (d / 1.5);
};

func(spec[nothing], float, map2)params(var(spec[nothing], float3, p)) {
  p = objmov(p);
  return <- (length(p) - 1.3);
};

func(spec[nothing], float, march)params(var(spec[in, nothing], float3, ro), var(spec[in, nothing], float3, rd)) {
  var(spec[nothing], float, precis) = 0.001;
  var(spec[nothing], float, h) = precis * 2.0;
  var(spec[nothing], float, d) = 0.0;
  for ([var(spec[nothing], int, i) = 0], [i < 35], [++i]) {
    if (abs(h) < precis || d > 15.0) {
      break;
    };
    d += h;
    var(spec[nothing], float, res) = map(ro + rd * d);
    h = res;
  };
  return <- (d);
};

func(spec[nothing], float3, normal)params(var(spec[in, nothing], const float3, p)) {
  var(spec[nothing], float2, e) = float2(-1.0, 1.0) * 0.040000000000000001;
  return <- (normalize(e.yxx * map(p + e.yxx) + e.xxy * map(p + e.xxy) + e.xyx * map(p + e.xyx) + e.yyy * map(p + e.yyy)));
};

func(spec[nothing], float, gradm)params(var(spec[in, nothing], float3, p)) {
  var(spec[nothing], float, e) = 0.059999999999999998;
  var(spec[nothing], float, d) = map2(float3(p.x, p.y - e, p.z)) - map2(float3(p.x, p.y + e, p.z));
  d += map2(float3(p.x - e, p.y, p.z)) - map2(float3(p.x + e, p.y, p.z));
  d += map2(float3(p.x, p.y, p.z - e)) - map2(float3(p.x, p.y, p.z + e));
  return <- (d);
};

func(spec[nothing], float, mapVol)params(var(spec[nothing], float3, p), var(spec[in, nothing], float, spd)) {
  var(spec[nothing], float, f) = smoothstep(0.0, 1.25, 1.7 - (p.y + dot(p.xz, p.xz) * 0.62));
  var(spec[nothing], float, g) = p.y;
  p.y *= 0.27000000000000002;
  p.z += gradm(p * 0.72999999999999998) * 3.5;
  p.y += iTime * 6.0;
  var(spec[nothing], float, d) = triNoise3d(p * float3(0.29999999999999999, 0.27000000000000002, 0.29999999999999999) - float3(0, iTime * 0.0, 0), spd * 0.69999999999999996) * 1.3999999999999999 + 0.01;
  d += max((g - 0.0) * 0.29999999999999999, 0.0);
  d *= f;
  return <- (clamp(d, 0.0, 1.0));
};

func(spec[nothing], float3, marchVol)params(var(spec[in, nothing], float3, ro), var(spec[in, nothing], float3, rd), var(spec[in, nothing], float, t), var(spec[in, nothing], float, mt)) {
  var(spec[nothing], float4, rz) = glsl_vec4(0.0);
  t -= (dot(rd, float3(0, 1, 0)) + 1.0);
  var(spec[nothing], float, tmt) = t + 15.0;
  for ([var(spec[nothing], int, i) = 0], [i < 25], [++i]) {
    if (rz.a > 0.98999999999999999) {
      break;
    };
    var(spec[nothing], float3, pos) = ro + t * rd;
    var(spec[nothing], float, r) = mapVol(pos, 0.10000000000000001);
    var(spec[nothing], float, gr) = clamp((r - mapVol(pos + float3(0.0, 0.69999999999999996, 0.0), 0.10000000000000001)) / 0.29999999999999999, 0.0, 1.0);
    var(spec[nothing], float3, lg) = float3(0.71999999999999997, 0.28000000000000003, 0.0) * 1.2 + 1.3 * float3(0.55000000000000004, 0.77000000000000001, 0.90000000000000002) * gr;
    var(spec[nothing], float4, col) = float4(lg, r * r * r * 2.5);
    col *= smoothstep(t - 0.0, t + 0.20000000000000001, mt);
    pos.y *= 0.69999999999999996;
    pos.zx *= ((pos.y - 5.0) * 0.14999999999999999 - 0.40000000000000002);
    var(spec[nothing], float, z2) = length(float3(pos.x, pos.y * 0.75 - 0.5, pos.z)) - 0.75;
    col.a *= smoothstep(0.40000000000000002, 1.2, 0.69999999999999996 - map2(float3(pos.x, pos.y * 0.17000000000000001, pos.z)));
    col.rgb *= col.a;
    rz = rz + col * (1.0 - rz.a);
    t += abs(z2) * 0.10000000000000001 + 0.12;
    if (t > mt || t > tmt) {
      break;
    };
  };
  rz.g *= rz.w * 0.90000000000000002 + 0.12;
  rz.r *= rz.w * 0.5 + 0.47999999999999998;
  return <- (clamp(rz.rgb, 0.0, 1.0));
};

func(spec[nothing], float, mapVol2)params(var(spec[nothing], float3, p), var(spec[in, nothing], float, spd)) {
  p *= 1.3;
  var(spec[nothing], float, f) = smoothstep(0.20000000000000001, 1.0, 1.3 - (p.y + length(p.xz) * 0.40000000000000002));
  p.y *= 0.050000000000000003;
  p.y += iTime * 1.7;
  var(spec[nothing], float, d) = triNoise3d(p * 1.1000000000000001, spd);
  d = clamp(d - 0.14999999999999999, 0.0, 0.75);
  d *= d * d * d * d * 47.0;
  d *= f;
  return <- (d);
};

func(spec[nothing], float3, marchVol2)params(var(spec[in, nothing], float3, ro), var(spec[in, nothing], float3, rd), var(spec[in, nothing], float, t), var(spec[in, nothing], float, mt)) {
  var(spec[nothing], float3, bpos) = ro + rd * t;
  t += length(float3(bpos.x, bpos.y, bpos.z)) - 1.0;
  t -= dot(rd, float3(0, 1, 0));
  var(spec[nothing], float4, rz) = glsl_vec4(0.0);
  var(spec[nothing], float, tmt) = t + 1.5;
  for ([var(spec[nothing], int, i) = 0], [i < 25], [++i]) {
    if (rz.a > 0.98999999999999999) {
      break;
    };
    var(spec[nothing], float3, pos) = ro + t * rd;
    var(spec[nothing], float, r) = mapVol2(pos, 0.01);
    var(spec[nothing], float3, lg) = float3(0.69999999999999996, 0.29999999999999999, 0.20000000000000001) * 1.5 + 2.0 * float3(1, 1, 1) * 0.75;
    var(spec[nothing], float4, col) = float4(lg, r * r * r * 3.0);
    col *= smoothstep(t - 0.25, t + 0.20000000000000001, mt);
    var(spec[nothing], float, z2) = length(float3(pos.x, pos.y * 0.90000000000000002, pos.z)) - 0.90000000000000002;
    col.a *= smoothstep(0.69999999999999996, 1.7, 1.0 - map2(float3(pos.x * 1.1000000000000001, pos.y * 0.40000000000000002, pos.z * 1.1000000000000001)));
    col.rgb *= col.a;
    rz = rz + col * (1.0 - rz.a);
    t += z2 * 0.014999999999999999 + abs(0.34999999999999998 - r) * 0.089999999999999996;
    if (t > mt || t > tmt) {
      break;
    };
  };
  return <- (clamp(rz.rgb, 0.0, 1.0));
};

func(spec[nothing], float3, hash33)params(var(spec[nothing], float3, p)) {
  p = frac(p * float3(443.89749999999998, 397.29730000000001, 491.18709999999999));
  p += dot(p.zxy, p.yxz + 19.27);
  return <- (frac(float3(p.x * p.y, p.z * p.x, p.y * p.z)));
};

func(spec[nothing], float3, stars)params(var(spec[in, nothing], float3, p)) {
  var(spec[nothing], float3, c) = glsl_vec3(0.0);
  var(spec[nothing], float, res) = iResolution.x * 0.80000000000000004;
  for ([var(spec[nothing], float, i) = 0.0], [i < 4.0], [++i]) {
    var(spec[nothing], float3, q) = frac(p * (0.14999999999999999 * res)) - 0.5;
    var(spec[nothing], float3, id) = floor(p * (0.14999999999999999 * res));
    var(spec[nothing], float2, rn) = hash33(id).xy;
    var(spec[nothing], float, c2) = 1.0 - smoothstep(0.0, 0.59999999999999998, length(q));
    c2 *= step(rn.x, 5.0000000000000001E-4 + i * i * 0.001);
    c += c2 * (lerp(float3(1.0, 0.48999999999999999, 0.10000000000000001), float3(0.75, 0.90000000000000002, 1.0), rn.y) * 0.25 + 0.75);
    p *= 1.3999999999999999;
  };
  return <- (c * c * 0.65000000000000002);
};

func(spec[nothing], float, curv)params(var(spec[in, nothing], float3, p), var(spec[in, nothing], float, w)) {
  var(spec[nothing], float2, e) = float2(-1.0, 1.0) * w;
  var(spec[nothing], float, t1) = map(p + e.yxx), var(spec[nothing], float, t2) = map(p + e.xxy);
  var(spec[nothing], float, t3) = map(p + e.xyx), var(spec[nothing], float, t4) = map(p + e.yyy);
  return <- (1.0 / e.y * (t1 + t2 + t3 + t4 - 4.0 * map(p)));
};

func(spec[nothing], void, mainImage)params(var(spec[out, nothing], float4, fragColor), var(spec[in, nothing], float2, fragCoord)) {
  var(spec[nothing], float2, p) = fragCoord.xy / iResolution.xy - 0.5;
  p.x *= iResolution.x / iResolution.y;
  var(spec[nothing], float2, mo) = iMouse.xy / iResolution.xy - 0.5;
  mo = float2(-0.27000000000000002, 0.31);
  mo.x *= iResolution.x / iResolution.y;
  var(spec[nothing], const float, roz) = 7.2999999999999998;
  var(spec[nothing], float3, ro) = float3(-1.5, 0.5, roz);
  var(spec[nothing], float3, rd) = normalize(float3(p, -1.5));
  mo.x += sin(iTime * 0.29999999999999999 + sin(iTime * 0.050000000000000003)) * 0.029999999999999999 + 0.029999999999999999;
  mo.y += sin(iTime * 0.40000000000000002 + sin(iTime * 0.059999999999999998)) * 0.029999999999999999;
  var(spec[nothing], float2x2, mx) = mm2(mo.x * 6.0);
  var(spec[nothing], float2x2, my) = mm2(mo.y * 6.0);
  ro.xz = mul(mx, ro.xz);
  rd.xz = mul(mx, rd.xz);
  ro.xy = mul(my, ro.xy);
  rd.xy = mul(my, rd.xy);
  var(spec[nothing], float, rz) = march(ro, rd);
  var(spec[nothing], float3, col) = stars(rd);
  var(spec[nothing], float, maxT) = rz;
  if (rz > 15.0) {
    maxT = 25.0;
  };
  var(spec[nothing], float3, mv) = marchVol(ro, rd, roz - 1.5, maxT);
  if (rz < 15.0) {
    var(spec[nothing], float3, pos) = ro + rz * rd;
    var(spec[nothing], float3, nor) = normal(pos);
    var(spec[nothing], float, crv) = clamp(curv(pos, 0.29999999999999999) * 0.34999999999999998, 0.0, 1.3);
    var(spec[nothing], float3, col2) = float3(1, 0.10000000000000001, 0.02) * (crv * 0.80000000000000004 + 0.20000000000000001) * 0.5;
    var(spec[nothing], float, frict) = dot(pos, normalize(float3(0.0, 1.0, 0.0)));
    col = col2 * (frict * 0.29999999999999999 + 0.69999999999999996);
    col += float3(1, 0.29999999999999999, 0.10000000000000001) * (crv * 0.69999999999999996 + 0.29999999999999999) * max((frict * 0.5 + 0.5), 0.0) * 1.3;
    col += float3(0.80000000000000004, 0.80000000000000004, 0.5) * (crv * 0.90000000000000002 + 0.10000000000000001) * pow(max(frict, 0.0), 1.5) * 1.8999999999999999;
    pos = objmov(pos);
    col *= 1.2 - mv;
    col *= triNoise3d(pos * 2.7999999999999998, 0.0) * 0.25 + 0.45000000000000001;
    col = pow(col, float3(1.5, 1.2, 1.2)) * 0.90000000000000002;
  };
  col += mv;
  col += marchVol2(ro, rd, roz - 5.5, rz);
  col = pow(col, glsl_vec3(1.3999999999999999)) * 1.1000000000000001;
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



