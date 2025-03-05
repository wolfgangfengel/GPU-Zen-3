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
struct(rmRes);
func(spec[nothing], rmRes, glsl_rmRes_ctor)params(var(spec[nothing], float3), var(spec[nothing], int), var(spec[nothing], bool));
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
var(spec[static, nothing], float, gameTime);
func(spec[nothing], float2x2, rot)params(var(spec[in, nothing], float, a)) {
  return <- (float2x2(cos(a), sin(a), -sin(a), cos(a)));
};

func(spec[nothing], float, opSmoothUnion)params(var(spec[nothing], float, d1), var(spec[nothing], float, d2), var(spec[nothing], float, k)) {
  var(spec[nothing], float, h) = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return <- (lerp(d2, d1, h) - k * h * (1.0 - h));
};

func(spec[nothing], float, petalDcp)params(var(spec[in, nothing], float2, uv), var(spec[in, nothing], float, w)) {
  uv.x = abs(uv.x) + 0.25 + 0.25 * w;
  return <- (length(uv) - 0.5);
};

func(spec[nothing], float, petal)params(var(spec[in, nothing], float3, p), var(spec[in, nothing], float, m)) {
  var(spec[nothing], float, tt) = (gameTime - floor(gameTime / 6.2831853070000001 * 0.5) * 6.2831853070000001 * 0.5);
  var(spec[nothing], float, ouv) = m - 0.014999999999999999;
  var(spec[nothing], float, w) = m;
  var(spec[nothing], float, a) = m;
  var(spec[nothing], const float, b) = 0.5;
  p.y -= 0.45000000000000001;
  p.z -= b * 1.0;
  p.zy = mul(rot(ouv * 2.0), p.zy);
  var(spec[nothing], float, pDcp) = petalDcp(p.xy, w);
  p.x = abs(p.x);
  p.xz = mul(rot(-0.25), p.xz);
  var(spec[nothing], float, c1) = length(p.yz) - b;
  return <- (max(max(pDcp, abs(c1) - 0.01), p.z));
};

func(spec[nothing], float2, repRot)params(var(spec[in, nothing], float2, p), var(spec[in, nothing], float, aIt)) {
  return <- (mul(rot(-(6.2831853070000001 / aIt) * floor((atan2(p.x, p.y) / 6.2831853070000001 + 0.5) * aIt) - 6.2831853070000001 * 0.5 - 6.2831853070000001 / (aIt * 2.0)), p));
};

func(spec[nothing], float, flower)params(var(spec[in, nothing], float3, p), var(spec[in, nothing], float, aIt), var(spec[in, nothing], float, m)) {
  p.xy = repRot(p.xy, aIt);
  return <- (petal(p, m));
};

func(spec[nothing], float, df)params(var(spec[in, nothing], float3, _pp), var(spec[inout, nothing], int, m)) {
  _pp.y = -_pp.y;
  _pp.xz = mul(rot(1.016), _pp.xz) , _pp.xy = mul(rot(-0.64000000000000001), _pp.xy);
  var(spec[nothing], float, dd) = 1.0E+10, var(spec[nothing], float, ee) = 1.0E+10;
  var(spec[nothing], float3, p) = _pp;
  var(spec[nothing], const float, fsz) = 0.25;
  var(spec[nothing], const float2, n) = float2(cos((6.2831853070000001 * 0.5 * 0.125)), sin((6.2831853070000001 * 0.5 * 0.125)));
  var(spec[nothing], bool, b) = false;
  for ([var(spec[nothing], float, g) = 0.0], [g < 3.0], [g++]) {
    p = (b = !b) ? ((p.xzy)) : ((p.zxy));
    var(spec[nothing], float, r) = length(p.xy);
    var(spec[nothing], float3, pp) = float3(log(r) - gameTime * (0.10000000000000001 + ((g + 1.0) * 0.050999999999999997)), atan2(p.x, p.y), p.z / r);
    var(spec[nothing], float, e) = dot(pp.xy, n), var(spec[nothing], float, f) = dot(pp.xy, float2(n.y, -n.x));
    block {
      var(spec[nothing], float, k) = 1.2020999999999999;
      e = (e - floor(e / k) * k) - k * 0.5;
    };
    var(spec[nothing], float, l) = 0.65000000000000002;
    f += 1.3;
    var(spec[nothing], float, i) = (floor(f / l) + g - floor(floor(f / l) + g / 3.0) * 3.0);
    f = (f - floor(f / l) * l) - l * 0.5;
    var(spec[nothing], float, d) = (length(float2(e, pp.z)) - 0.014999999999999999 / r) * r;
    var(spec[nothing], bool, j) = i == 0.0;
    dd = opSmoothUnion(dd, d, 0.10000000000000001);
    var(spec[nothing], float, ff) = flower(float3(e, f, pp.z + 0.059999999999999998) / fsz, smoothstep(-1.0, 1.0, r * r) * (j ? ((5.0)) : ((2.0))), smoothstep(1.0, -0.0, r * r)) * fsz * r;
    ee = min(ee, ff);
    if (ee == ff) {
      m = j ? ((1)) : ((0));
    };
  };
  var(spec[nothing], float, ff) = min(dd, ee);
  if (ff == dd) {
    m = 0;
  };
  return <- (ff * 0.80000000000000004);
};

func(spec[nothing], float3, normal)params(var(spec[in, nothing], float3, p), var(spec[inout, nothing], int, m)) {
  var(spec[nothing], float, d) = df(p, m);
  var(spec[nothing], float2, u) = float2(0.0, 2.0000000000000001E-4);
  return <- (normalize(float3(df(p + u.yxx, m), df(p + u.xyx, m), df(p + u.xxy, m)) - d));
};

struct(rmRes) {
  field(spec[nothing], float3, p);
  field(spec[nothing], int, i);
  field(spec[nothing], bool, h);
};
func(spec[nothing], rmRes, rm)params(var(spec[in, nothing], float3, c), var(spec[in, nothing], float3, r), var(spec[inout, nothing], int, m)) {
  var(spec[nothing], rmRes, s) = glsl_rmRes_ctor(c + r * 0.0, 0, false);
  var(spec[nothing], float, d);
  var(spec[nothing], int, i) = 0;
  for ([], [i < 16], [i++]) {
    d = df(s.p, m);
    if (d < 2.0000000000000001E-4) {
      s.h = true;
      break;
    };
    if (distance(c, s.p) > 30.0) {
      break;
    };
    s.p += d * r;
  };
  s.i = i;
  return <- (s);
};

func(spec[nothing], void, mainImage)params(var(spec[out, nothing], float4, fragColor), var(spec[in, nothing], float2, fragCoord)) {
  var(spec[nothing], float2, uv) = float2(1071.0, 503.0) / iResolution.xy;
  uv = float2(0.59499999999999997, 0.4965);
  var(spec[nothing], float2, coord) = uv * iResolution.xy;
  coord = fragCoord;
  var(spec[nothing], int, m) = 0;
  var(spec[nothing], float2, st) = (coord - iResolution.xy * 0.5) / iResolution.x;
  gameTime = iTime;
  var(spec[nothing], float3, c) = float3(0.0, 0.0, -10.0), var(spec[nothing], float3, r) = normalize(float3(st, 1.0));
  var(spec[nothing], rmRes, res) = rm(c, r, m);
  var(spec[nothing], float3, sky) = (float3(0.95499999999999996, 0.91200000000000003, 0.93100000000000005) - dot(st, st) * 0.20000000000000001);
  var(spec[nothing], float3, color) = sky;
  if (res.h) {
    var(spec[nothing], float3, n) = normal(res.p, m);
    var(spec[nothing], const float3, ld) = normalize(float3(0.0, 1.0, -0.10000000000000001));
    var(spec[nothing], float, d) = max(0.0, dot(n, ld));
    var(spec[nothing], float, s) = pow(max(0.0, dot(r, reflect(ld, n))), 1.0);
    color = lerp(float3(0.5, 0.76300000000000001, 0.91500000000000004), glsl_vec3(1.0), d);
    color *= m == 1 ? ((float3(0.90500000000000003, 0.17000000000000001, 0.29199999999999998))) : ((float3(0.88500000000000001, 0.88200000000000001, 0.94499999999999995)));
    color = lerp(color, sky, smoothstep(20.0, 25.0, distance(res.p, c)));
    color = lerp(color, sky, smoothstep(0.5, 3.0, dot(st, st) * 10.0));
  };
  fragColor = float4(color, 1.0);
};

func(spec[nothing], rmRes, glsl_rmRes_ctor)params(var(spec[nothing], float3, _p), var(spec[nothing], int, _i), var(spec[nothing], bool, _h)) {
  var(spec[nothing], rmRes, __stru_tmp) = { _p, _i, _h };
  return <- __stru_tmp;
};

func(spec[nothing], td_float_x4, glsl_float_x4_ctor)params(var(spec[nothing], float, v0), var(spec[nothing], float, v1), var(spec[nothing], float, v2), var(spec[nothing], float, v3)) {
  var(spec[nothing], float, __arr_tmp[4]) = { v0, v1, v2, v3 };
  return <- __arr_tmp;
};

func(spec[nothing], td_vec3_x4, glsl_vec3_x4_ctor)params(var(spec[nothing], float3, v0), var(spec[nothing], float3, v1), var(spec[nothing], float3, v2), var(spec[nothing], float3, v3)) {
  var(spec[nothing], float3, __arr_tmp[4]) = { v0, v1, v2, v3 };
  return <- __arr_tmp;
};



