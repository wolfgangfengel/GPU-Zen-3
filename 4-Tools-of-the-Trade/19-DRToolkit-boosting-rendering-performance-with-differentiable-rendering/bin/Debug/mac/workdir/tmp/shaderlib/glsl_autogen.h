//----------------------------------------
// these code generated from gen_glsl_h.dsl
//---begin---
vec2 glsl_vec2(dvec2 arg)
{
    return vec2(arg);
}            
vec2 glsl_vec2(ivec2 arg)
{
    return vec2(arg);
}            
vec2 glsl_vec2(uvec2 arg)
{
    return vec2(arg);
}            
vec2 glsl_vec2(bvec2 arg)
{
    return vec2(arg);
}            
dvec2 glsl_dvec2(vec2 arg)
{
    return dvec2(arg);
}            
dvec2 glsl_dvec2(ivec2 arg)
{
    return dvec2(arg);
}            
dvec2 glsl_dvec2(uvec2 arg)
{
    return dvec2(arg);
}            
dvec2 glsl_dvec2(bvec2 arg)
{
    return dvec2(arg);
}            
ivec2 glsl_ivec2(vec2 arg)
{
    return ivec2(arg);
}            
ivec2 glsl_ivec2(dvec2 arg)
{
    return ivec2(arg);
}            
ivec2 glsl_ivec2(uvec2 arg)
{
    return ivec2(arg);
}            
ivec2 glsl_ivec2(bvec2 arg)
{
    return ivec2(arg);
}            
uvec2 glsl_uvec2(vec2 arg)
{
    return uvec2(arg);
}            
uvec2 glsl_uvec2(dvec2 arg)
{
    return uvec2(arg);
}            
uvec2 glsl_uvec2(ivec2 arg)
{
    return uvec2(arg);
}            
uvec2 glsl_uvec2(bvec2 arg)
{
    return uvec2(arg);
}            
bvec2 glsl_bvec2(vec2 arg)
{
    return bvec2(arg);
}            
bvec2 glsl_bvec2(dvec2 arg)
{
    return bvec2(arg);
}            
bvec2 glsl_bvec2(ivec2 arg)
{
    return bvec2(arg);
}            
bvec2 glsl_bvec2(uvec2 arg)
{
    return bvec2(arg);
}            
vec3 glsl_vec3(dvec3 arg)
{
    return vec3(arg);
}            
vec3 glsl_vec3(ivec3 arg)
{
    return vec3(arg);
}            
vec3 glsl_vec3(uvec3 arg)
{
    return vec3(arg);
}            
vec3 glsl_vec3(bvec3 arg)
{
    return vec3(arg);
}            
dvec3 glsl_dvec3(vec3 arg)
{
    return dvec3(arg);
}            
dvec3 glsl_dvec3(ivec3 arg)
{
    return dvec3(arg);
}            
dvec3 glsl_dvec3(uvec3 arg)
{
    return dvec3(arg);
}            
dvec3 glsl_dvec3(bvec3 arg)
{
    return dvec3(arg);
}            
ivec3 glsl_ivec3(vec3 arg)
{
    return ivec3(arg);
}            
ivec3 glsl_ivec3(dvec3 arg)
{
    return ivec3(arg);
}            
ivec3 glsl_ivec3(uvec3 arg)
{
    return ivec3(arg);
}            
ivec3 glsl_ivec3(bvec3 arg)
{
    return ivec3(arg);
}            
uvec3 glsl_uvec3(vec3 arg)
{
    return uvec3(arg);
}            
uvec3 glsl_uvec3(dvec3 arg)
{
    return uvec3(arg);
}            
uvec3 glsl_uvec3(ivec3 arg)
{
    return uvec3(arg);
}            
uvec3 glsl_uvec3(bvec3 arg)
{
    return uvec3(arg);
}            
bvec3 glsl_bvec3(vec3 arg)
{
    return bvec3(arg);
}            
bvec3 glsl_bvec3(dvec3 arg)
{
    return bvec3(arg);
}            
bvec3 glsl_bvec3(ivec3 arg)
{
    return bvec3(arg);
}            
bvec3 glsl_bvec3(uvec3 arg)
{
    return bvec3(arg);
}            
vec4 glsl_vec4(dvec4 arg)
{
    return vec4(arg);
}            
vec4 glsl_vec4(ivec4 arg)
{
    return vec4(arg);
}            
vec4 glsl_vec4(uvec4 arg)
{
    return vec4(arg);
}            
vec4 glsl_vec4(bvec4 arg)
{
    return vec4(arg);
}            
dvec4 glsl_dvec4(vec4 arg)
{
    return dvec4(arg);
}            
dvec4 glsl_dvec4(ivec4 arg)
{
    return dvec4(arg);
}            
dvec4 glsl_dvec4(uvec4 arg)
{
    return dvec4(arg);
}            
dvec4 glsl_dvec4(bvec4 arg)
{
    return dvec4(arg);
}            
ivec4 glsl_ivec4(vec4 arg)
{
    return ivec4(arg);
}            
ivec4 glsl_ivec4(dvec4 arg)
{
    return ivec4(arg);
}            
ivec4 glsl_ivec4(uvec4 arg)
{
    return ivec4(arg);
}            
ivec4 glsl_ivec4(bvec4 arg)
{
    return ivec4(arg);
}            
uvec4 glsl_uvec4(vec4 arg)
{
    return uvec4(arg);
}            
uvec4 glsl_uvec4(dvec4 arg)
{
    return uvec4(arg);
}            
uvec4 glsl_uvec4(ivec4 arg)
{
    return uvec4(arg);
}            
uvec4 glsl_uvec4(bvec4 arg)
{
    return uvec4(arg);
}            
bvec4 glsl_bvec4(vec4 arg)
{
    return bvec4(arg);
}            
bvec4 glsl_bvec4(dvec4 arg)
{
    return bvec4(arg);
}            
bvec4 glsl_bvec4(ivec4 arg)
{
    return bvec4(arg);
}            
bvec4 glsl_bvec4(uvec4 arg)
{
    return bvec4(arg);
}            

ivec2 greaterThan(vec2 x, vec2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan(vec3 x, vec3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan(vec4 x, vec4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan(vec2 x, vec2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan(vec3 x, vec3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan(vec4 x, vec4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual(vec2 x, vec2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual(vec3 x, vec3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual(vec4 x, vec4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual(vec2 x, vec2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual(vec3 x, vec3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual(vec4 x, vec4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal(vec2 x, vec2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal(vec3 x, vec3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal(vec4 x, vec4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual(vec2 x, vec2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual(vec3 x, vec3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual(vec4 x, vec4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
ivec2 greaterThan(dvec2 x, dvec2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan(dvec3 x, dvec3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan(dvec4 x, dvec4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan(dvec2 x, dvec2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan(dvec3 x, dvec3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan(dvec4 x, dvec4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual(dvec2 x, dvec2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual(dvec3 x, dvec3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual(dvec4 x, dvec4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual(dvec2 x, dvec2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual(dvec3 x, dvec3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual(dvec4 x, dvec4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal(dvec2 x, dvec2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal(dvec3 x, dvec3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal(dvec4 x, dvec4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual(dvec2 x, dvec2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual(dvec3 x, dvec3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual(dvec4 x, dvec4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
ivec2 greaterThan(ivec2 x, ivec2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan(ivec3 x, ivec3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan(ivec4 x, ivec4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan(ivec2 x, ivec2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan(ivec3 x, ivec3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan(ivec4 x, ivec4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual(ivec2 x, ivec2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual(ivec3 x, ivec3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual(ivec4 x, ivec4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual(ivec2 x, ivec2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual(ivec3 x, ivec3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual(ivec4 x, ivec4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal(ivec2 x, ivec2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal(ivec3 x, ivec3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal(ivec4 x, ivec4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual(ivec2 x, ivec2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual(ivec3 x, ivec3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual(ivec4 x, ivec4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
ivec2 greaterThan(uvec2 x, uvec2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan(uvec3 x, uvec3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan(uvec4 x, uvec4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan(uvec2 x, uvec2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan(uvec3 x, uvec3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan(uvec4 x, uvec4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual(uvec2 x, uvec2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual(uvec3 x, uvec3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual(uvec4 x, uvec4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual(uvec2 x, uvec2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual(uvec3 x, uvec3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual(uvec4 x, uvec4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal(uvec2 x, uvec2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal(uvec3 x, uvec3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal(uvec4 x, uvec4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual(uvec2 x, uvec2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual(uvec3 x, uvec3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual(uvec4 x, uvec4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
ivec2 greaterThan(bvec2 x, bvec2 y)
{
    return ivec2(1, 1) - ivec2(step(x, y));
}
ivec3 greaterThan(bvec3 x, bvec3 y)
{
    return ivec3(1, 1, 1) - ivec3(step(x, y));
}
ivec4 greaterThan(bvec4 x, bvec4 y)
{
    return ivec4(1, 1, 1, 1) - ivec4(step(x, y));
}
ivec2 lessThan(bvec2 x, bvec2 y)
{
    return ivec2(1.0, 1.0) - ivec2(step(y, x));
}
ivec3 lessThan(bvec3 x, bvec3 y)
{
    return ivec3(1.0, 1.0, 1.0) - ivec3(step(y, x));
}
ivec4 lessThan(bvec4 x, bvec4 y)
{
    return ivec4(1.0, 1.0, 1.0, 1.0) - ivec4(step(y, x));
}
ivec2 greaterThanEqual(bvec2 x, bvec2 y)
{
    return ivec2(step(y, x));
}
ivec3 greaterThanEqual(bvec3 x, bvec3 y)
{
    return ivec3(step(y, x));
}
ivec4 greaterThanEqual(bvec4 x, bvec4 y)
{
    return ivec4(step(y, x));
}
ivec2 lessThanEqual(bvec2 x, bvec2 y)
{
    return ivec2(step(x, y));
}
ivec3 lessThanEqual(bvec3 x, bvec3 y)
{
    return ivec3(step(x, y));
}
ivec4 lessThanEqual(bvec4 x, bvec4 y)
{
    return ivec4(step(x, y));
}
ivec2 equal(bvec2 x, bvec2 y)
{
    return ivec2(step(x, y) * step(y, x));
}
ivec3 equal(bvec3 x, bvec3 y)
{
    return ivec3(step(y, x) * step(y, x));
}
ivec4 equal(bvec4 x, bvec4 y)
{
    return ivec4(step(y, x) * step(y, x));
}
ivec2 notEqual(bvec2 x, bvec2 y)
{
    return (ivec2(1, 1) - ivec2(step(x, y))) * (ivec2(1, 1) - ivec2(step(y, x)));
}
ivec3 notEqual(bvec3 x, bvec3 y)
{
    return (ivec3(1, 1, 1) - ivec3(step(x, y))) * (ivec3(1, 1, 1) - ivec3(step(y, x)));
}
ivec4 notEqual(bvec4 x, bvec4 y)
{
    return (ivec4(1, 1, 1, 1) - ivec4(step(x, y))) * (ivec4(1, 1, 1, 1) - ivec4(step(y, x)));
}
//---end---
//----------------------------------------
