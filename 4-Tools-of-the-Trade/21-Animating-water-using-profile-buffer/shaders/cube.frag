#version 430 core

const vec3 underwaterColor = vec3(0.4, 0.9, 1.0);

in vec4 fModelPosition;
in vec4 fWorldPosition;
in vec3 fNormal;
in vec2 fTexCoord;

out vec4 FragColor;

layout(location = 4) uniform mat4 WaterMapProjectionMatrix;
layout(location = 5) uniform mat4 WaterMapViewMatrix;
layout(location = 6) uniform vec3 LightPosition;
layout(location = 7) uniform float TextureScale;
layout(location = 8) uniform float Time;

layout(binding = 0) uniform sampler2D WaterMapDepth;
layout(binding = 1) uniform sampler2D WaterMapNormals;
layout(binding = 2) uniform sampler2D Texture;
layout(binding = 3) uniform sampler2D NoiseNormalTexture;
layout(binding = 4) uniform sampler2D CausticTexture;
layout(binding = 5) uniform sampler1D SubSurfaceScatteringTexture;

vec3 light = vec3(2.0 / 3, 2.0 /3 , -1.0 /3);
vec3 sphereCenter = vec3(-0.4, -0.75, 0.2);
float sphereRadius = 0.25;

const float IOR_AIR = 1.0;
const float IOR_WATER = 1.333;
const float poolHeight = 1.0;

float waterlevel(float u, float v)
{
	float d = sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 3.1415926;
    return cos(d * 30.0f) * 0.01 * (3.1415926 - d) - 0.1;
}

vec2 intersectCube(vec3 origin, vec3 ray, vec3 cubeMin, vec3 cubeMax) {
    vec3 tMin = (cubeMin - origin) / ray;
    vec3 tMax = (cubeMax - origin) / ray;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}


float inverseDepthRangeTransformation(float depth) {
    return (2.0 * depth - gl_DepthRange.near - gl_DepthRange.far) /
            (gl_DepthRange.far - gl_DepthRange.near);
}

vec3 positionFromDepth(vec2 texCoord, sampler2D depthTexture, mat4 inverseViewProjection) {
    vec4 n = vec4(texCoord * 2.0 - 1.0, 0.0, 0.0);
    float depth = texture2D(depthTexture, texCoord).r;
    n.z = inverseDepthRangeTransformation(depth);
    n.w = 1.0;
    vec4 worldPosition = inverseViewProjection * n;

    return worldPosition.xyz / worldPosition.w;
}

vec3 getWaterWorldPosition(vec2 texCoord) {
	mat4 inverseViewProjection = inverse(WaterMapProjectionMatrix * WaterMapViewMatrix);
	return positionFromDepth(texCoord, WaterMapDepth, inverseViewProjection);
}

vec3 decodeNormal(vec4 normal) {
    return vec3(normal.xyz * 2.0 - 1.0);
}

vec3 getWallColor(vec3 point) {
    float scale = 0.5;
    
    vec3 wallColor;
    vec3 normal;
    if (abs(point.x) > 0.999) {
      wallColor = texture2D(Texture, point.yz * 0.5 + vec2(1.0, 0.5)).rgb;
      normal = vec3(-point.x, 0.0, 0.0);
    } else if (abs(point.z) > 0.999) {
      wallColor = texture2D(Texture, point.yx * 0.5 + vec2(1.0, 0.5)).rgb;
      normal = vec3(0.0, 0.0, -point.z);
    } else {
      wallColor = texture2D(Texture, point.xz * 0.5 + 0.5).rgb;
      normal = vec3(0.0, 1.0, 0.0);
    }
    
    scale /= length(point); /* pool ambient occlusion */
    scale *= 1.0 - 0.9 / pow(length(point - sphereCenter) / sphereRadius, 4.0); /* sphere ambient occlusion */
    
    /* caustics */
#if 1
    vec3 refractedLight = -refract(-light, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    float diffuse = max(0.0, dot(refractedLight, normal));
    if (point.y < waterlevel(point.x * 0.5f + 0.5f, point.z * 0.5f + 0.5f)) {
        vec4 caustic = texture2D(WaterMapNormals, 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5);
        scale += diffuse * caustic.r * 2.0  * caustic.g;
    } else {
        /* shadow for the rim of the pool */
        vec2 t = intersectCube(point, refractedLight, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
        diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
        scale += diffuse * 0.5;
    }
#endif

    return wallColor * scale;
}

void main() {

    float u = fModelPosition.x * 0.5f + 0.5f;
	float v = fModelPosition.z * 0.5f + 0.5f;

    FragColor = vec4(getWallColor(fModelPosition.xyz), 1.0f);

	if (fModelPosition.y < waterlevel(u, v)) 
		FragColor.rgb *= underwaterColor * 1.2f;
        

    //vec4 testColor = vec4(texture2D(WaterMapNormals, fModelPosition.xz * 0.5 + vec2(0.5, 0.5)).g);
    //FragColor = testColor;
}




















