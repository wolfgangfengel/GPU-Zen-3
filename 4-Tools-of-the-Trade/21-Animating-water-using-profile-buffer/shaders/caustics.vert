#version 430 core

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out vec3 fNormal;

out vec3 oldPos;
out vec3 newPos;
out vec3 ray;

layout(location = 0) uniform mat4 WorldMatrix;
layout(location = 1) uniform mat4 ViewMatrix;
layout(location = 2) uniform mat4 ProjectionMatrix;
layout(location = 3) uniform mat3 NormalMatrix;

vec3 light = vec3(2.0 / 3, 2.0 /3 , -1.0 /3);

const float IOR_AIR = 1.0;
const float IOR_WATER = 1.333;
const float poolHeight = 1.0;

vec2 intersectCube(vec3 origin, vec3 ray, vec3 cubeMin, vec3 cubeMax) {
    vec3 tMin = (cubeMin - origin) / ray;
    vec3 tMax = (cubeMax - origin) / ray;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

 /* project the ray onto the plane */
vec3 project(vec3 origin, vec3 ray, vec3 refractedLight) {
	vec2 tcube = intersectCube(origin, ray, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
	origin += ray * tcube.y;
	float tplane = (-origin.y - 1.0) / refractedLight.y;
	return origin + refractedLight * tplane;
}

float waterlevel(float u, float v)
{
	float d = sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 3.1415926;
    return cos(d * 30.0f) * 0.01 * (3.1415926 - d) - 0.1;
}

void main() {

	float u = vPosition.x * 0.5 + 0.5;
	float v = vPosition.z * 0.5 + 0.5;

	float ux = u + 0.00001f;
	float vx = v;
	vec3 px = vec3(vPosition.x + 0.00001f, waterlevel(ux, vx), vPosition.z);

    float uy = u;
	float vy = v + 0.00001f;
	vec3 py = vec3(vPosition.x, waterlevel(uy, vy), vPosition.z + 0.00001f);

    vec3 p = vec3(vPosition.x, waterlevel(u, v), vPosition.z);

    vec3 normal = normalize(cross(p - px, p - py));
	//normal = vec3(0.0, 1.0, 0.0);

	/* project the vertices along the refracted vertex ray */
	vec3 refractedLight = refract(-light, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
	ray = refract(-light, normal, IOR_AIR / IOR_WATER);
	oldPos = project(vPosition.xzy, refractedLight, refractedLight);
	newPos = project(vPosition.xzy + vec3(0.0, waterlevel(u, v), 0.0), ray, refractedLight);
	
	gl_Position = vec4(0.75 * (newPos.xz + refractedLight.xz / refractedLight.y), 0.0, 1.0);


	//fNormal = NormalMatrix * vNormal;
	//gl_Position = ProjectionMatrix * ViewMatrix * WorldMatrix * vPosition;
}