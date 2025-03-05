#version 430 core


vec3 light = vec3(2.0 / 3, 2.0 /3 , -1.0 /3);

vec3 sphereCenter = vec3(-0.4, -0.75, 0.2);
float sphereRadius = 0.25;

const float IOR_AIR = 1.0;
const float IOR_WATER = 1.333;
const float poolHeight = 1.0;

in vec3 fNormal;

in vec3 oldPos;
in vec3 newPos;
in vec3 ray;

out vec4 FragColor;

vec2 intersectCube(vec3 origin, vec3 ray, vec3 cubeMin, vec3 cubeMax) {
    vec3 tMin = (cubeMin - origin) / ray;
    vec3 tMax = (cubeMax - origin) / ray;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

vec3 encodeNormal(vec3 normal) {
    return (normalize(normal) + 1.0) * 0.5;
}

void main() {

	float oldArea = length(dFdx(oldPos)) * length(dFdy(oldPos));
	float newArea = length(dFdx(newPos)) * length(dFdy(newPos));
	FragColor = vec4(oldArea / newArea * 0.2, 1.0, 0.0, 0.0);

	//FragColor = vec4(1.2, 0.2, 0.0, 0.0);

	vec3 refractedLight = refract(-light, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
	
	/* compute a blob shadow and make sure we only draw a shadow if the player is blocking the light */
	vec3 dir = (sphereCenter - newPos) / sphereRadius;
	vec3 area = cross(dir, refractedLight);
	float shadow = dot(area, area);
	float dist = dot(dir, -refractedLight);
	shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
	shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
	shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
	FragColor.g = shadow;
	
	/* shadow for the rim of the pool */
	vec2 t = intersectCube(newPos, -refractedLight, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
	FragColor.r *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (newPos.y - refractedLight.y * t.y - 2.0 / 12.0)));

	//FragColor = vec4(1, 0, 0, 1);

}