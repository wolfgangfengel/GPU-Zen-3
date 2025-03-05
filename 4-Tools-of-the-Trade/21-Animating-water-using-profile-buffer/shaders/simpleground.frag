#version 430 core

#define SHADOW_BIAS 0.005f
#define NEAR 0.12

const int numBlockerSearchSamples = 64;
const int numPCFSamples = 64;

const vec3 underwaterColor = vec3(0.4, 0.9, 1.0);

uniform vec3			lightDir;
uniform mat4			shadowMatrix;
uniform sampler2D		groundTex;
uniform sampler2D		shadowTex;
uniform sampler1D		distribution0;
uniform sampler1D		distribution1;
uniform sampler2D		layerDataTex;
uniform sampler2D		depthDataTex;
uniform float			xTexScale;
uniform int				largestSide;

in vec4 pos;
in vec3 normal;
in vec2 texCoord;

out vec4 FragColor;

vec2 RandomDirection(sampler1D distribution, float u)
{
   return texture(distribution, u).xy * 2 - vec2(1);
}


float SearchWidth(float uvLightSize, float receiverDistance)
{
	return uvLightSize * (receiverDistance - NEAR);
}


float FindBlockerDistance_DirectionalLight(vec3 shadowCoords, sampler2D shadowMap, float uvLightSize)
{
	int blockers = 0;
	float avgBlockerDistance = 0;
	float searchWidth = SearchWidth(uvLightSize, shadowCoords.z);
	for (int i = 0; i < numBlockerSearchSamples; i++)
	{
		float z = texture(shadowMap, shadowCoords.xy + RandomDirection(distribution0, i / float(numBlockerSearchSamples)) * searchWidth).r;
		if (z < (shadowCoords.z - SHADOW_BIAS))
		{
			blockers++;
			avgBlockerDistance += z;
		}
	}
	if (blockers > 0)
		return avgBlockerDistance / blockers;
	else
		return -1;
}

float PCF_DirectionalLight(vec3 shadowCoords, sampler2D shadowMap, float uvRadius)
{
	float sum = 0;
	for (int i = 0; i < numPCFSamples; i++)
	{
		float z = texture(shadowMap, shadowCoords.xy + RandomDirection(distribution1, i / float(numPCFSamples)) * uvRadius).r;
		sum += (z < (shadowCoords.z - SHADOW_BIAS)) ? 1 : 0;
	}
	return sum / numPCFSamples;
}

float PCSS_DirectionalLight(vec3 shadowCoords, sampler2D shadowMap, float uvLightSize)
{
	// blocker search
	float blockerDistance = FindBlockerDistance_DirectionalLight(shadowCoords, shadowMap, uvLightSize);
	if (blockerDistance == -1)
		return 1;		

	// penumbra estimation
	float penumbraWidth = (shadowCoords.z - blockerDistance) / blockerDistance;

	// percentage-close filtering
	float uvRadius = penumbraWidth * uvLightSize * NEAR / shadowCoords.z;
	return 1 - PCF_DirectionalLight(shadowCoords, shadowMap, uvRadius);
}

vec3 ShadowCoords(mat4 shadowMapViewProjection)
{
	vec4 projectedCoords = shadowMapViewProjection * pos;
	vec3 shadowCoords = projectedCoords.xyz / projectedCoords.w;
	shadowCoords = shadowCoords * 0.5 + 0.5;
	return shadowCoords;
}


void main() 
{
	const float directIntensity = 0.8f;
	const float ambientIntensity = 0.6f;

	// diffuse
	float diffuseLight = max(0, dot(normal, lightDir)) * directIntensity;
	vec3 textureColor = texture2D(groundTex, texCoord).xyz;

	textureColor = vec3(0.5);

	// soft shadow
	float visibility = PCSS_DirectionalLight(ShadowCoords(shadowMatrix), shadowTex, 0.15f);

	// ambient
	float ambient = ambientIntensity;

	// final
	FragColor = vec4(textureColor * (diffuseLight * visibility + ambient), 1.0f);

	// underwater
	float waterDepth = texture(depthDataTex, vec2(pos.x * xTexScale, pos.z) + vec2(0.5f, 0.5f) / largestSide).x - pos.y;
	if (waterDepth > 0)
	{
		FragColor.rgb *= underwaterColor * 1.2f;
	}
}




















