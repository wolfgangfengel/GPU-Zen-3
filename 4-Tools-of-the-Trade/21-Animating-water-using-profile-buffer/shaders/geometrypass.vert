#version 430 core

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;

out vec4 pos;
out vec3 normal;
out vec2 texCoord;

uniform mat4 viewMatrix;
uniform mat4 cameraMatrix;

void main() 
{
	pos = viewMatrix * inPosition;
	texCoord = inTexCoord;

	mat3 normalMatrix = transpose(inverse(mat3(viewMatrix)));
    normal = normalMatrix * inNormal;

	gl_Position = cameraMatrix * inPosition;
}