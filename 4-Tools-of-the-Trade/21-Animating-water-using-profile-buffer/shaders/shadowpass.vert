#version 430 core

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;

uniform mat4 cameraMatrix;

void main() {
	gl_Position = cameraMatrix * inPosition;
}