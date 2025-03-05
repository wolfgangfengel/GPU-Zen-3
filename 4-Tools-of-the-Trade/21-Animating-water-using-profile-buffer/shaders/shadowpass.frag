#version 430 core

layout(location = 0, index = 0) out float depth;

void main() {
	depth = gl_FragCoord.z;
}