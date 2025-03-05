#version 430 core

in vec2 fTexCoord;

out vec4 FragColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D ssao;

void main() {

	FragColor = vec4(texture2D(gPosition, fTexCoord).xyz, 1);
	FragColor += vec4(texture2D(gNormal, fTexCoord).xyz, 1);
	FragColor += vec4(texture2D(gAlbedo, fTexCoord).xyz, 1);

	//FragColor = vec4(vec3(texture2D(ssao, fTexCoord).r), 1);
	FragColor += vec4(texture2D(ssao, fTexCoord).xyz, 1);

	//FragColor = vec4(fTexCoord, 0, 1);
}




















