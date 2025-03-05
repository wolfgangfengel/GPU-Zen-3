#version 430 core

in vec4 pos;
in vec3 normal;
in vec2 texCoord;

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;

void main() 
{
    // store the fragment position vector in the first gbuffer texture
    gPosition = pos.xyz;

    // also store the per-fragment normals into the gbuffer
    gNormal = normalize(normal);

    // and the diffuse per-fragment color
    gAlbedo.rgb = vec3(0.95);
}




















