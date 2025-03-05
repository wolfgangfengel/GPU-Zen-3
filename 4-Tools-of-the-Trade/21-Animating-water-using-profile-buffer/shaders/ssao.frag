#version 430 core

in vec2 fTexCoord;

out float FragColor;

uniform mat4 projection;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D noiseTex;

float samples[64 * 3] =
{
	0.0497709, -0.0447092, 0.0499634,
	0.0145746, 0.0165311, 0.00223862,
	-0.0406477, -0.0193748, 0.0319336,
	0.0137781, -0.091582, 0.0409242,
	0.055989, 0.0597915, 0.0576589,
	0.0922659, 0.0442787, 0.0154511,
	-0.00203926, -0.054402, 0.066735,
	-0.00033053, -0.000187337, 0.000369319,
	0.0500445, -0.0466499, 0.0253849,
	0.0381279, 0.0314015, 0.032868,
	-0.0318827, 0.0204588, 0.0225149,
	0.0557025, -0.0369742, 0.0544923,
	0.0573717, -0.0225403, 0.0755416,
	-0.0160901, -0.00376843, 0.0554733,
	-0.0250329, -0.024829, 0.0249512,
	-0.0336879, 0.0213913, 0.0254024,
	-0.0175298, 0.0143856, 0.00534829,
	0.0733586, 0.112052, 0.0110145,
	-0.0440559, -0.0902836, 0.083683,
	-0.0832772, -0.00168341, 0.0849867,
	-0.0104057, -0.0328669, 0.019273,
	0.00321131, -0.00488206, 0.00416381,
	-0.00738321, -0.0658346, 0.067398,
	0.0941413, -0.00799846, 0.14335,
	0.0768329, 0.126968, 0.106999,
	0.000392719, 0.000449695, 0.00030161,
	-0.104793, 0.0654448, 0.101737,
	-0.00445152, -0.119638, 0.161901,
	-0.0745526, 0.0344493, 0.224138,
	-0.0027583, 0.00307776, 0.00292255,
	-0.108512, 0.142337, 0.166435,
	0.046882, 0.103636, 0.0595757,
	0.134569, -0.0225121, 0.130514,
	-0.16449, -0.155644, 0.12454,
	-0.187666, -0.208834, 0.0577699,
	-0.043722, 0.0869255, 0.0747969,
	-0.00256364, -0.00200082, 0.00406967,
	-0.0966957, -0.182259, 0.299487,
	-0.225767, 0.316061, 0.089156,
	-0.0275051, 0.287187, 0.317177,
	0.207216, -0.270839, 0.110132,
	0.0549017, 0.104345, 0.323106,
	-0.13086, 0.119294, 0.280219,
	0.154035, -0.0653706, 0.229842,
	0.0529379, -0.227866, 0.148478,
	-0.187305, -0.0402247, 0.0159264,
	0.141843, 0.0471631, 0.134847,
	-0.0442676, 0.0556155, 0.0558594,
	-0.0235835, -0.0809697, 0.21913,
	-0.142147, 0.198069, 0.00519361,
	0.158646, 0.230457, 0.0437154,
	0.03004, 0.381832, 0.163825,
	0.083006, -0.309661, 0.0674131,
	0.226953, -0.23535, 0.193673,
	0.381287, 0.332041, 0.529492,
	-0.556272, 0.294715, 0.301101,
	0.42449, 0.00564689, 0.117578,
	0.3665, 0.00358836, 0.0857023,
	0.329018, 0.0308981, 0.178504,
	-0.0829377, 0.512848, 0.0565553,
	0.867363, -0.00273376, 0.100138,
	0.455745, -0.772006, 0.0038413,
	0.417291, -0.154846, 0.462514,
	-0.442722, -0.679282, 0.186503
};

// parameters (you'd probably want to use them as uniforms to more easily tweak the effect)
int kernelSize = 64;
float radius = 0.25;
float bias = 0.025;

// tile noise texture over screen based on screen dimensions divided by noise size
const vec2 noiseScale = vec2(1920.0/4.0, 1080.0/4.0); 

void main()
{
    // get input for SSAO algorithm
    vec3 fragPos = texture(gPosition, fTexCoord).xyz;
	vec3 normal = normalize(texture(gNormal, fTexCoord).rgb);
	vec3 randomVec = normalize(texture(noiseTex, fTexCoord * noiseScale).xyz);

	// create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
	
	// iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * vec3(samples[i * 3], samples[i * 3 + 1], samples[i * 3 + 2]);//samples[i]; // from tangent to view-space
        samplePos = fragPos + samplePos * radius; 
        
        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
        
        // get sample depth
        float sampleDepth = texture(gPosition, offset.xy).z; // get depth value of kernel sample
        
        // range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }
    occlusion = 1.0 - (occlusion / kernelSize);

	FragColor = occlusion;
}



















