#version 430 core

const float IOR_AIR = 1.0;
const float IOR_WATER = 1.333;
const vec3 abovewaterColor = vec3(0.9, 1.0, 1.25);
const vec3 underwaterColor = vec3(0.4, 0.9, 1.0);
const float normalNoiseStrength = 0.6f;	
const float normalNoiseScale = 10.0f;

out vec4 FragColor;

in vec4 pos;
in vec3 normal;
in vec2 texCoord;
in vec3 fNdc;

uniform float			xTexScale;
uniform int				largestSide;
uniform float			time;
uniform vec3			lightDir;
uniform mat4			cameraMatrix;
uniform samplerCube		skyCubeTex;
uniform sampler2D		layerDataTex;
uniform sampler2D		depthDataTex;
uniform sampler2D		foamDataTex;
uniform sampler2D		topdownTex;
uniform sampler2D		normNoiseTex;
uniform sampler2D		noiseTex;
uniform sampler2D		waterNoiseTex;

uniform sampler2D		pbNFieldTex;

uniform float			cameraDistance;
uniform float			ampla;
uniform float			L;

uniform float			TAU;
uniform float			DOMAIN_SCALE;
uniform int				SEG_PER_DIR;
uniform int				FINE_DIR_NUM;


vec3 getRayColor(vec3 origin, vec3 ray, vec3 waterColor) 
{
	vec3 color;
#if 0
	color = texture(skyCubeTex, ray).rgb;
	color += vec3(pow(max(0.0, dot(lightDir, ray)), 5000.0)) * vec3(10.0, 8.0, 6.0);
	return color;
#endif
#if 1
	vec3 info = vec3(0);
	bool hit = false;
	for (int i = 0; i < 1000; i++)
	{
		vec3 p = origin + i * ray * 0.001f;
		if (p.x < 0 || p.x > 1 ||p.z < 0 || p.z > 1) 
		{
			if (p.y < 0)
				return vec3(0,0,0);
			else
				break;
		}
		info = texture(layerDataTex, vec2(p.x * xTexScale, pos.z) + vec2(0.5f) / largestSide).xyz;
		if (info.y > p.y)
		{
			hit = true;
			color = texture(topdownTex, p.zx).xyz;
			break;
		}
	}
	if (!hit)
	{
		color = texture(skyCubeTex, ray).rgb;
		color += vec3(pow(max(0.0, dot(lightDir, ray)), 5000.0)) * vec3(10.0, 8.0, 6.0);
	}
    if (ray.y < info.x) color *= waterColor;
	return color;
#endif
}

void main() 
{
	// read pre-computed information, including velocity, foam, depth, etc. 
	float foam = texture2D(foamDataTex, vec2(pos.x, pos.z)).x;
	float depth = texture2D(depthDataTex, vec2(pos.x, pos.z)).x;

	vec3 newNormal = vec3(0.0f, 1.0f, 0.0f);
	vec4 layeredData = texture2D(layerDataTex, vec2(pos.x, pos.z));
	float delta_mag = layeredData.y;
	float vx = layeredData.z * 2.0f - 1.0f;
	float vz = layeredData.w * 2.0f - 1.0f;
	float mag = sqrt(vx * vx + vz * vz);

	// Gaussian quadrature nodes and associated weights
	float nodes[32] = {-0.9972638618494816f, -0.9856115115452684f, -0.9647622555875064f, -0.9349060759377397f, -0.8963211557660522f, -0.84936761373257f, -0.7944837959679424f, -0.7321821187402897f, -0.6630442669302152f, -0.5877157572407623f, -0.5068999089322294f, -0.42135127613063533f, -0.33186860228212767f, -0.23928736225213706f, -0.1444719615827965f, -0.04830766568773831f, 0.04830766568773831f, 0.1444719615827965f, 0.23928736225213706f, 0.33186860228212767f, 0.42135127613063533f, 0.5068999089322294f, 0.5877157572407623f, 0.6630442669302152f, 0.7321821187402897f, 0.7944837959679424f, 0.84936761373257f, 0.8963211557660522f, 0.9349060759377397f, 0.9647622555875064f, 0.9856115115452684f, 0.9972638618494816f};
	float weights[32] = {0.007018610009469298f, 0.016274394730905965f, 0.025392065309262427f, 0.034273862913021626f, 0.042835898022226426f, 0.050998059262376244f, 0.058684093478535704f, 0.06582222277636175f, 0.07234579410884845f, 0.07819389578707031f, 0.08331192422694685f, 0.08765209300440391f, 0.09117387869576386f, 0.09384439908080457f, 0.09563872007927483f, 0.09654008851472781f, 0.09654008851472781f, 0.09563872007927483f, 0.09384439908080457f, 0.09117387869576386f, 0.08765209300440391f, 0.08331192422694685f, 0.07819389578707031f, 0.07234579410884845f, 0.06582222277636175f, 0.058684093478535704f, 0.050998059262376244f, 0.042835898022226426f, 0.034273862913021626f, 0.025392065309262427f, 0.016274394730905965f, 0.007018610009469298f};

	// we only handle area with non-zero flow strength, i.e., still water produces no waves.
	if (mag != 0.f)
	{
		// compute adaptive limits of integration
		int INT_NUM = min(16, 4 + int(500. * (2. * delta_mag / TAU) ));
		int SEG_HW = SEG_PER_DIR * INT_NUM;

		float ANGLE_HW = float(SEG_HW) / float(FINE_DIR_NUM) * TAU;

		int nnodes = 32;

		float scale = min(1.f, mag);

		// compute primary angle
		float angle = atan(vx, vz);
		if (angle < 0.f) angle += TAU;

		vec2 v_pos = pos.zx * DOMAIN_SCALE;
		
		const float a = 0.f;
		const float b = TAU;

		const float A = (b - a) * 0.5f;
		const float B = (b + a) * 0.5f;

		vec3 tz = vec3(0.f, 0.f, 0.f);
		vec3 tx = vec3(0.f, 0.f, 0.f);
		// integrate over all integration nodes
		for(int sid = 0; sid < nnodes; ++sid)
		{
			// transform from [-1, 1] to [0, 2*PI]
			float rel_a = nodes[sid];
			float trans_a = A * rel_a + B;
			float int_weight = weights[sid];

			// compute current wave direction
			vec2 kdir = vec2(cos(trans_a), sin(trans_a));
			float kdir_x = dot(v_pos, kdir);

			// fetch value from profile buffers
			float weight = kdir_x / L;
			float frac = weight - floor(weight);
			vec4 pb_norm = (texture(pbNFieldTex, vec2(frac, 0.5f)) * 2.0f - vec4(1.0f)) * 10.0f;
			
			// compute weight associate with current wave direction
			float diffa = abs(trans_a - angle);
			float anglem = min(min(diffa, TAU - diffa), ANGLE_HW);
			float w = 1.f - anglem / ANGLE_HW;
			vec2 pbb_norm = (1.f - scale) * pb_norm.xy + scale * pb_norm.wz;
			vec2 tt = int_weight * w * pbb_norm;

			// summation
			tz.zy += kdir.x * tt;
			tx.xy += kdir.y * tt;
		}

		tz.zy *= A * ampla;
		tx.xy *= A * ampla;
		
		tz.z += 1.f;
		tx.x += 1.f;

		// Ty = Tz x Tx
		newNormal = cross(tz, tx);
		newNormal = normalize(newNormal);
	}
	else
	{
		// FragColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
		// return;
	}
	

	///////////////////////////////////////////////////////////

	// Eye vector
	mat4 inverseViewProjection = inverse(cameraMatrix);
	vec4 eyeVectorFront = inverseViewProjection * vec4(fNdc.x, fNdc.y, -1, 1);
	vec4 eyeVectorBack  = inverseViewProjection * vec4(fNdc.x, fNdc.y,  1, 1);	
	vec3 eyeVectorFrontWDiv = eyeVectorFront.xyz / eyeVectorFront.w;
	vec3 eyeVectorBackWDiv  = eyeVectorBack.xyz / eyeVectorBack.w;	
	vec3 eyeVector = normalize(eyeVectorFrontWDiv - eyeVectorBackWDiv);	

	// Reflection
	vec3 reflectionVector = reflect(-eyeVector, newNormal);	
	//vec4 reflection = 0.8f * texture(skyCubeTex, reflectionVector);
	vec4 reflection = 0.01f * texture(skyCubeTex, reflectionVector);

	vec3 lightScattering= mix(vec3(1.0,1.0,1.0), abovewaterColor * 0.6, max(0.0f, min(1, depth * 20.0f) - 0e-6f));

	float diffuseLight = max(0, dot(newNormal, lightDir));

	vec3 incomingRay = -eyeVector;
	vec3 reflectedRay = reflect(incomingRay, newNormal);
	vec3 refractedRay = refract(incomingRay, newNormal, IOR_AIR / IOR_WATER);
    float fresnel = mix(0.25, 1.0, pow(1.0 - dot(newNormal, -incomingRay), 3.0));

	vec3 newPos = pos.xyz;
	newPos.y = 0.0f;

	newPos += newNormal * 0.01f;

	vec3 reflectedColor = getRayColor(newPos, reflectedRay, lightScattering);
    vec3 refractedColor = getRayColor(newPos, refractedRay, lightScattering);

	vec3 specular = vec3(pow(max(dot(incomingRay, reflectedRay), 0.0), 32)) * 1;  

    FragColor = vec4(mix(refractedColor, reflectedColor, fresnel) + specular + vec3(1.0,1.0,1.0) * foam, 1.0);
}