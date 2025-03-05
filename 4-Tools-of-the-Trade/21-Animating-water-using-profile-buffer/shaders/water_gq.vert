#version 430 core

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;

out vec4 pos;
out vec3 normal;
out vec2 texCoord;
out vec3 fNdc;

uniform mat4 cameraMatrix;

uniform sampler2D		layerDataTex;
uniform sampler2D		pbOFieldTex;
uniform sampler2D		pbNFieldTex;

uniform float			ampla;
uniform float			L;

uniform float			TAU;
uniform float			DOMAIN_SCALE;
uniform int				SEG_PER_DIR;
uniform int				FINE_DIR_NUM;

void main() 
{
	pos = inPosition;
	texCoord = inTexCoord;
	normal = inNormal;

	vec4 dc = cameraMatrix * inPosition;
	fNdc = dc.xyz / dc.w;

	// read pre-computed information 
	vec4 layeredData = texture2D(layerDataTex, vec2(pos.x, pos.z));
	float delta_mag = layeredData.y;
	float vx = layeredData.z * 2.0f - 1.0f;
	float vz = layeredData.w * 2.0f - 1.0f;
	float mag = sqrt(vx * vx + vz * vz);
	
	vec3 offset = vec3(0.0f, 0.0f, 0.0f);

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
			vec4 pb = (texture(pbOFieldTex, vec2(frac, 0.5f)) * 2.0f - vec4(1.0f)) * 10.0f;
			vec2 pbb =  (1.f - scale) * pb.xy + scale * pb.wz;

			// compute weight associate with current wave direction
			float diffa = abs(trans_a - angle);
			float anglem = min(min(diffa, TAU - diffa), ANGLE_HW);
			float w = 1.f - anglem / ANGLE_HW;

			vec2 ttoffset = int_weight * ampla * w * pbb * A;
			// summation
			offset[0] += ttoffset[0] * kdir[1];
			offset[1] += ttoffset[1];
			offset[2] += ttoffset[0] * kdir[0];
		}
	}
	if(mag==0.) offset = vec3(0, 0, 0);
	gl_Position = cameraMatrix * (inPosition + vec4(offset * 0.01f, 0));
}