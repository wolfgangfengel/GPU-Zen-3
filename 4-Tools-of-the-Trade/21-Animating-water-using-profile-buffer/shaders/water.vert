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
	const float dx = TAU / FINE_DIR_NUM;
	pos = inPosition;
	texCoord = inTexCoord;
	normal = inNormal;

	vec4 dc = cameraMatrix * inPosition;
	fNdc = dc.xyz / dc.w;

	vec4 layeredData = texture2D(layerDataTex, vec2(pos.x, pos.z));
	float delta_mag = layeredData.y;
	float vx = layeredData.z * 2.0f - 1.0f;
	float vz = layeredData.w * 2.0f - 1.0f;
	float mag = sqrt(vx * vx + vz * vz);
	
	vec3 offset = vec3(0.0f, 0.0f, 0.0f);
	if (mag != 0.f)
	{
		int INT_NUM = min(16, 4 + int(500. * (2. * delta_mag / TAU) ));
		int SEG_HW = SEG_PER_DIR * INT_NUM;
		float ANGLE_HW = float(SEG_HW) / float(FINE_DIR_NUM) * TAU;

		float scale = min(1.f, mag);

		float angle = atan(vx, vz);

		if (angle < 0.f) angle += TAU;

		int angleid = int(floor(angle / TAU * FINE_DIR_NUM));
	
		vec2 v_pos = pos.zx * DOMAIN_SCALE;
		
		const float da = 1.0f / FINE_DIR_NUM;
		
		for (int i = -SEG_HW; i < SEG_HW; ++i)
		{
			float a = da * (angleid + i);

			if (a < 0.0f) a += 1.0f;
			if (a > 1.0f) a -= 1.0f;

			float trans_a = a * TAU;
			float int_weight = dx;

			vec2 kdir = vec2(cos(trans_a), sin(trans_a));
			float kdir_x = dot(v_pos, kdir) + TAU * sin(4023432 * trans_a);

			float weight = kdir_x / L;
			float frac = weight - floor(weight);
			vec4 pb = (texture(pbOFieldTex, vec2(frac, 0.5f)) * 2.0f - vec4(1.0f)) * 10.0f;
			vec2 pbb = (1.f - scale) * pb.xy + scale * pb.wz;

			float diffa = abs(trans_a - angle);
			float anglem = min(min(diffa, TAU - diffa), ANGLE_HW);
			float w = 1.f - anglem / ANGLE_HW;

			vec2 ttoffset = int_weight * ampla * w * pbb;

			offset[2] += ttoffset[0] * kdir[0];
			offset[1] += ttoffset[1];
			offset[0] += ttoffset[0] * kdir[1];
		}
	}
	if(mag==0.) offset = vec3(0, 0, 0);
	gl_Position = cameraMatrix * (inPosition + vec4(offset * 0.01f, 0));
}