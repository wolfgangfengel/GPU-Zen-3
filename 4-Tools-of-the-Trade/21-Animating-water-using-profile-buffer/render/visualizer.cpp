
#define STB_IMAGE_IMPLEMENTATION
#pragma warning( disable : 26451 26453 6001 6262 6385)
#include <stb_image.h>

#include "visualizer.h"

#include "PoissonGenerator.h"

#include "common/Defines.h"

float ourLerp(float a, float b, float f)
{
	return a + f * (b - a);
}

void Geometry::initialize() {
	Vertex unitQuadVertices[6];
	unitQuadVertices[0].position = Vec3(-1, -1, 0);
	unitQuadVertices[0].normal = Vec3(0, 1, 0);
	unitQuadVertices[0].texCoord = Vec2(0, 0);

	unitQuadVertices[1].position = Vec3(1, -1, 0);
	unitQuadVertices[1].normal = Vec3(0, 1, 0);
	unitQuadVertices[1].texCoord = Vec2(1, 0);

	unitQuadVertices[2].position = Vec3(1, 1, 0);
	unitQuadVertices[2].normal = Vec3(0, 1, 0);
	unitQuadVertices[2].texCoord = Vec2(1, 1);

	unitQuadVertices[3].position = Vec3(-1, -1, 0);
	unitQuadVertices[3].normal = Vec3(0, 1, 0);
	unitQuadVertices[3].texCoord = Vec2(0, 0);

	unitQuadVertices[4].position = Vec3(1, 1, 0);
	unitQuadVertices[4].normal = Vec3(0, 1, 0);
	unitQuadVertices[4].texCoord = Vec2(1, 1);

	unitQuadVertices[5].position = Vec3(-1, 1, 0);
	unitQuadVertices[5].normal = Vec3(0, 1, 0);
	unitQuadVertices[5].texCoord = Vec2(0, 1);

	// Array Buffer
	glGenBuffers(1, &arrayBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unitQuadVertices), unitQuadVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Vertex Array Object
	glGenVertexArrays(1, &vertexArrayObject);
	setAttribPointer(vertexArrayObject, Attributes::Position, arrayBuffer, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
	setAttribPointer(vertexArrayObject, Attributes::TexCoord, arrayBuffer, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, texCoord));
}

void Geometry::render() {
	glBindVertexArray(vertexArrayObject);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
}

unsigned char* importImage(const std::string& filename) {
	int x, y, n;
	return stbi_load(filename.c_str(), &x, &y, &n, 4);
}

GLuint createCubemap() {
	GLuint id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_CUBE_MAP, id);

	auto imagePositiveX = importImage("../textures/Sky/sky01/posx.jpg");
	auto imageNegativeX = importImage("../textures/Sky/sky01/negx.jpg");
	auto imagePositiveY = importImage("../textures/Sky/sky01/posy.jpg");
	auto imageNegativeY = importImage("../textures/Sky/sky01/negy.jpg");
	auto imagePositiveZ = importImage("../textures/Sky/sky01/posz.jpg");
	auto imageNegativeZ = importImage("../textures/Sky/sky01/negz.jpg");

	auto size = 512;

	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePositiveX);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageNegativeX);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePositiveY);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageNegativeY);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagePositiveZ);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageNegativeZ);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	return id;
}

GLuint importTexture(const std::string& filename) {
	int width, height, n;
	unsigned char* dataPtr = stbi_load(filename.c_str(), &width, &height, &n, 4);

	auto pixelCount = width * height;

	std::vector<unsigned char> data(dataPtr, dataPtr + pixelCount * 4);

	GLuint id;
	glGenTextures(1, &id);
	//glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dataPtr);

	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	return id;
}

void createPoissonDiscDistribution(GLuint texture, size_t numSamples)
{
	PoissonGenerator::DefaultPRNG PRNG;
	auto points = PoissonGenerator::GeneratePoissonPoints(numSamples * 2, PRNG);
	size_t attempts = 0;
	while (points.size() < numSamples && ++attempts < 100)
		points = PoissonGenerator::GeneratePoissonPoints(numSamples * 2, PRNG);
	if (attempts == 100)
	{
		std::cout << "couldn't generate Poisson-disc distribution with " << numSamples << " samples" << std::endl;
		numSamples = points.size();
	}
	std::vector<float> data(numSamples * 2);
	for (auto i = 0, j = 0; i < numSamples; i++, j += 2)
	{
		auto& point = points[i];
		data[j] = point.x;
		data[j + 1] = point.y;
	}
	glBindTexture(GL_TEXTURE_1D, texture);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RG, (int)numSamples, 0, GL_RG, GL_FLOAT, &data[0]);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void Visualizer::initialize(const int frame_w, const int frame_h, const int world_w, const int world_h, const float cameraDistance, const float ampla, const float L)
{
	this->cameraDistance = cameraDistance;
	this->ampla = ampla;
	this->L = L;
	if (
		!s_ground.BuildProgramFiles("../shaders/ground.vert", "../shaders/ground.frag") ||
		!s_simpleground.BuildProgramFiles("../shaders/simpleground.vert", "../shaders/simpleground.frag") ||
		//!s_water.BuildProgramFiles			("../shaders/water.vert",			"../shaders/water.frag") ||
		!s_water.BuildProgramFiles("../shaders/water_gq.vert", "../shaders/water_gq.frag") ||
		//!s_water.BuildProgramFiles			("../shaders/water.vert",			"../shaders/water_gq.frag") ||
		!s_shadowPass.BuildProgramFiles("../shaders/shadowPass.vert", "../shaders/shadowPass.frag") ||
		!s_geometryPass.BuildProgramFiles("../shaders/geometryPass.vert", "../shaders/geometryPass.frag") ||
		!s_displayBuffer.BuildProgramFiles("../shaders/displaybuffer.vert", "../shaders/displaybuffer.frag") ||
		!s_ssao.BuildProgramFiles("../shaders/ssao.vert", "../shaders/ssao.frag") ||
		!s_ssaoBlur.BuildProgramFiles("../shaders/ssao.vert", "../shaders/ssaoblur.frag")
		)
		return;

	s_ground.RegisterParam(SHADER_PARAM_cameraMatrix, "cameraMatrix");
	s_ground.RegisterParam(SHADER_PARAM_shadowMatrix, "shadowMatrix");
	s_ground.RegisterParam(SHADER_PARAM_lightDir, "lightDir");
	s_ground.RegisterParam(SHADER_PARAM_shadowTex, "shadowTex");
	s_ground.RegisterParam(SHADER_PARAM_dist0Tex, "distribution0");
	s_ground.RegisterParam(SHADER_PARAM_dist1Tex, "distribution1");
	s_ground.RegisterParam(SHADER_PARAM_ssaoTex, "ssao");
	s_ground.RegisterParam(SHADER_PARAM_layerDatalTex, "layerDataTex");
	s_ground.RegisterParam(SHADER_PARAM_xTexScale, "xTexScale");
	s_ground.RegisterParam(SHADER_PARAM_largestSide, "largestSide");
	s_ground.BindProgram();
	s_ground.SetParam(SHADER_PARAM_shadowTex, TEXTURE_SHADOW);
	s_ground.SetParam(SHADER_PARAM_dist0Tex, TEXTURE_DIST0);
	s_ground.SetParam(SHADER_PARAM_dist1Tex, TEXTURE_DIST1);
	s_ground.SetParam(SHADER_PARAM_ssaoTex, TEXTURE_SSAO);
	s_ground.SetParam(SHADER_PARAM_layerDatalTex, TEXTURE_LAYERDATA);

	s_simpleground.RegisterParam(SHADER_PARAM_cameraMatrix, "cameraMatrix");
	s_simpleground.RegisterParam(SHADER_PARAM_shadowMatrix, "shadowMatrix");
	s_simpleground.RegisterParam(SHADER_PARAM_lightDir, "lightDir");
	s_simpleground.RegisterParam(SHADER_PARAM_shadowTex, "shadowTex");
	s_simpleground.RegisterParam(SHADER_PARAM_dist0Tex, "distribution0");
	s_simpleground.RegisterParam(SHADER_PARAM_dist1Tex, "distribution1");
	s_simpleground.RegisterParam(SHADER_PARAM_layerDatalTex, "layerDataTex");
	s_simpleground.RegisterParam(SHADER_PARAM_xTexScale, "xTexScale");
	s_simpleground.RegisterParam(SHADER_PARAM_largestSide, "largestSide");
	s_simpleground.BindProgram();
	s_simpleground.SetParam(SHADER_PARAM_shadowTex, TEXTURE_SHADOW);
	s_simpleground.SetParam(SHADER_PARAM_dist0Tex, TEXTURE_DIST0);
	s_simpleground.SetParam(SHADER_PARAM_dist1Tex, TEXTURE_DIST1);
	s_simpleground.SetParam(SHADER_PARAM_ssaoTex, TEXTURE_SSAO);
	s_simpleground.SetParam(SHADER_PARAM_layerDatalTex, TEXTURE_LAYERDATA);

	s_water.RegisterParam(SHADER_PARAM_time, "time");
	s_water.RegisterParam(SHADER_PARAM_lightDir, "lightDir");
	s_water.RegisterParam(SHADER_PARAM_cameraMatrix, "cameraMatrix");
	s_water.RegisterParam(SHADER_PARAM_skyCubeTex, "skyCubeTex");
	s_water.RegisterParam(SHADER_PARAM_layerDatalTex, "layerDataTex");
	s_water.RegisterParam(SHADER_PARAM_depthDatalTex, "depthDataTex");
	s_water.RegisterParam(SHADER_PARAM_foamDatalTex, "foamDataTex");
	s_water.RegisterParam(SHADER_PARAM_topdownTex, "topdownTex");
	s_water.RegisterParam(SHADER_PARAM_normNoiseTex, "normNoiseTex");
	s_water.RegisterParam(SHADER_PARAM_waterNoiseTex, "waterNoiseTex");
	s_water.RegisterParam(SHADER_PARAM_xTexScale, "xTexScale");
	s_water.RegisterParam(SHADER_PARAM_largestSide, "largestSide");
	s_water.RegisterParam(SHADER_PARAM_pbNormFieldTex, "pbNFieldTex");
	s_water.RegisterParam(SHADER_PARAM_pbOffsetFieldTex, "pbOFieldTex");
	s_water.RegisterParam(SHADER_PARAM_cameraDistance, "cameraDistance");
	s_water.RegisterParam(SHADER_PARAM_ampla, "ampla");
	s_water.RegisterParam(SHADER_PARAM_L, "L");

	s_water.RegisterParam(SHADER_PARAM_TAU, "TAU");
	s_water.RegisterParam(SHADER_PARAM_DOMAIN_SCALE, "DOMAIN_SCALE");
	s_water.RegisterParam(SHADER_PARAM_SEG_PER_DIR, "SEG_PER_DIR");
	s_water.RegisterParam(SHADER_PARAM_FINE_DIR_NUM, "FINE_DIR_NUM");

	s_water.BindProgram();
	s_water.SetParam(SHADER_PARAM_skyCubeTex, TEXTURE_SKYCUBE);
	s_water.SetParam(SHADER_PARAM_layerDatalTex, TEXTURE_LAYERDATA);
	s_water.SetParam(SHADER_PARAM_depthDatalTex, TEXTURE_DEPTHDATA);
	s_water.SetParam(SHADER_PARAM_foamDatalTex, TEXTURE_FOAMDATA);
	s_water.SetParam(SHADER_PARAM_topdownTex, TEXTURE_TOPDOWN);
	s_water.SetParam(SHADER_PARAM_normNoiseTex, TEXTURE_NORMNOISE);
	s_water.SetParam(SHADER_PARAM_waterNoiseTex, TEXTURE_WATERNOISE);
	s_water.SetParam(SHADER_PARAM_pbNormFieldTex, TEXTURE_PBNORMFIELD);
	s_water.SetParam(SHADER_PARAM_pbOffsetFieldTex, TEXTURE_PBOFFSETFIELD);
	s_water.SetParam(SHADER_PARAM_cameraDistance, this->cameraDistance);
	s_water.SetParam(SHADER_PARAM_ampla, this->ampla);
	s_water.SetParam(SHADER_PARAM_L, this->L);

	s_water.SetParam(SHADER_PARAM_TAU, TAU);
	s_water.SetParam(SHADER_PARAM_DOMAIN_SCALE, DOMAIN_SCALE);
	s_water.SetParam(SHADER_PARAM_SEG_PER_DIR, SEG_PER_DIR);
	s_water.SetParam(SHADER_PARAM_FINE_DIR_NUM, FINE_DIR_NUM);

	s_shadowPass.RegisterParam(SHADER_PARAM_cameraMatrix, "cameraMatrix");

	s_geometryPass.RegisterParam(SHADER_PARAM_cameraMatrix, "cameraMatrix");
	s_geometryPass.RegisterParam(SHADER_PARAM_viewMatrix, "viewMatrix");

	s_ssao.RegisterParam(SHADER_PARAM_projectionMatrix, "projection");
	s_ssao.RegisterParam(SHADER_PARAM_gPositionTex, "gPosition");
	s_ssao.RegisterParam(SHADER_PARAM_gNormalTex, "gNormal");
	s_ssao.RegisterParam(SHADER_PARAM_noiseTex, "noiseTex");
	s_ssao.BindProgram();
	s_ssao.SetParam(SHADER_PARAM_gPositionTex, TEXTURE_GPOSITION);
	s_ssao.SetParam(SHADER_PARAM_gNormalTex, TEXTURE_GNORMAL);
	s_ssao.SetParam(SHADER_PARAM_noiseTex, TEXTURE_NOISE);

	s_ssaoBlur.RegisterParam(SHADER_PARAM_ssaoTex, "ssao");
	s_ssaoBlur.BindProgram();
	s_ssaoBlur.SetParam(SHADER_PARAM_ssaoTex, TEXTURE_SSAO);

	s_displayBuffer.RegisterParam(SHADER_PARAM_gPositionTex, "gPosition");
	s_displayBuffer.RegisterParam(SHADER_PARAM_gNormalTex, "gNormal");
	s_displayBuffer.RegisterParam(SHADER_PARAM_gAlbedoTex, "gAlbedo");
	s_displayBuffer.RegisterParam(SHADER_PARAM_ssaoTex, "ssao");
	s_displayBuffer.BindProgram();
	s_displayBuffer.SetParam(SHADER_PARAM_gPositionTex, TEXTURE_GPOSITION);
	s_displayBuffer.SetParam(SHADER_PARAM_gNormalTex, TEXTURE_GNORMAL);
	s_displayBuffer.SetParam(SHADER_PARAM_gAlbedoTex, TEXTURE_GALBEDO);
	s_displayBuffer.SetParam(SHADER_PARAM_ssaoTex, TEXTURE_SSAO);

	xTexScale = float(world_h - 2) / (world_w - 2);

	_world_w = world_w;
	_world_h = world_h;

	largestSide = MAX(world_h, world_w);

	l_lightDir = cyPoint3f(0.1f, 0.3f, 0.2f);
	l_lightDir.Normalize();

	m_waterMesh.genBuffers();
	m_terrainMesh.genBuffers();

	glGenTextures(2, t_distributions);
	createPoissonDiscDistribution(t_distributions[0], gNumBlockerSearchSamples);
	createPoissonDiscDistribution(t_distributions[1], gNumPCFSamples);

	m_unitQuad.initialize();

	m_fbSize = Vec2{ (float)frame_w, (float)frame_h };

	//////////////////////////////////////////////////////////////////////////
	std::vector<float> dataPtr(world_w * world_h * 4, 0);

	glGenTextures(1, &t_layerDataTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t_layerDataTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, world_w, world_h, 0, GL_RGBA, GL_FLOAT, &dataPtr[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_layer_data_resource, t_layerDataTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	glBindTexture(GL_TEXTURE_2D, 0);

	//////////////////////////////////////////////////////////////////////////
	glGenTextures(1, &t_depthDataTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t_depthDataTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, world_w, world_h, 0, GL_RGBA, GL_FLOAT, &dataPtr[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_depth_data_resource, t_depthDataTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	glBindTexture(GL_TEXTURE_2D, 0);

	//////////////////////////////////////////////////////////////////////////
	glGenTextures(1, &t_foamDataTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t_foamDataTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, world_w, world_h, 0, GL_RGBA, GL_FLOAT, &dataPtr[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_foam_data_resource, t_foamDataTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	glBindTexture(GL_TEXTURE_2D, 0);

	//////////////////////////////////////////////////////////////////////////
	std::vector<float> normFieldDataPtr(PB_RESOLUTION * 2 * 4, 0);

	glGenTextures(1, &t_pbNormFieldTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t_pbNormFieldTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, PB_RESOLUTION, 2, 0, GL_RGBA, GL_FLOAT, &normFieldDataPtr[0]);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_pb_norm_field_resource, t_pbNormFieldTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	glBindTexture(GL_TEXTURE_2D, 0);

	//////////////////////////////////////////////////////////////////////////
	std::vector<float> offsetFieldDataPtr(PB_RESOLUTION * 2 * 4, 0);

	glGenTextures(1, &t_pbOffsetFieldTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t_pbOffsetFieldTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, PB_RESOLUTION, 2, 0, GL_RGBA, GL_FLOAT, &offsetFieldDataPtr[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_pb_offset_field_resource, t_pbOffsetFieldTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	glBindTexture(GL_TEXTURE_2D, 0);

	//////////////////////////////////////////////////////////////////////////

	sh_shadow.Init(SHADOW_MAP_SIZE, l_lightDir, 4);
	sh_shadow.SetTarget(cyPoint3f(0, 0, 0));

	//////////////////////////////////////////////////////////////////////////
	// configure g-buffer framebuffer
	// ------------------------------
	glGenFramebuffers(1, &gBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

	// position color buffer
	glGenTextures(1, &gPosition);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, frame_w, frame_h, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
	// normal color buffer
	glGenTextures(1, &gNormal);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, frame_w, frame_h, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
	// color + specular color buffer
	glGenTextures(1, &gAlbedo);
	glBindTexture(GL_TEXTURE_2D, gAlbedo);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame_w, frame_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedo, 0);
	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);

	// create and attach depth buffer (renderbuffer)
	glGenRenderbuffers(1, &gRboDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, gRboDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, frame_w, frame_h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRboDepth);
	// finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//////////////////////////////////////////////////////////////////////////
	// configure topdown framebuffer
	// ------------------------------
	glGenFramebuffers(1, &gTopdown);
	glBindFramebuffer(GL_FRAMEBUFFER, gTopdown);

	glGenTextures(1, &gTopdownTex);
	glBindTexture(GL_TEXTURE_2D, gTopdownTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame_w, frame_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gTopdownTex, 0);

	// create and attach depth buffer (renderbuffer)
	glGenRenderbuffers(1, &gRboTopdownDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, gRboTopdownDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, frame_w, frame_h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gRboTopdownDepth);
	// finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//////////////////////////////////////////////////////////////////////////
	// also create framebuffer to hold SSAO processing stage 
	// -----------------------------------------------------

	glGenFramebuffers(1, &ssaoFBO);
	glGenFramebuffers(1, &ssaoBlurFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);

	// SSAO color buffer
	glGenTextures(1, &ssaoTex);
	glBindTexture(GL_TEXTURE_2D, ssaoTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, frame_w, frame_h, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoTex, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Framebuffer not complete!" << std::endl;

	// and blur stage
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
	glGenTextures(1, &ssaoBlurTex);
	glBindTexture(GL_TEXTURE_2D, ssaoBlurTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, frame_w, frame_h, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoBlurTex, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

#if 0
	// generate sample kernel
	// ----------------------
	std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0); // generates random floats between 0.0 and 1.0
	std::default_random_engine generator;
	std::vector<cyPoint3f> ssaoKernel;
	for (unsigned int i = 0; i < 64; ++i)
	{
		cyPoint3f sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
		sample.Normalize();
		sample *= randomFloats(generator);
		float scale = float(i) / 64.0f;

		// scale samples s.t. they're more aligned to center of kernel
		scale = ourLerp(0.1f, 1.0f, scale * scale);
		sample *= scale;
		ssaoKernel.push_back(sample);
		std::cout << sample.x << ", " << sample.y << ", " << sample.z << ", " << std::endl;
	}
#endif

	// generate noise texture
	// ----------------------
	std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
	std::default_random_engine generator;
	std::vector<cyPoint3f> ssaoNoise;
	for (unsigned int i = 0; i < 16; i++)
	{
		cyPoint3f noise(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, 0.0f); // rotate around z-axis (in tangent space)
		ssaoNoise.push_back(noise);
	}
	glGenTextures(1, &t_noiseTex);
	glBindTexture(GL_TEXTURE_2D, t_noiseTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	t_skyCubeTex = createCubemap();

	Mat4 topViewMat = Mat4::lookAt(Vec3(0.5f, 0.5f, 0.5f), Vec3(0.5f, 0.0f, 0.5f), Vec3(1, 0, 0));
	Mat4 topProjMat = Mat4::ortho(-0.5, 0.5, -0.5, 0.5);

	m_topCameraMat = topProjMat * topViewMat;

	m_topViewSize = Vec2(BUFFER_SIZE, BUFFER_SIZE);

	t_normNoiseTex = importTexture("../textures/water3NormalTexture.png");
	t_waterNoiseTex = importTexture("../textures/waterNoiseTexture.png");
}


void Visualizer::render(const Camera& camera, const float dt, const float time, bool draw_water, bool draw_sand)
{
#if 1
	// 1. geometry pass: render scene's geometry/color data into gbuffer
	// -----------------------------------------------------------------
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	s_geometryPass.BindProgram();
	s_geometryPass.SetParamMatrix4(SHADER_PARAM_cameraMatrix, &camera.GetMatrix()[0]);
	s_geometryPass.SetParamMatrix4(SHADER_PARAM_viewMatrix, &camera.GetViewMatrix()[0]);
	m_terrainMesh.render();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// 2. generate SSAO texture
	// ------------------------
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_GPOSITION);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_GNORMAL);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_NOISE);
	glBindTexture(GL_TEXTURE_2D, t_noiseTex);
	s_ssao.BindProgram();
	s_ssao.SetParamMatrix4(SHADER_PARAM_projectionMatrix, &camera.GetProjMatrix()[0]);
	glDisable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	m_unitQuad.render();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// 3. blur SSAO texture to remove noise
	// ------------------------------------
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	s_ssaoBlur.BindProgram();
	glActiveTexture(GL_TEXTURE0 + TEXTURE_SSAO);
	glBindTexture(GL_TEXTURE_2D, ssaoTex);
	m_unitQuad.render();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif

#if 1
	// 4. shadow pass
	// ------------------------------------
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glActiveTexture(GL_TEXTURE0 + TEXTURE_SHADOW);
	sh_shadow.BindTexture();
	sh_shadow.BeginRenderShadow();
	s_shadowPass.BindProgram();
	s_shadowPass.SetParamMatrix4(SHADER_PARAM_cameraMatrix, &sh_shadow.GetMatrix()[0]);
	m_terrainMesh.render();
	sh_shadow.EndRenderShadow();

	glViewport(0, 0, 1920, 1080);
#endif

#if 1
	// 5. top down view
	// ------------------------------------
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gTopdown);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glViewport(0, 0, GLsizei(mTopViewSize.x), GLsizei(mTopViewSize.y));
	cyMatrix4f shadowMatrix = sh_shadow.GetLookupMatrix() * m_topCameraMat.GetInverse();

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glActiveTexture(GL_TEXTURE0 + TEXTURE_DIST0);
	glBindTexture(GL_TEXTURE_1D, t_distributions[0]);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_DIST1);
	glBindTexture(GL_TEXTURE_1D, t_distributions[1]);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_SSAO);
	glBindTexture(GL_TEXTURE_2D, ssaoBlurTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_LAYERDATA);
	glBindTexture(GL_TEXTURE_2D, t_layerDataTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_DEPTHDATA);
	glBindTexture(GL_TEXTURE_2D, t_depthDataTex);

	s_simpleground.BindProgram();
	s_simpleground.SetParamMatrix4(SHADER_PARAM_cameraMatrix, &m_topCameraMat[0]);
	s_simpleground.SetParam(SHADER_PARAM_lightDir, l_lightDir.x, l_lightDir.y, l_lightDir.z);
	s_simpleground.SetParamMatrix4(SHADER_PARAM_shadowMatrix, &sh_shadow.GetMatrix()[0]);
	s_simpleground.SetParam(SHADER_PARAM_xTexScale, xTexScale);
	s_simpleground.SetParam(SHADER_PARAM_largestSide, largestSide);
	m_terrainMesh.render();

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
#endif

#if 0
	// testing render buffer to screen
	// -----------------------------------------------------------------
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_GPOSITION);
	glBindTexture(GL_TEXTURE_2D, t_layerDataTex);
	//glBindTexture(GL_TEXTURE_2D, gTopdownTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_GNORMAL);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_GALBEDO);
	glBindTexture(GL_TEXTURE_2D, gAlbedo);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_SSAO);
	//glBindTexture(GL_TEXTURE_2D, t_layerDataTex);
	glBindTexture(GL_TEXTURE_2D, ssaoBlurTex);

	s_displayBuffer.BindProgram();
	glDisable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	m_unitQuad.render();
#endif


#if 1
	glViewport(0, 0, GLsizei(m_fbSize.x), GLsizei(m_fbSize.y));

	shadowMatrix = sh_shadow.GetLookupMatrix() * camera.GetMatrixInverse();

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glActiveTexture(GL_TEXTURE0 + TEXTURE_DIST0);
	glBindTexture(GL_TEXTURE_1D, t_distributions[0]);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_DIST1);
	glBindTexture(GL_TEXTURE_1D, t_distributions[1]);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_SSAO);
	glBindTexture(GL_TEXTURE_2D, ssaoBlurTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_LAYERDATA);
	glBindTexture(GL_TEXTURE_2D, t_layerDataTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_DEPTHDATA);
	glBindTexture(GL_TEXTURE_2D, t_depthDataTex);
	glActiveTexture(GL_TEXTURE0 + TEXTURE_FOAMDATA);
	glBindTexture(GL_TEXTURE_2D, t_foamDataTex);

	s_ground.BindProgram();
	s_ground.SetParamMatrix4(SHADER_PARAM_cameraMatrix, &camera.GetMatrix()[0]);
	s_ground.SetParam(SHADER_PARAM_lightDir, l_lightDir.x, l_lightDir.y, l_lightDir.z);
	s_ground.SetParamMatrix4(SHADER_PARAM_shadowMatrix, &sh_shadow.GetMatrix()[0]);
	s_ground.SetParam(SHADER_PARAM_xTexScale, xTexScale);
	s_ground.SetParam(SHADER_PARAM_largestSide, largestSide);
	m_terrainMesh.render();

	if (draw_water)
	{
		glActiveTexture(GL_TEXTURE0 + TEXTURE_SKYCUBE);
		glBindTexture(GL_TEXTURE_CUBE_MAP, t_skyCubeTex);
		glActiveTexture(GL_TEXTURE0 + TEXTURE_TOPDOWN);
		glBindTexture(GL_TEXTURE_2D, gTopdownTex);
		glActiveTexture(GL_TEXTURE0 + TEXTURE_NORMNOISE);
		glBindTexture(GL_TEXTURE_2D, t_normNoiseTex);
		glActiveTexture(GL_TEXTURE0 + TEXTURE_WATERNOISE);
		glBindTexture(GL_TEXTURE_2D, t_waterNoiseTex);
		glActiveTexture(GL_TEXTURE0 + TEXTURE_PBNORMFIELD);
		glBindTexture(GL_TEXTURE_2D, t_pbNormFieldTex);
		glActiveTexture(GL_TEXTURE0 + TEXTURE_PBOFFSETFIELD);
		glBindTexture(GL_TEXTURE_2D, t_pbOffsetFieldTex);
		s_water.BindProgram();
		s_water.SetParamMatrix4(SHADER_PARAM_cameraMatrix, &camera.GetMatrix()[0]);
		s_water.SetParam(SHADER_PARAM_lightDir, l_lightDir.x, l_lightDir.y, l_lightDir.z);
		s_water.SetParam(SHADER_PARAM_time, time);
		s_water.SetParam(SHADER_PARAM_xTexScale, xTexScale);
		s_water.SetParam(SHADER_PARAM_largestSide, largestSide);
		s_water.SetParam(SHADER_PARAM_cameraDistance, this->cameraDistance);
		s_water.SetParam(SHADER_PARAM_ampla, this->ampla);
		s_water.SetParam(SHADER_PARAM_L, this->L);

		s_water.SetParam(SHADER_PARAM_TAU, TAU);
		s_water.SetParam(SHADER_PARAM_DOMAIN_SCALE, DOMAIN_SCALE);
		s_water.SetParam(SHADER_PARAM_SEG_PER_DIR, SEG_PER_DIR);
		s_water.SetParam(SHADER_PARAM_FINE_DIR_NUM, FINE_DIR_NUM);
		m_waterMesh.render();
	}
#endif
}

void Visualizer::UpdateCameraDistance(const float& cameraDistance)
{
	this->cameraDistance = cameraDistance;
}
