
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "camera.h"
#include "mesh.h"
#include "mathUtils.h"

#include "shadow.h"
#include "cyGLSL.h"

#define BUFFER_SIZE   1024

#define SHADOW_MAP_SIZE	4096

enum {
	TEXTURE_SHADOW,
	TEXTURE_SANDNORM,
	TEXTURE_DIST0,
	TEXTURE_DIST1,
	TEXTURE_GPOSITION,
	TEXTURE_GNORMAL,
	TEXTURE_GALBEDO,
	TEXTURE_NOISE,
	TEXTURE_SSAO,
	TEXTURE_SKYCUBE,
	TEXTURE_LAYERDATA,
	TEXTURE_DEPTHDATA,
	TEXTURE_FOAMDATA,
	TEXTURE_TOPDOWN,
	TEXTURE_NORMNOISE,
	TEXTURE_WATERNOISE,
	TEXTURE_PBNORMFIELD,
	TEXTURE_PBOFFSETFIELD
};

enum {
	SHADER_PARAM_cameraMatrix,
	SHADER_PARAM_viewMatrix,
	SHADER_PARAM_projectionMatrix,
	SHADER_PARAM_xTexScale,
	SHADER_PARAM_largestSide,
	SHADER_PARAM_time,
	SHADER_PARAM_lightDir,
	SHADER_PARAM_shadowMatrix,
	SHADER_PARAM_shadowTex,
	SHADER_PARAM_sandNormTex,
	SHADER_PARAM_dist0Tex,
	SHADER_PARAM_dist1Tex,
	SHADER_PARAM_gPositionTex,
	SHADER_PARAM_gNormalTex,
	SHADER_PARAM_gAlbedoTex,
	SHADER_PARAM_noiseTex,
	SHADER_PARAM_ssaoTex,
	SHADER_PARAM_skyCubeTex,
	SHADER_PARAM_layerDatalTex,
	SHADER_PARAM_depthDatalTex,
	SHADER_PARAM_topdownTex,
	SHADER_PARAM_normNoiseTex,
	SHADER_PARAM_waterNoiseTex,
	SHADER_PARAM_pbNormFieldTex,
	SHADER_PARAM_pbOffsetFieldTex,
	SHADER_PARAM_cameraDistance,
	SHADER_PARAM_foamDatalTex,
	SHADER_PARAM_ampla,
	SHADER_PARAM_L,


	SHADER_PARAM_TAU,
	SHADER_PARAM_DOMAIN_SCALE,
	SHADER_PARAM_SEG_PER_DIR,
	SHADER_PARAM_FINE_DIR_NUM
};


struct FramebufferData
{
	GLuint colorTex = -1;
	GLuint depthTex = -1;
	GLuint id = -1;
};

enum class RenderMode
{
	Normal,
	Background,
	Water,
	WaterMap,
	TopView
};

struct Geometry
{
	Geometry() : arrayBuffer(-1), vertexArrayObject(-1) {}

	GLuint arrayBuffer = -1;
	GLuint vertexArrayObject = -1;

	void initialize();
	void render();
};

class Visualizer
{
public:
	Visualizer() {}
	~Visualizer() {}

	void initialize(const int windowWidth, const int windowHeight, const int world_w, const int world_h, const float cameraDistance, const float ampla, const float periodcity);

	void render(const Camera& camera, const float dt, const float time, bool draw_water, bool draw_sand);

	cudaGraphicsResource* getCudaLayerDataResource() {
		glActiveTexture(TEXTURE_LAYERDATA);;
		glBindTexture(GL_TEXTURE_2D, t_layerDataTex);;
		return cuda_layer_data_resource;
	}

	cudaGraphicsResource* getCudaDepthDataResource() {
		glActiveTexture(TEXTURE_DEPTHDATA);;
		glBindTexture(GL_TEXTURE_2D, t_depthDataTex);;
		return cuda_depth_data_resource;
	}

	cudaGraphicsResource* getCudaFoamDataResource() {
		glActiveTexture(TEXTURE_DEPTHDATA);;
		glBindTexture(GL_TEXTURE_2D, t_foamDataTex);;
		return cuda_foam_data_resource;
	}

	cudaGraphicsResource* getCudaPbNormFieldResource() {
		glActiveTexture(TEXTURE_PBNORMFIELD);;
		glBindTexture(GL_TEXTURE_2D, t_pbNormFieldTex);;
		return cuda_pb_norm_field_resource;
	}

	cudaGraphicsResource* getCudaPbOffsetFieldResource() {
		glActiveTexture(TEXTURE_PBOFFSETFIELD);;
		glBindTexture(GL_TEXTURE_2D, t_pbOffsetFieldTex);;
		return cuda_pb_offset_field_resource;
	}


	float xTexScale;

	Mesh		m_terrainMesh;
	Mesh		m_waterMesh;
	Geometry	m_unitQuad;

	DirShadow	sh_shadow;

	cyGLSLProgram s_simpleground;
	cyGLSLProgram s_ground;
	cyGLSLProgram s_water;
	cyGLSLProgram s_shadowPass;
	cyGLSLProgram s_geometryPass;
	cyGLSLProgram s_displayBuffer;
	cyGLSLProgram s_ssao;
	cyGLSLProgram s_ssaoBlur;

	cyPoint3f l_lightDir;

	int _world_w, _world_h;

	int largestSide;

	float cameraDistance;
	float ampla;
	float L;
	void UpdateCameraDistance(const float& cameraDistance);

private:

	GLuint					t_distributions[2] = { 0, 0 };
	GLuint					t_noiseTex = -1;
	GLuint					t_normNoiseTex = -1;
	GLuint					t_skyCubeTex = -1;
	GLuint					t_layerDataTex = -1;
	GLuint					t_depthDataTex = -1;
	GLuint					t_foamDataTex = -1;
	GLuint					t_waterNoiseTex = -1;
	GLuint					t_pbNormFieldTex = -1;
	GLuint					t_pbOffsetFieldTex = -1;

	Vec2					m_fbSize;
	Vec2					m_topViewSize;
	Mat4					m_topCameraMat;

	// gbuffer
	GLuint gBuffer = -1;
	GLuint gPosition = -1;
	GLuint gNormal = -1;
	GLuint gAlbedo = -1;
	GLuint gRboDepth = -1;

	// gTopdown
	GLuint gTopdown = -1;
	GLuint gTopdownTex = -1;
	GLuint gRboTopdownDepth = -1;

	// ssao
	GLuint ssaoFBO = -1;
	GLuint ssaoBlurFBO = -1;
	GLuint ssaoTex = -1;
	GLuint ssaoBlurTex = -1;

	const int gNumBlockerSearchSamples = 64;
	const int gNumPCFSamples = 64;

	cudaGraphicsResource* cuda_layer_data_resource;
	cudaGraphicsResource* cuda_depth_data_resource;
	cudaGraphicsResource* cuda_foam_data_resource;
	cudaGraphicsResource* cuda_pb_norm_field_resource;
	cudaGraphicsResource* cuda_pb_offset_field_resource;
};