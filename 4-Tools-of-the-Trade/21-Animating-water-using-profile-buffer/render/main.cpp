#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <ctime>
#include <string>

#if defined(_WIN32)
#include <GL/glew.h>
#include <GL/freeglut.h>
#else
#include <GLut/glut.h>
#endif

#include "scene/RiverScene.h"
#include "scene/TestODScene.h"
#include "scene/TestFDScene.h"
#include "scene/TestSwirlScene.h"
#include "scene/TestOutwardScene.h"

#include "common/common.h"
#include "common/Timer.h"
#include "common/CImg.h"

#include "visualizer.h"
#include "mesh.h"
#include "camera.h"

#include <cu/CudaDevice.h>

#include "cu/helper_cuda.h"

using namespace cimg_library;

int   frame = 0;

Scene* gScene = nullptr;

bool g_simulation = false;
bool g_draw_force = false;
bool g_draw_water = true;
bool g_draw_sand = true;

int gGridRes[2];

real gScale = (real)1.f / ((real)DX * ((real)GRID_W + (real)2.f));

enum class MouseMode {
	MOUSEMODE_NONE,
	MOUSEMODE_ROTATE,
	MOUSEMODE_MOVE,
	MOUSEMODE_ZOOM,
};

Vec2 gMousePos;
MouseMode gMouseMode = MouseMode::MOUSEMODE_NONE;
Camera gCamera;

Visualizer gVisualizer;

Timer<real> timer_global;

float gDx;

bool gWireframe = false;

float gLastTime = 0;

//-------------------------------------------------------------------------------

void render(const float dt, const Camera& camera) {

	float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f * 0.2f;

	if (gWireframe) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	gVisualizer.render(camera, dt, time, g_draw_water, g_draw_sand);
}

void GlutDisplay()
{
	auto time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	auto dt = time - gLastTime;
	render(dt, gCamera);

#ifdef EXPORT_IMG
	if (frame <= -100)
	{
		unsigned char* buffer = new unsigned char[WINDOW_WIDTH * WINDOW_HEIGHT * 3];
		glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, buffer);

		CImg<unsigned char> image(WINDOW_WIDTH, WINDOW_HEIGHT, 1, 3, 0);
#pragma omp parallel for
		for (int j = 0; j < WINDOW_HEIGHT; j++)
		{
			for (int i = 0; i < WINDOW_WIDTH; i++)
			{
				image(i, WINDOW_HEIGHT - 1 - j, 0) = buffer[3 * (j * WINDOW_WIDTH + i)];
				image(i, WINDOW_HEIGHT - 1 - j, 1) = buffer[3 * (j * WINDOW_WIDTH + i) + 1];
				image(i, WINDOW_HEIGHT - 1 - j, 2) = buffer[3 * (j * WINDOW_WIDTH + i) + 2];
			}
		}
		char name[100];
		snprintf(name, 100, "./frame_%04d.bmp", frame);
		std::string path = std::string(name);
		image.save(path.c_str());
		delete[] buffer;
		if (frame == EXPORT_STEPS)
		{
			exit(0);
		}
	}
#endif

	glutSwapBuffers();
}

void GlutReshape(int w, int h)
{
	glViewport(0, 0, w, h);

	gCamera.SetAspect((float)w / float(h));
}

//-------------------------------------------------------------------------------

void GlutMouse(int button, int state, int x, int y)
{
	if (state == GLUT_UP) {
		gMouseMode = MouseMode::MOUSEMODE_NONE;
	}
	else {
		switch (button) {
		case GLUT_LEFT_BUTTON:
			gMouseMode = MouseMode::MOUSEMODE_ROTATE;
			break;
		case GLUT_MIDDLE_BUTTON:
			gMouseMode = MouseMode::MOUSEMODE_MOVE;
			break;
		case GLUT_RIGHT_BUTTON:
			gMouseMode = MouseMode::MOUSEMODE_ZOOM;
			break;
		}
	}
	gMousePos = Vec2{ float(x), float(y) };
}

void GlutMotion(int x, int y)
{
	Vec2 mouseDif = Vec2((float)x, (float)y) - gMousePos;
	switch (gMouseMode) {
	case MouseMode::MOUSEMODE_ROTATE:
		gCamera.AddRotation(mouseDif.x * 0.01f, mouseDif.y * 0.01f);
		glutPostRedisplay();
		break;
	case MouseMode::MOUSEMODE_MOVE:
		gCamera.AddOffset(-mouseDif.x * 0.01f, -mouseDif.y * 0.01f);
		glutPostRedisplay();
		break;
	case MouseMode::MOUSEMODE_ZOOM:
		gCamera.AddDistance(mouseDif.y * 0.1f);
		glutPostRedisplay();
		break;
	}
	gMousePos = Vec2{ float(x), float(y) };

#ifdef SHOW_CAMERA_INFO
	printf("gCamera.SetDistance(%ff); gCamera.SetRotation(%ff, %ff); gCamera.SetOffset(%ff, %ff);\n",
		gCamera.GetDistance(), gCamera.rot[0], gCamera.rot[1], gCamera.Get_Offset()[0], gCamera.Get_Offset()[1]);
	gVisualizer.UpdateCameraDistance(gCamera.GetDistance());
#endif

}


//-------------------------------------------------------------------------------

void CleanUp()
{
	delete gScene;

	CudaDevice::shutdown();
}

//-------------------------------------------------------------------------------

float height(int i, int j)
{
	const real* solidHeight = gScene->GetTerrainHeight();
	int id = j * gGridRes[0] + i;
	float solid_height = solidHeight[id] * gScale;
	if (i == 0 || j == 0 || i == (GRID_W + 1) || j == (GRID_L + 1))
		solid_height = (float)1.e-3;
	return solid_height;
}

void resetVisualizer()
{
	gScene->TransferTerrainHeightToCPU();

	gScene->GetGridRes(gGridRes);
	gDx = 1.0f / gGridRes[0];

	float d = 1.0f / gGridRes[1];

	std::vector<Vertex> verticesTmp(gGridRes[0] * gGridRes[1]);

#pragma omp parallel for
	for (int idx1D = 0; idx1D < gGridRes[0] * gGridRes[1]; idx1D++)
	{
		int index[2] = { 0, 0 };

		index[1] = idx1D / gGridRes[0];
		index[0] = idx1D - index[1] * gGridRes[0];

		Vec3 pos = Vec3{ index[0] * d,     height(index[0], index[1]),     index[1] * d };

		Vec2 texCoord = Vec2(float(index[0]), float(index[1])) / 50.0f;

		verticesTmp[idx1D] = (Vertex{ pos, Vec3(0,0,0), texCoord });
	}

	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;
	unsigned verticesPerRow = gGridRes[0];
	int count = 0;
	for0(x, gGridRes[0] - 1)
	{
		for0(y, gGridRes[1] - 1)
		{
			Vertex& p0 = verticesTmp[(x + 1) + (y + 1) * verticesPerRow];
			Vertex& p1 = verticesTmp[x + (y + 1) * verticesPerRow];
			Vertex& p2 = verticesTmp[x + y * verticesPerRow];
			Vertex& p3 = verticesTmp[(x + 1) + y * verticesPerRow];

			Vec3 normal = (p0.position - p2.position).Cross(p0.position - p1.position);
			normal.Normalize();

			vertices.push_back(Vertex{ p2.position, normal, p2.texCoord });
			vertices.push_back(Vertex{ p1.position, normal, p1.texCoord });
			vertices.push_back(Vertex{ p0.position, normal, p0.texCoord });
			indices.push_back(count++);
			indices.push_back(count++);
			indices.push_back(count++);

			normal = (p3.position - p2.position).Cross(p3.position - p0.position);
			normal.Normalize();

			vertices.push_back(Vertex{ p2.position, normal, p2.texCoord });
			vertices.push_back(Vertex{ p0.position, normal, p0.texCoord });
			vertices.push_back(Vertex{ p3.position, normal, p3.texCoord });
			indices.push_back(count++);
			indices.push_back(count++);
			indices.push_back(count++);
		}
	}
	auto waterLevelFunction = [](Vec2 coordinate) {
		int i = (int)coordinate.x;
		int j = (int)coordinate.y;
		if (i == 0 || j == 0 || i == (GRID_W + 1) || j == (GRID_L + 1))
			return (float)0.;
		const real* waterHeight = gScene->GetHeight();
		float height = waterHeight[j * gGridRes[0] + i] * gScale;
		const real* solidHeight = gScene->GetTerrainHeight();
		float solid_height = solidHeight[j * gGridRes[0] + i] * gScale;
		return solid_height + height;
		};

	gVisualizer.m_terrainMesh.initialize(gGridRes[0] - 1, gGridRes[1] - 1, gDx, indices, vertices);
	gVisualizer.m_waterMesh.initialize(gGridRes[0] - 1, gGridRes[1] - 1, gDx, waterLevelFunction, false);
}

//-------------------------------------------------------------------------------

void GlutKeyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case ' ':
		g_simulation = !g_simulation;
		break;
	case 27:                // ESC
		CleanUp();
		exit(0);
		break;
	case 'r':
		break;
	case 'w':
		g_draw_water = !g_draw_water;
		break;
	}
}

//-------------------------------------------------------------------------------

void GlutIdle()
{
	if (g_simulation)
	{
		gVisualizer.UpdateCameraDistance(gCamera.GetDistance());

		//////////////////////////////////////////////////////////////////////////
		gScene->ProfileBufferStep();

		//////////////////////////////////////////////////////////////////////////
		gScene->PrepareFbNormFieldResource();
		real* CudaPbNormField = gScene->GetCudaPbNormField();
		struct cudaGraphicsResource* cuda_pb_norm_field_resource = gVisualizer.getCudaPbNormFieldResource();;
		cudaArray* pb_norm_field_ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pb_norm_field_resource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pb_norm_field_ptr, cuda_pb_norm_field_resource, 0, 0));
		checkCudaErrors(cudaMemcpyToArray(pb_norm_field_ptr, 0, 0, CudaPbNormField, sizeof(real) * 4 * 2 * PB_RESOLUTION, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pb_norm_field_resource));

		//////////////////////////////////////////////////////////////////////////
		gScene->PrepareFbOffsetFieldResource();
		real* CudaPbOffsetField = gScene->GetCudaPbOffsetField();
		struct cudaGraphicsResource* cuda_pb_offset_field_resource = gVisualizer.getCudaPbOffsetFieldResource();;
		cudaArray* pb_offset_field_ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pb_offset_field_resource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pb_offset_field_ptr, cuda_pb_offset_field_resource, 0, 0));
		checkCudaErrors(cudaMemcpyToArray(pb_offset_field_ptr, 0, 0, CudaPbOffsetField, sizeof(real) * 4 * 2 * PB_RESOLUTION, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pb_offset_field_resource));

		frame++;
	}

	glutPostRedisplay();
}


void Update()
{
	glutPostRedisplay();
}

//-------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << CyanHead()
			<< "Usage: ./CuSSWE [Test]"
			<< CyanTail() << std::endl;
		return 0;
	}

	srand(0);

	const std::string name(argv[1]);

	int res[2];
	res[0] = GRID_W;				// z axis
	res[1] = GRID_L;				// x axis

	// Initialize scene based on command line arguments
	
	if (name == "River")
	{
		gScene = new RiverScene();
	}
	else if (name == "FixedDirection")
	{
		gScene = new TestFDScene();
	}
	else if (name == "OppositeDirection")
	{
		gScene = new TestODScene();
	}
	else if (name == "Outward")
	{
		gScene = new TestOutwardScene();
	}
	else if (name == "Swirl")
	{
		gScene = new TestSwirlScene();
	}
	else
	{
		std::cout << RedHead() << "Unsupported scene: " << name << RedTail() << std::endl;
		return 0;
	}

	std::cout << "Running test: " << name << std::endl;

	const real dx = ToReal(DX);
	real minc[2] = { 0, 0 };
	real domain[2] = { 0, 0 };
	domain[0] = res[0] * dx;
	domain[1] = res[1] * dx;

	std::cout << "RES: " << res[0] << " " << res[1] << std::endl;

	gScene->Configure(minc, domain, res);

	CudaDevice::startup();
	CudaDevice::reportMemory("before GPU memory allocation");

	gScene->Initialize();

	CudaDevice::reportMemory("after GPU memory allocation");

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	if (glutGet(GLUT_SCREEN_WIDTH) > 0 && glutGet(GLUT_SCREEN_HEIGHT) > 0)
	{
		glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - WINDOW_WIDTH) / 2, (glutGet(GLUT_SCREEN_HEIGHT) - WINDOW_HEIGHT) / 2);
	}
	else { glutInitWindowPosition(50, 50); }
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

	std::string title = "ProfileBuffer";
	glutCreateWindow(title.c_str());

	glutIdleFunc(GlutIdle);
	glutDisplayFunc(GlutDisplay);
	glutReshapeFunc(GlutReshape);
	glutKeyboardFunc(GlutKeyboard);
	glutMouseFunc(GlutMouse);
	glutMotionFunc(GlutMotion);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	glGetError();

	gCamera.SetAspect((float)WINDOW_WIDTH / WINDOW_HEIGHT);

	// top view
	gCamera.SetDistance(1.500000f); gCamera.SetRotation(-1.5f * (float)M_PI, 0.5f * (float)M_PI); gCamera.SetOffset(0.009988f, -0.120000f);

	// unnatural shaking
	gCamera.SetDistance(0.100000f); gCamera.SetRotation(8.447616f, 0.930798f); gCamera.SetOffset(0.109988f, -0.140000f);

	// sharp boundary
	gCamera.SetDistance(0.100000f); gCamera.SetRotation(6.727598f, 1.380798f); gCamera.SetOffset(-0.060012f, 0.100000f);

	gCamera.SetDistance(0.200000f); gCamera.SetRotation(6.307604f, 1.450799f); gCamera.SetOffset(0.029988f, 0.090000f);

	gCamera.SetDistance(0.300000f); gCamera.SetRotation(6.307604f, 1.450799f); gCamera.SetOffset(0.029988f, 0.090000f);
	gCamera.SetDistance(0.500000f); gCamera.SetRotation(5.347605f, 0.640799f); gCamera.SetOffset(0.029988f, 0.090000f);
	gCamera.SetDistance(0.500000f); gCamera.SetRotation(5.217605f, 0.330799f); gCamera.SetOffset(0.029988f, 0.090000f);

	gCamera.SetDistance(0.500001f); gCamera.SetRotation(6.267602f, 1.680798f); gCamera.SetOffset(0.189988f, 0.230000f);
	gCamera.SetDistance(0.500001f); gCamera.SetRotation(9.137602f, 0.320798f); gCamera.SetOffset(0.189988f, 0.230000f);


	gCamera.SetDistance(1.200001f); gCamera.SetRotation(7.447595f, 0.510798f); gCamera.SetOffset(-0.120012f, -0.040000f);


	// Swirl
	gCamera.SetDistance(0.500001f); gCamera.SetRotation(11.307605f, 0.490798f); gCamera.SetOffset(-0.120012f, -0.040000f);

	// River front
	gCamera.SetDistance(0.800001f); gCamera.SetRotation(11.257607f, 0.340798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	// River back
	gCamera.SetDistance(0.600001f); gCamera.SetRotation(14.817605f, 0.480798f); gCamera.SetOffset(-0.120012f, -0.040000f);

	// Outward
	gCamera.SetDistance(0.600001f); gCamera.SetRotation(17.517603f, 0.680798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	gCamera.SetDistance(0.600001f); gCamera.SetRotation(17.557602f, 0.870798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	gCamera.SetDistance(0.700001f); gCamera.SetRotation(18.787592f, 0.450799f); gCamera.SetOffset(0.019988f, -0.160000f);
	gCamera.SetDistance(0.700001f); gCamera.SetRotation(19.077587f, 0.350799f); gCamera.SetOffset(0.019988f, -0.160000f);

	gCamera.SetDistance(0.500001f); gCamera.SetRotation(18.257572f, 0.350799f); gCamera.SetOffset(0.019988f, -0.160000f);

	// River front
	gCamera.SetDistance(0.500001f); gCamera.SetRotation(17.947569f, 0.310799f); gCamera.SetOffset(0.019988f, -0.160000f);
	// River back
	//gCamera.SetDistance(0.500000f); gCamera.SetRotation(21.037592f, 0.290799f); gCamera.SetOffset(-0.040012f, -0.020000f);


	// FixedDirection
	gCamera.SetDistance(0.800001f); gCamera.SetRotation(10.497598f, 0.270798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	// Swirl
	gCamera.SetDistance(0.500001f); gCamera.SetRotation(11.307605f, 0.490798f); gCamera.SetOffset(-0.120012f, -0.040000f);


	gCamera.SetDistance(0.500001f); gCamera.SetRotation(18.257572f, 0.350799f); gCamera.SetOffset(0.019988f, -0.160000f);
	gCamera.SetDistance(1.500000f); gCamera.SetRotation(-1.5f * (float)M_PI, 0.5f * (float)M_PI); gCamera.SetOffset(0.009988f, -0.120000f);


	gCamera.SetDistance(2.600001f); gCamera.SetRotation(12.537597f, 1.170798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	gCamera.SetDistance(0.800001f); gCamera.SetRotation(10.497598f, 0.270798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	
	
	// Fixed Limits, Fixed Direction
	gCamera.SetDistance(0.600001f); gCamera.SetRotation(10.517600f, 0.400798f); gCamera.SetOffset(-0.120012f, -0.040000f);
	
	

	gCamera.SetDistance(0.500001f); gCamera.SetRotation(9.407600f, 0.380798f); gCamera.SetOffset(-0.120012f, -0.040000f);

	gCamera.SetDistance(0.500001f); gCamera.SetRotation(11.307605f, 0.490798f); gCamera.SetOffset(-0.120012f, -0.040000f);

	gCamera.SetDistance(0.500000f); gCamera.SetRotation(13.297605f, 0.370798f); gCamera.SetOffset(0.169988f, -0.150000f);

	gCamera.SetDistance(0.500001f); gCamera.SetRotation(11.307605f, 0.490798f); gCamera.SetOffset(-0.120012f, -0.040000f);


	gVisualizer.initialize(WINDOW_WIDTH, WINDOW_HEIGHT, gScene->GetGrid().res_v[0], gScene->GetGrid().res_v[1], gCamera.GetDistance(), gScene->GetAmpla(), gScene->GetPeriodicity());
	resetVisualizer();
	size_t num_bytes;
	real* water_pos_data;
	real* water_norm_data;

	struct cudaGraphicsResource* cudaWaterPosResource = gVisualizer.m_waterMesh.getCudaPos();
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaWaterPosResource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&water_pos_data, &num_bytes,
		cudaWaterPosResource));

	struct cudaGraphicsResource* cudaWaterNormResource = gVisualizer.m_waterMesh.getCudaNorm();
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaWaterNormResource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&water_norm_data, &num_bytes,
		cudaWaterNormResource));

	gScene->PrepareGridWorldPos(water_pos_data, water_norm_data);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaWaterPosResource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaWaterNormResource, 0));

	//////////////////////////////////////////////////////////////////////////

	real* CudaLayerData = gScene->GetCudaLayeredData();

	struct cudaGraphicsResource* cuda_layer_data_resource = gVisualizer.getCudaLayerDataResource();;
	cudaArray* layer_data_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_layer_data_resource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&layer_data_ptr, cuda_layer_data_resource, 0, 0));

	int num_texels = gScene->GetGrid().total_voxels;
	checkCudaErrors(cudaMemcpyToArray(layer_data_ptr, 0, 0, CudaLayerData, sizeof(real) * 4 * num_texels, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_layer_data_resource));

	//////////////////////////////////////////////////////////////////////////

	real* CudaDepthData = gScene->GetCudaDepthData();

	struct cudaGraphicsResource* cuda_depth_data_resource = gVisualizer.getCudaDepthDataResource();;
	cudaArray* depth_data_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_depth_data_resource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&depth_data_ptr, cuda_depth_data_resource, 0, 0));

	//int num_texels = gScene->GetGrid().total_voxels;
	checkCudaErrors(cudaMemcpyToArray(depth_data_ptr, 0, 0, CudaDepthData, sizeof(real) * 4 * num_texels, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_depth_data_resource));

	//////////////////////////////////////////////////////////////////////////

	real* CudaFoamData = gScene->GetCudaFoamData();

	struct cudaGraphicsResource* cuda_foam_data_resource = gVisualizer.getCudaFoamDataResource();;
	cudaArray* foam_data_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_foam_data_resource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&foam_data_ptr, cuda_foam_data_resource, 0, 0));

	//int num_texels = gScene->GetGrid().total_voxels;
	checkCudaErrors(cudaMemcpyToArray(foam_data_ptr, 0, 0, CudaFoamData, sizeof(real) * 4 * num_texels, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_foam_data_resource));

	//////////////////////////////////////////////////////////////////////////

	gScene->ProfileBufferStep();

	//////////////////////////////////////////////////////////////////////////

	gScene->PrepareFbNormFieldResource();

	real* CudaPbNormField = gScene->GetCudaPbNormField();

	struct cudaGraphicsResource* cuda_pb_norm_field_resource = gVisualizer.getCudaPbNormFieldResource();;
	cudaArray* pb_norm_field_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pb_norm_field_resource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pb_norm_field_ptr, cuda_pb_norm_field_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(pb_norm_field_ptr, 0, 0, CudaPbNormField, sizeof(real) * 4 * 2 * PB_RESOLUTION, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pb_norm_field_resource));

	//////////////////////////////////////////////////////////////////////////

	gScene->PrepareFbOffsetFieldResource();

	real* CudaPbOffsetField = gScene->GetCudaPbOffsetField();

	struct cudaGraphicsResource* cuda_pb_offset_field_resource = gVisualizer.getCudaPbOffsetFieldResource();;
	cudaArray* pb_offset_field_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pb_offset_field_resource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&pb_offset_field_ptr, cuda_pb_offset_field_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(pb_offset_field_ptr, 0, 0, CudaPbOffsetField, sizeof(real) * 4 * 2 * PB_RESOLUTION, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pb_offset_field_resource));

	//////////////////////////////////////////////////////////////////////////

	glutMainLoop();

	return 0;
}