#include "scene/TestODScene.h"

#include <cuda_runtime.h>
#include "cu/helper_cuda.h"
#include "cu/cuKernelWraper.cuh"

#if defined(_WIN32)
#include <GL/freeglut.h>
#else
#include <GLut/glut.h>
#endif

void TestODScene::Initialize()
{
	s_fast = 10.f;
	s_slow = 10.f;
	k_min = .1f;
	k_max = 50.f;
	ampla = .5f;
	L = 20;
	integration_nodes = 100;
	//*************************************************************************************

	Initialize_Fields();
	PrecomputeGroupSpeeds();

	t = 100.0;
	dt = grid_.dx[0] / gv * pow(10.f, -0.9f);
}

void TestODScene::Initialize_Water_Depth_Field()
{
#ifdef TRACK_WHERE
	std::cout << "TestODScene::Initialize_Water_Depth_Field" << std::endl;
#endif
	for (int i = 1; i <= GRID_W; i++)
	{
		for (int j = 1; j <= GRID_L; j++)
		{
			real hw = (556.f / 4000.f * DOMAIN_SCALE) - h_H_field_[i * grid_.res_v[0] + j];
			h_hw_field_[i * grid_.res_v[1] + j] = hw;
		}
	}
	checkCudaErrors(cudaMemcpy(d_hw_field_, h_hw_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}

void TestODScene::Initialize_Solid_Field()
{
	for (int j = 1; j <= GRID_L; ++j)
	{
		for (int i = 1; i <= GRID_W; ++i)
		{
			int id = grid_.Voxel_Flatten_ID(i, j);
			h_H_field_[id] = 1.f;
		}
	}
	checkCudaErrors(cudaMemcpy(d_H_field_, h_H_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}