#include "scene/TestFDScene.h"

#include <cuda_runtime.h>
#include "cu/helper_cuda.h"
#include "cu/cuKernelWraper.cuh"

#if defined(_WIN32)
#include <GL/freeglut.h>
#else
#include <GLut/glut.h>
#endif

void TestFDScene::Initialize()
{
	//*************************************************************************************
	s_fast = 10.f;
	s_slow = 10.f;
	k_min = .01f;
	k_max = 100.f;
	ampla = .2f;
	L = 80;
	integration_nodes = 400;
	//*************************************************************************************
	Initialize_Fields();
	PrecomputeGroupSpeeds();

	t = 100.0;
	dt = grid_.dx[0] / gv * pow(10.f, -0.9f);
}

void TestFDScene::Initialize_Water_Depth_Field()
{
#ifdef TRACK_WHERE
	std::cout << "TestFDScene::Initialize_Water_Depth_Field" << std::endl;
#endif
	real domainxn = (80.f / 4000.f * DOMAIN_SCALE) / (grid_.res_v[0] - 2) * grid_.res_v[0];
	real domainxp = 8.f;
	real coeff = domainxn / domainxp * 10.f;
	for (int i = 1; i <= GRID_W; i++)
	{
		for (int j = 1; j <= GRID_L; j++)
		{
			real hw = (556.f / 4000.f * DOMAIN_SCALE) - h_H_field_[i * grid_.res_v[0] + j];
			int id = grid_.Voxel_Flatten_ID(i, j);
			h_hw_field_[id] = hw;
		}
	}
	checkCudaErrors(cudaMemcpy(d_hw_field_, h_hw_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}

void TestFDScene::Initialize_Solid_Field()
{
	std::ifstream file("..\\image\\terrain.txt");

	int width = 1024;
	int id = 0;

	int ts = 1024; int si = 1; int sj = 1;
	int shift_i = 0;
	int shift_j = 0;

	real domainxn = (80.f / 4000.f * DOMAIN_SCALE) / (grid_.res_v[0] - 2) * grid_.res_v[0];
	real domainxp = 8.f;
	real coeff = domainxn / domainxp * 10.f;
	for (int j = sj; j <= sj + ts - 1; ++j)
	{
		for (int i = si; i <= si + ts - 1; ++i)
		{
			int id = grid_.Voxel_Flatten_ID(i, j);
			h_H_field_[id] = 1.f;
		}
	}

	checkCudaErrors(cudaMemcpy(d_H_field_, h_H_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}