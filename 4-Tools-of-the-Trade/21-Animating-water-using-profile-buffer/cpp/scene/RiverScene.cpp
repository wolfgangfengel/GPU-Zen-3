#include "scene/RiverScene.h"

#include <cuda_runtime.h>
#include "cu/helper_cuda.h"
#include "cu/cuKernelWraper.cuh"

#include "heightmap.cpp"
auto& hmap = height_data;

#if defined(_WIN32)
#include <GL/freeglut.h>
#else
#include <GLut/glut.h>
#endif

void RiverScene::Initialize()
{
	//*************************************************************************************
	//* Profile Buffer @river
	//*************************************************************************************
	s_fast = 10.f;
	s_slow = 10.f;
	k_min = .1f;
	k_max = 200.f;


	ampla = 0.15f;
	ampla = 0.075f;
	L = 120;
	L = 20;
	integration_nodes = 100;
	//*************************************************************************************

	Initialize_Fields();
	PrecomputeGroupSpeeds();

	t = 100.0;
	dt = grid_.dx[0] / gv * pow(10.f, -0.9f);
}

void RiverScene::Initialize_Water_Depth_Field()
{
#ifdef TRACK_WHERE
	std::cout << "RiverScene::Initialize_Water_Depth_Field" << std::endl;
#endif
	real domainxn = (80.f / 4000.f * DOMAIN_SCALE) / (grid_.res_v[0] - 2) * grid_.res_v[0];
	real domainxp = 8.f;
	real coeff = domainxn / domainxp * 10.f;
	for (int i = 1; i <= GRID_W; i++)
	{
		for (int j = 1; j <= GRID_L; j++)
		{
			real hw = (356.f / 4000.f * DOMAIN_SCALE) - h_H_field_[i * grid_.res_v[0] + j];
			h_hw_field_[i * grid_.res_v[1] + j] = hw;
		}
	}
	checkCudaErrors(cudaMemcpy(d_hw_field_, h_hw_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}

void RiverScene::Initialize_Solid_Field()
{
	std::ifstream file("..\\image\\terrain.txt");

	int width = 1024;
	real* data = new real[width * width];
	real cur_data;
	int id = 0;

	if (file.is_open()) {
		std::string line;
		real h_max = (real)-1.f;
		real h_min = (real)100000000000.f;
		while (std::getline(file, line)) {
			// using printf() in all tests for consistency
			cur_data = (float)std::atof(line.c_str());
			if (cur_data > h_max)
			{
				h_max = cur_data;
			}
			else if (cur_data < h_min)
			{
				h_min = cur_data;
			}
			data[id++] = cur_data;
		}
		file.close();
		printf("min: %f, max: %f\n", h_min, h_max);
	}

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
			real H_cur = data[(j - sj) * width + (i - si)];
			real H_cur_scaled = coeff * (H_cur / (real)65535.0f * 8.f + 0.02f);
			h_H_field_[id] = H_cur_scaled;
		}
	}
	delete[] data;

	checkCudaErrors(cudaMemcpy(d_H_field_, h_H_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
}