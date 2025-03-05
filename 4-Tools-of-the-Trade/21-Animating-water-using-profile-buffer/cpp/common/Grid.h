#pragma once

#include <ostream>
#include "config_type.h"
#include "cuda_runtime.h"
#include <cmath>

struct Grid
{
	real domain_min_corner[2], domain_max_corner[2];
	real voxel_min_corner[2], voxel_max_corner[2];
	real facex_min_corner[2], facex_max_corner[2];
	real facey_min_corner[2], facey_max_corner[2];
	real node_min_corner[2], node_max_corner[2];
	real dx[2];
	real one_over_dx[2];
	int res_v[2], res_fx[2], res_fy[2];
	int total_voxels, total_facex, total_facey;
	int max_face_num, max_res_dimx, max_res_dimy;
	int n_ghost = 1;


	// Profile Buffer
	

	Grid() {}  // default constructor
	Grid(real* minc_input, real* maxc_input, int* nv_input) 
	{
		domain_min_corner[0] = minc_input[0];
		domain_min_corner[1] = minc_input[1];
		domain_max_corner[0] = maxc_input[0];
		domain_max_corner[1] = maxc_input[1];
		res_v[0] = nv_input[0];
		res_v[1] = nv_input[1];

		dx[0] = (domain_max_corner[0] - domain_min_corner[0]) / (real)(res_v[0]);
		dx[1] = dx[0];
		one_over_dx[0] = (real)(1.0) / dx[0];
		one_over_dx[1] = one_over_dx[0];

		res_fx[0] = res_v[0] + 1; 
		res_fx[1] = res_v[1] + 2 * n_ghost;
		res_fy[0] = res_v[0] + 2 * n_ghost; 
		res_fy[1] = res_v[1] + 1;
		res_v[0] = res_v[0] + 2 * n_ghost;
		res_v[1] = res_v[1] + 2 * n_ghost;

		printf("res_v: [%d, %d]\n", res_v[0], res_v[1]);
		printf("res_fx: [%d, %d]\n", res_fx[0], res_fx[1]);
		printf("res_fy: [%d, %d]\n", res_fy[0], res_fy[1]);
		total_voxels = res_v[0] * res_v[1];
		total_facex = res_fx[0] * res_fx[1];
		total_facey = res_fy[0] * res_fy[1];
		max_face_num = std::max(total_facex, total_facey);
		max_res_dimx = std::max(res_fx[0], res_fy[0]);
		max_res_dimy = std::max(res_fx[1], res_fy[1]);
		printf("total_voxels: %d; total_facex: %d; total_facey: %d; max_face_num: %d; max_res_dimx: %d; max_res_dimy: %d\n", 
			total_voxels, total_facex, total_facey, max_face_num, max_res_dimx, max_res_dimy);

		real halfDx = (real)(0.5) * dx[0];  // assume dx[0] == dx[1]

		voxel_min_corner[0] = domain_min_corner[0] - n_ghost * dx[0] + halfDx;
		voxel_min_corner[1] = domain_min_corner[1] - n_ghost * dx[1] + halfDx;
		voxel_max_corner[0] = domain_max_corner[0] + n_ghost * dx[0] - halfDx;
		voxel_max_corner[1] = domain_max_corner[1] + n_ghost * dx[1] - halfDx;

		facex_min_corner[0] = domain_min_corner[0];
		facex_min_corner[1] = domain_min_corner[1] - halfDx;
		facex_max_corner[0] = domain_max_corner[0];
		facex_max_corner[1] = domain_max_corner[1] + halfDx;

		facey_min_corner[0] = domain_min_corner[0] - halfDx;
		facey_min_corner[1] = domain_min_corner[1];
		facey_max_corner[0] = domain_max_corner[0] + halfDx;
		facey_max_corner[1] = domain_max_corner[1];

		node_min_corner[0] = domain_min_corner[0] - (real)(n_ghost) * dx[0];
		node_min_corner[1] = domain_min_corner[1] - (real)(n_ghost) * dx[1];
		node_max_corner[0] = domain_max_corner[0] + (real)(n_ghost) * dx[0];
		node_max_corner[1] = domain_max_corner[1] + (real)(n_ghost) * dx[1];

		printf("dx: [%f, %f]; min_corner: [%f, %f]; max_corner: [%f, %f]\n", 
			dx[0], dx[1], 
			domain_min_corner[0], domain_min_corner[1],
			domain_max_corner[0], domain_max_corner[1]);
		printf("voxel_min_corner: [%f, %f]; voxel_max_corner: [%f, %f]\n",
			voxel_min_corner[0], voxel_min_corner[1],
			voxel_max_corner[0], voxel_max_corner[1]);
		printf("facex_min_corner: [%f, %f]; facex_max_corner: [%f, %f]\n",
			facex_min_corner[0], facex_min_corner[1],
			facex_max_corner[0], facex_max_corner[1]);
		printf("facey_min_corner: [%f, %f]; facey_max_corner: [%f, %f]\n",
			facey_min_corner[0], facey_min_corner[1],
			facey_max_corner[0], facey_max_corner[1]);
		printf("node_min_corner: [%f, %f]; node_max_corner: [%f, %f]\n",
			node_min_corner[0], node_min_corner[1],
			node_max_corner[0], node_max_corner[1]);
	}

	inline __device__ __host__
	bool Voxel_In_Range(int* index)
	{
		return index[0] >= 0 && index[0] < res_v[0] && index[1] >= 0 && index[1] < res_v[1];
	}

	inline __device__ __host__
	bool Voxel_In_Range(int index0, int index1)
	{
		return index0 >= 0 && index0 < res_v[0] && index1 >= 0 && index1 < res_v[1];
	}

	inline __device__ __host__
	bool Face_In_Range(int axis, int* index)
	{
		if (axis == 0)  // face x
		{
			return index[0] >= 0 && index[0] < res_fx[0] && index[1] >= 0 && index[1] < res_fx[1];
		}
		else // axis == 1 face y
		{
			return index[0] >= 0 && index[0] < res_fy[0] && index[1] >= 0 && index[1] < res_fy[1];
		}
		return false;
	}
	
	inline __device__ __host__
	bool Face_In_Range_Flatten(int axis, int flattenIdx)
	{
		if (axis == 0)  // face x
		{
			return flattenIdx >= 0 && flattenIdx < total_facex;
		}
		else // axis == 1 face y
		{
			return flattenIdx >= 0 && flattenIdx < total_facey;
		}
		return false;
	}

	inline __device__ __host__
		bool Face_In_Range(int axis, int index0, int index1)
	{
		if (axis == 0)  // face x
		{
			return index0 >= 0 && index0 < res_fx[0] && index1 >= 0 && index1 < res_fx[1];
		}
		else // axis == 1 face y
		{
			return index0 >= 0 && index0 < res_fy[0] && index1 >= 0 && index1 < res_fy[1];
		}
		return false;
	}

	inline __device__ __host__
	void Voxel(int* index, real* rtnPos)
	{
		// Warning: rtnPos should be initialize before calling this function
		rtnPos[0] = voxel_min_corner[0] + index[0] * dx[0];  // x axis 
		rtnPos[1] = voxel_min_corner[1] + index[1] * dx[1];  // y axis 	
	}

	inline __device__ __host__
	void Face(int axis, int* index, real* rtnPos)
	{
		// Warning: rtnPos should be initialize before calling this function
		if (axis == 0)  // face x
		{
			rtnPos[0] = facex_min_corner[0] + index[0] * dx[0];  // x axis 
			rtnPos[1] = facex_min_corner[1] + index[1] * dx[1];  // y axis 		
		}
		else // axis == 1 face y
		{
			rtnPos[0] = facey_min_corner[0] + index[0] * dx[0];  // x axis 
			rtnPos[1] = facey_min_corner[1] + index[1] * dx[1];  // y axis 	
		}
	}

	inline __device__ __host__
	void Clamp_To_Voxel_Domain(real* x) 
	{
		// TOCHECK: fmin fmax
		x[0] = fmin(voxel_max_corner[0], fmax(voxel_min_corner[0], x[0]));  // x axis 
		x[1] = fmin(voxel_max_corner[1], fmax(voxel_min_corner[1], x[1]));  // y axis 	
	}

	inline __device__ __host__
	void Clamp_To_Face_Domain(int axis, real* x)
	{
		// TOCHECK: fmin fmax
		if (axis == 0)  // face x
		{
			x[0] = fmin(facex_max_corner[0], fmax(facex_min_corner[0], x[0]));  // x axis 
			x[1] = fmin(facex_max_corner[1], fmax(facex_min_corner[1], x[1]));  // y axis 		
		}
		else // axis == 1 face y
		{
			x[0] = fmin(facey_max_corner[0], fmax(facey_min_corner[0], x[0]));  // x axis 
			x[1] = fmin(facey_max_corner[1], fmax(facey_min_corner[1], x[1]));  // y axis 	
		}
	}

	inline __device__ __host__
	void Get_Base_Voxel(real* x, int* rtnIdx)
	{
		Clamp_To_Voxel_Domain(x);
		rtnIdx[0] = int((x[0] - voxel_min_corner[0]) * one_over_dx[0]);  // x axis
		rtnIdx[1] = int((x[1] - voxel_min_corner[1]) * one_over_dx[1]);  // y axis
	}

	inline __device__ __host__
	void Get_Base_Face(int axis, real* x, int* rtnIdx)
	{
		Clamp_To_Face_Domain(axis, x);
		if (axis == 0)  // face x
		{
			rtnIdx[0] = int((x[0] - facex_min_corner[0]) * one_over_dx[0]);  // x axis 
			rtnIdx[1] = int((x[1] - facex_min_corner[1]) * one_over_dx[1]);  // y axis 		
		}
		else // axis == 1 face y
		{
			rtnIdx[0] = int((x[0] - facey_min_corner[0]) * one_over_dx[0]);  // x axis 
			rtnIdx[1] = int((x[1] - facey_min_corner[1]) * one_over_dx[1]);  // y axis 	
		}
	}

	// TOCHECK: this function is not called in current version
	inline __device__ __host__
	void Get_Base_Voxel_Weight(real* x, real* rtnWeight)
	{
		Clamp_To_Voxel_Domain(x);
		int	base_cell[2] = { 0, 0 };
		Get_Base_Voxel(x, base_cell);
		if ((base_cell[0] == 0 && base_cell[1] == 0) ||
			(base_cell[0] == res_v[0] - 1 && base_cell[1] == res_v[1] - 1))
		{
			rtnWeight[0] = (real)(1.0);
			rtnWeight[1] = (real)(1.0);
		}
		else
		{
			rtnWeight[0] = (real)(1.0) - (x[0] - voxel_min_corner[0]) * one_over_dx[0] + (real)(base_cell[0]);  // x axis 
			rtnWeight[1] = (real)(1.0) - (x[1] - voxel_min_corner[1]) * one_over_dx[1] + (real)(base_cell[1]);  // y axis 
		}
	}

	inline __device__ __host__
	void Get_Base_Face_Weight(int axis, real* x, real* rtnWeight)
	{
		Clamp_To_Face_Domain(axis, x);
		int	base_face[2] = { 0, 0 };
		Get_Base_Face(axis, x, base_face);
		if (axis == 0)  // face x
		{	
#if 0
			if (base_face[0] == res_fx[0] - 1 && base_face[1] == res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0);				
			}
			else if (base_face[0] == res_fx[0] - 1 && base_face[1] != res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);
			}
			else if (base_face[0] != res_fx[0] - 1 && base_face[1] == res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);
				rtnWeight[1] = (real)(1.0);
			}
			else
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);
				rtnWeight[1] = (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);
			}
#else
			rtnWeight[0] = (base_face[0] == res_fx[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);  // x axis 
			rtnWeight[1] = (base_face[1] == res_fx[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);  // y axis 
#endif
		}
		else // axis == 1 face y
		{	
#if 0
			if (base_face[0] == res_fy[0] - 1 && base_face[1] == res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0);
			}
			else if (base_face[0] == res_fy[0] - 1 && base_face[1] != res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);
			}
			else if (base_face[0] != res_fy[0] - 1 && base_face[1] == res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);
				rtnWeight[1] = (real)(1.0);
			}
			else
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);
				rtnWeight[1] = (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);
			}
#else
			rtnWeight[0] = (base_face[0] == res_fy[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(base_face[0]);  // x axis 
			rtnWeight[1] = (base_face[1] == res_fy[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(base_face[1]);  // y axis 
#endif
		}
	}

	inline __device__ __host__
	void Face_Interpolation(int axis, real* x, int* rtnBase_face, real* rtnWeight)
	{
		Clamp_To_Face_Domain(axis, x);
		Get_Base_Face(axis, x, rtnBase_face);
		if (axis == 0)  // face x
		{
#if 0
			if (rtnBase_face[0] == res_fx[0] - 1 && rtnBase_face[1] == res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0);
			}
			else if (rtnBase_face[0] == res_fx[0] - 1 && rtnBase_face[1] != res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);
			}
			else if (rtnBase_face[0] != res_fx[0] - 1 && rtnBase_face[1] == res_fx[1] - 1)
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);
				rtnWeight[1] = (real)(1.0);
			}
			else
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);
				rtnWeight[1] = (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);
			}
#else
			rtnWeight[0] = (rtnBase_face[0] == res_fx[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);  // x axis 
			rtnWeight[1] = (rtnBase_face[1] == res_fx[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);  // y axis 
#endif
		}
		else // axis == 1 face y
		{
#if 0
			if (rtnBase_face[0] == res_fy[0] - 1 && rtnBase_face[1] == res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0);
			}
			else if (rtnBase_face[0] == res_fy[0] - 1 && rtnBase_face[1] != res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0);
				rtnWeight[1] = (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);
			}
			else if (rtnBase_face[0] != res_fy[0] - 1 && rtnBase_face[1] == res_fy[1] - 1)
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);
				rtnWeight[1] = (real)(1.0);
			}
			else
			{
				rtnWeight[0] = (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);
				rtnWeight[1] = (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);
			}
#else
			rtnWeight[0] = (rtnBase_face[0] == res_fy[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);  // x axis 
			rtnWeight[1] = (rtnBase_face[1] == res_fy[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);  // y axis 
#endif
		}
	}

	inline __device__ __host__
		void Face_X_Interpolation(real* x, int* rtnBase_face, real* rtnWeight)
	{
		//Clamp_To_Face_Domain(axis, x);
		x[0] = fmin(facex_max_corner[0], fmax(facex_min_corner[0], x[0]));  // x axis 
		x[1] = fmin(facex_max_corner[1], fmax(facex_min_corner[1], x[1]));  // y axis 
		//Get_Base_Face(axis, x, rtnBase_face);
		rtnBase_face[0] = int((x[0] - facex_min_corner[0]) * one_over_dx[0]);  // x axis 
		rtnBase_face[1] = int((x[1] - facex_min_corner[1]) * one_over_dx[1]);  // y axis 		

		rtnWeight[0] = (rtnBase_face[0] == res_fx[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facex_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);  // x axis 
		rtnWeight[1] = (rtnBase_face[1] == res_fx[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facex_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);  // y axis 
	}

	inline __device__ __host__
		void Face_Y_Interpolation(real* x, int* rtnBase_face, real* rtnWeight)
	{
		//Clamp_To_Face_Domain(axis, x);
		x[0] = fmin(facey_max_corner[0], fmax(facey_min_corner[0], x[0]));  // x axis 
		x[1] = fmin(facey_max_corner[1], fmax(facey_min_corner[1], x[1]));  // y axis 	
		//Get_Base_Face(axis, x, rtnBase_face);
		rtnBase_face[0] = int((x[0] - facey_min_corner[0]) * one_over_dx[0]);  // x axis 
		rtnBase_face[1] = int((x[1] - facey_min_corner[1]) * one_over_dx[1]);  // y axis 		

		rtnWeight[0] = (rtnBase_face[0] == res_fy[0] - 1) ? (real)(1.0) : (real)(1.0) - (x[0] - facey_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_face[0]);  // x axis 
		rtnWeight[1] = (rtnBase_face[1] == res_fy[1] - 1) ? (real)(1.0) : (real)(1.0) - (x[1] - facey_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_face[1]);  // y axis 
	}

	inline __device__ __host__
	void Voxel_Interpolation(real* x, int* rtnBase_voxel, real* rtnWeight)
	{
		Clamp_To_Voxel_Domain(x);
		Get_Base_Voxel(x, rtnBase_voxel);
		if (rtnBase_voxel[0] == res_v[0] - 1 && rtnBase_voxel[1] == res_v[1] - 1)
		{
			rtnWeight[0] = (real)(1.0);
			rtnWeight[1] = (real)(1.0);
		}
		else if (rtnBase_voxel[0] == res_v[0] - 1 && rtnBase_voxel[1] != res_v[1] - 1)
		{
			rtnWeight[0] = (real)(1.0);
			rtnWeight[1] = (real)(1.0) - (x[1] - voxel_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_voxel[1]);
		}
		else if (rtnBase_voxel[0] != res_v[0] - 1 && rtnBase_voxel[1] == res_v[1] - 1)
		{
			rtnWeight[0] =  (real)(1.0) - (x[0] - voxel_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_voxel[0]);
			rtnWeight[1] =  (real)(1.0);
			return;
		}
		else
		{
			rtnWeight[0] = (real)(1.0) - (x[0] - voxel_min_corner[0]) * one_over_dx[0] + (real)(rtnBase_voxel[0]);
			rtnWeight[1] = (real)(1.0) - (x[1] - voxel_min_corner[1]) * one_over_dx[1] + (real)(rtnBase_voxel[1]);
		}
	}

	inline __device__ __host__
	real Voxel_Volume() {	return dx[0] * dx[1];	}

	inline __device__ __host__
	int Voxel_Flatten_ID(int* index) { return index[1] * res_v[0] + index[0]; }
	
	inline __device__ __host__
	int Voxel_Flatten_ID(int index0, int index1) { return index1 * res_v[0] + index0; }

	inline __device__ __host__
	int Face_Flatten_ID(int axis, int* index)
	{
#if 0
		if (axis == 0)  // face x
		{
			 return index[1] * res_fx[0] + index[0];
		}
		else if (axis == 1)  // face y
		{
			return index[1] * res_fy[0] + index[0];
		}
		else
		{
			printf("Fatal error: 3rd dimension does NOT exist\n");
			return -1;
		}
#else
		return (axis == 0) ? index[1] * res_fx[0] + index[0] :  // x axis
							 index[1] * res_fy[0] + index[0];   // y axis
#endif
	}

	inline __device__ __host__
	int Face_Flatten_ID(int axis, int index0, int index1)
	{
#if 0
		if (axis == 0)  // face x
		{
			return index1 * res_fx[0] + index0;
		}
		else if (axis == 1)  // face y
		{
			return index1 * res_fy[0] + index0;
		}
		else
		{
			printf("Fatal error: 3rd dimension does NOT exist\n");
			return -1;
		}
#else
		return (axis == 0) ? index1 * res_fx[0] + index0 :  // x axis
							 index1 * res_fy[0] + index0;   // y axis
#endif
	}

	inline __device__ __host__
	void Voxel_Unflatten_Index(int id, int* rtnIdx)
	{
		rtnIdx[1] = id / res_v[0];
		rtnIdx[0] = id - rtnIdx[1] * res_v[0];
	}

	inline __device__ __host__
	void Face_Unflatten_Index(int axis, int id, int* rtnIdx)
	{
		if (axis == 0)  // face x
		{
			rtnIdx[1] = id / res_fx[0];
			rtnIdx[0] = id - rtnIdx[1] * res_fx[0];
		}
		else if (axis == 1)  // face y
		{
			rtnIdx[1] = id / res_fy[0];
			rtnIdx[0] = id - rtnIdx[1] * res_fy[0];
		}
		else
		{
			printf("Fatal error: 3rd dimension does NOT exist\n");
			rtnIdx[0] = 0;
			rtnIdx[1] = 0;
		}
	}
};