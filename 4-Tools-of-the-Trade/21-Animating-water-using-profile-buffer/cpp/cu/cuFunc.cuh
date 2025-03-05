#ifndef __CU_FUNC_CUH_
#define __CU_FUNC_CUH_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include "cuMath.cuh"
#include "../config_type.h"
#include "../common/SSWE.ShareCode.h"

#define WarpSize 32
#define FULL_MASK 0xffffffff

namespace SSWECUDA
{
	struct RealFlagPack
	{
		real realVal;
		uint32_t flag;
	};

	struct Real2Pack
	{
		real realVal1;
		real realVal2;
	};

	struct Real3Pack
	{
		real realVal1;
		real realVal2;
		real realVal3;
	};
	__global__ void cuCompute_Water_Grid(real* water_field, real* terrain_field, real* water_normal_field,
		Grid grid, real* water_grid_pos, real* water_grid_norm,
		const real dx, const real scaling);

	__global__ void cuUpdate_Layered_Data(real* water_field,
		real* flowmap_field,
		real* ampla_field,
		real* terrain_field,
		Grid grid, real* layered_data, real* depth_data, real* foam_data, const real scaling);

	__global__ void cuUpdate_Fb_Norm_Field(real* pbs_field, real* pbf_field, real* pb_norm_field);
	__global__ void cuUpdate_Fb_Offset_Field(real* pbs_field, real* pbf_field, real* pb_offset_field);

	// Profile Buffer
	__global__ void cuPrecompute_Profile_Buffer(real* PB_field,
		real t, real k_min, real k_max, real scale, real L, int integration_nodes,
		Grid grid);
}
#endif
