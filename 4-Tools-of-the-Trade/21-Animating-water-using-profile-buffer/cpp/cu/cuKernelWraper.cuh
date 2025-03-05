#ifndef __CU_KERNEL_WRAPER_CUH_
#define __CU_KERNEL_WRAPER_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include "CudaExecutionPolicy.h"
#include "CudaDevice.h"

#include "cuFunc.cuh"

#include "../config_type.h"
#include "../common/SSWE.ShareCode.h"

namespace SSWECUDA
{
	void Precompute_Profile_Buffer(real* d_PB_field_,
		real t, real k_min, real k_max, real scale, real L, int integration_nodes,
		Grid grid);
	void Compute_Water_Grid(real* water_field, real* terrain_field, real* water_normal_field,
		Grid grid, real* water_grid_pos, real* water_grid_norm, const real dx);
	void Update_Layered_Data(real* water_field,
		real* flowmap_field,
		real* ampla_field,
		real* terrain_field,
		Grid grid, real* layered_data, real* depth_data, real* foam_data, const real dx);
	void Update_Fb_Norm_Field(real* pbs_field, real* pbf_field, real* pb_norm_field);
	void Update_Fb_Offset_Field(real* pbs_field, real* pbf_field, real* pb_offset_field);
}

#endif
