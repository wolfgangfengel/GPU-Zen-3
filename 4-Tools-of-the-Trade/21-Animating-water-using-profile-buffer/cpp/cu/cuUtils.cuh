#ifndef __CU_UTILS_CUH_
#define __CU_UTILS_CUH_
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "config_type.h" 

__forceinline__ __device__ constexpr void get2DIdx(const int idx1D, const int* res, int* idx2D) {
	idx2D[0] = idx1D / res[1];
	idx2D[1] = idx1D % res[1];
}

__forceinline__ __device__ constexpr int get1DIdx(const int* idx2D, const int* res) {
	return  res[1] * idx2D[0] + idx2D[1];
}

#endif
