#pragma once
#ifndef _MLCONSTANTPARAM2DGPU_
#define _MLCONSTANTPARAM2DGPU_
#include "../inc/mlcudaCommon.h"

__constant__ float ex2d_gpu[9] = { 0,1,0,-1,0,1,-1,-1,1 };
__constant__ float ey2d_gpu[9] = { 0,0,1,0,-1,1,1,-1,-1 };
__constant__ int index2dInv_gpu[9] = { 0,3,4,1,2,7,8,5,6 };
__constant__ float w2d_gpu[9] = { 4.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };
__constant__ float cs_gpu = 0.57735f;

#endif // !_MLCONSTANTPARAMGPU_
