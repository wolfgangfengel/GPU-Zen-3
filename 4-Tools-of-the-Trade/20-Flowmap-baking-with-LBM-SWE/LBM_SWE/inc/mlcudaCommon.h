#pragma once
#ifndef _MLCUDACOMMON_
#define _MLCUDACOMMON_


#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "../../3rdParty/helper_cuda.h"
#include "builtin_types.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define ML_CUDA_BLOCK_SIZE 8

#define MLCUDA_DEVICE


#endif // !_MLCUDACOMMON_