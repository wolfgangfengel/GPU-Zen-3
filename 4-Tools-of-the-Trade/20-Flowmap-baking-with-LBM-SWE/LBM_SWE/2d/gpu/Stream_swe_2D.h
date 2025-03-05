#pragma once
#ifndef _MLSTREAMGPU2DH_
#define _MLSTREAMGPU2DH_

#include "../../flow/lwflow_swe2D.h"

extern "C"
{
	void Stream_SWE_2DGpu(mlFlow2D* mlflow, float w1, MLFluidParam2D* param);
}


#endif // !_MLSTREAMGPU2DH
