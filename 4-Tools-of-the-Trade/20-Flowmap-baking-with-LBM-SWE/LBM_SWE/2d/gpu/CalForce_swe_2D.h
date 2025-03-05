#pragma once
#ifndef CALFORCESWE2DH
#define CALFORCESWE2DH

#include "../../flow/lwflow_swe2D.h"
extern "C"
{
	void CalForce_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param);
}


#endif