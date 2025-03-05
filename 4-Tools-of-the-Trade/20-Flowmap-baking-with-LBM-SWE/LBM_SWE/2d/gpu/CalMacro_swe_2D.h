#pragma once
#ifndef CALMACROSWE2DH
#define CALMACROSWE2DH

#include "../../flow/lwflow_swe2D.h"

extern "C"
{
	void Inlet_SWE_2DGpu(mlFlow2D* mlflow, float height, float ux, float uy, MLFluidParam2D* param);
	void Outlet_SWE_2DGpu(mlFlow2D* mlflow, REAL hin, MLFluidParam2D* param);
	void CalMacro_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param);
	void AddForcetoU_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param);
}



#endif // !CALMACROSWE2DH
