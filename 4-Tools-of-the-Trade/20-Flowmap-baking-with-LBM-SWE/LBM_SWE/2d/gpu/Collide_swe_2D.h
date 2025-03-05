#pragma once
#ifndef COLLIDESWE2DH
#define COLLIDESWE2DH

#include "../../flow/lwflow_swe2D.h"

extern "C"
{
	void mlCollide2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param);
}
#endif // !COLLIDESWE2DH
