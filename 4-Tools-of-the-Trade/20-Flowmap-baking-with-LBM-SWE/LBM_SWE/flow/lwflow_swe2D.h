#pragma once
#ifndef LWFLOW_SWE2DH
#define LWFLOW_SWE2DH

#include "../inc/mlCommon.h"
#include "../inc/mlLbmCommon.h"
#include "../inc/mlFluidCommon.h"
#include "../3rdParty/helper_cuda.h"

class  mlFlow2D
{
public:
	REAL* f;//post-stream
	REAL* fPost;//post-collide
	MLLATTICENODE_FLAG* flag;// domain flag
	REAL* ux;
	REAL* uy;
	REAL* ZBed;
	REAL* h;//water depth
	MLFluidParam2D* param;
	REAL* forcex;
	REAL* forcey;
	long count = 0;
	REAL vis_shear;
	REAL c_s = 1 / sqrtf(3);

	REAL L;
	void mlCreate(
		REAL x0, REAL y0,
		long width, long height,
		REAL deltax,
		REAL box_w, REAL box_h,
		REAL vis, REAL L, REAL gy
	);
	void mlClear();
};
inline void mlFlow2D::mlCreate(REAL x0, REAL y0,
	long width, long height,
	REAL deltax, REAL box_w, REAL box_h, REAL vis, REAL L, REAL gy)
{
	this->vis_shear = vis;
	param = new MLFluidParam2D[1];
	long sample_x_count = 0; long sample_y_count = 0;
	REAL endx = 0; REAL endy = 0;
	REAL i = 0;
	for (i = x0; i < box_w + x0; i += deltax)
	{
		sample_x_count++;
	}
	endx = i - deltax;
	for (i = y0; i < box_h + y0; i += deltax)
	{
		sample_y_count++;
	}
	endy = i - deltax;

	count = sample_x_count * sample_y_count;

	param->delta_x = deltax;	param->delta_t = deltax;
	param->validCount = count;
	param->box_sizex = box_w; 	param->box_sizey = box_h;
	param->domian_sizex = width;		param->domian_sizey = height;
	param->samplesx = sample_x_count;	param->samplesy = sample_y_count;

	param->Scale_time = 0.01;
	param->Scale_length = L / width;
	param->gx = 0;
	param->gy = gy / param->Scale_length * param->Scale_time * param->Scale_time;;
	f = new REAL[9 * count];
	fPost = new REAL[9 * count];

	ux = new REAL[count];
	uy = new REAL[count];

	flag = new MLLATTICENODE_FLAG[count];
	h = new REAL[count];
	forcex = new REAL[count];
	forcey = new REAL[count];
	ZBed = new REAL[count];

	int num = 0;
	for (long y = 0; y < sample_y_count; y++)
	{
		for (long x = 0; x < sample_x_count; x++)
		{
			flag[num] = ML_FLUID;
			ux[num] = 0; uy[num] = 0;
			h[num] = 0;
			forcex[num] = 0;
			forcey[num] = 0;
			ZBed[num] = 0;
			num++;
		}
	}
}

inline void mlFlow2D::mlClear()
{

}
#endif