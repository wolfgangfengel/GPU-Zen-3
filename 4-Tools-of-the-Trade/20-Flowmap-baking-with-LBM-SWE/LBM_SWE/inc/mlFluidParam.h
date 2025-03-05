#pragma once
#ifndef _MLFLUIDPARAM_
#define _MLFLUIDPARAM_
 
 
 
 
struct MLFluidParam2D
{
	long samplesx;     //the sample numbers in each dimensions
	long samplesy;     //the sample numbers in each dimensions
	long domian_sizex;     //the domain size
	long domian_sizey;     //the domain size
	float box_sizex;     //the domain size
	float box_sizey;     //the domain size
								//mlVertex2f smoke_start_pt;
							//mlVertex2f smoke_end_pt;
							//GVLSize2f smoke_size; //the smoke size
	REAL delta_x;
	REAL delta_t;

	int scaleNum;
	REAL vis_shear;
	REAL vis_bulk;
	int validCount;
	REAL gx;
	REAL gy;
	REAL gravity;
	REAL Scale_time;
	REAL Scale_length;
	REAL MannC;
};

 
struct HybridMappingParm
{
	REAL rho0;
	REAL rhol0;
	REAL P0;
	REAL delta_x;
	REAL delta_t;
	REAL T0;

public:
	HybridMappingParm()
	{}
	HybridMappingParm
	(
		REAL _rho0,
		REAL _rhol0,
		REAL _P0,
		REAL _delta_x,
		REAL _delta_t,
		REAL _T0
	)
	{
		rho0 = _rho0;
		rhol0 = _rhol0;
		P0 = _P0;
		delta_x = _delta_x;
		delta_t = _delta_t;
		T0 = _T0;
	}
};

 

struct MLMappingParam
{
	REAL lp, tp, xp;
	REAL t0p, l0p;
	REAL N;
	REAL u0p;
	REAL viscosity_p;
	REAL viscosity_k;
	REAL labma;
	REAL roup;
public:
	MLMappingParam()
	{}
	MLMappingParam
	(
		REAL _uop,
		REAL _labma,
		REAL _l0p,
		REAL _N,
		REAL _roup
	)
	{
		u0p = _uop;
		labma = _labma;
		l0p = _l0p;
		N = _N;
		roup = _roup;
	}
};

#endif // !_MLFLUIDPARAM_
