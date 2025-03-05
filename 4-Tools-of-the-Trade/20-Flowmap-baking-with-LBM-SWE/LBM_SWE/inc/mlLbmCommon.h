#pragma  once
#ifndef _MLLBMCOMMON_
#define _MLLBMCOMMON_

enum MLLATTICENODE_FLAG
{
	ML_INVALID,
	ML_EMPTY,
	ML_FLUID,
	ML_FLUID_REST,
	ML_WALL,
	ML_WALL_LEFT,
	ML_WALL_RIGHT,
	ML_WALL_FOR,
	ML_WALL_BACK,
	ML_WALL_DOWN,
	ML_WALL_UP,
	ML_SOILD,
	ML_INLET,
	ML_INLET0,
	ML_INLET1,
	ML_INLET2,
	ML_OUTLET,
	ML_SMOKE,
	ML_WALL_CORNER,
	//ML_INTERFACE
	ML_XD,
	ML_YD
};

enum ML_COLLIDE_MODEL
{
	ML_COLLIDE_BGK,
	ML_COLLIDE_MRT
};

enum ML_BC_TYPE//boundary condition type
{
	ML_BC_BOUNCE_BACK
};


struct  ML2D_PARAM
{
	long sample_x_count;//sampling count in x direction
	long sample_y_count;//sampling count in y direction
	long sample_z_count;//sampling count in z direction

	long display_width;//display width
	long display_height;//display height
	long display_length;//display length

	float x0, y0,z0;//the center coordinate of the lattice grid

	float delta_grid;

	float delta_t;

	float vis_shear;//shear viscosity
	float vis_bulk;//bulk viscosity
};
#define IX2D(i,j,x) ((i)+(j)*(x))
#define IX3D(i,j,k,dim1,dim2) ((i)+(j)*(dim1)+(k)*(dim1)*(dim2))
#endif // !_MLLBMCOMMON_