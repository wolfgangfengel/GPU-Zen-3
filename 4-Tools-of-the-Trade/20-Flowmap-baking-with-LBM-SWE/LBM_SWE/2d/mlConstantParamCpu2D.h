#pragma once
#ifndef _MLCONSTANTPARAMCPU2D_
#define _MLCONSTANTPARAMCPU2D_
float ex2d_cpu[9] = { 0,1,0,-1,0,1,-1,-1,1 };
float ey2d_cpu[9] = { 0,0,1,0,-1,1,1,-1,-1, };
int index2dInv_cpu[9] = { 0,3,4,1,2,7,8,5,6 };
float w2d_cpu[9] = { 4.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };

float cs_cpu = 1.0 / sqrtf(3);
#endif // !_MLCONSTANTPARAMCPU_