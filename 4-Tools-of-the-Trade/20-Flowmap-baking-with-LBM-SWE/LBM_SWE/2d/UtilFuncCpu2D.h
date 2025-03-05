#pragma once
#ifndef _MLEQSTATEGPU2DH_
#define _MLEQSTATEGPU2DH_

#include "cuda_runtime.h"
#include "../inc/mlCommon.h"
#include "../inc/mlLatticeNode.h"

#include "mlConstantParamCpu2D.h"
template<class T>
class mlEqStateCpu2D
{
public:
	MLFUNC_TYPE void mlComputeEqstate(T ux, T uy, T hei, T G, MlLatticeNodeD2Q9& node_in_out);
	MLFUNC_TYPE void mlGetFirstOrderMoment(MlLatticeNodeD2Q9& node, T& h_out);
	MLFUNC_TYPE void mlGetSecondOrderMoment(MlLatticeNodeD2Q9& node, T& ux, T& uy);

	MLFUNC_TYPE void mlComputeCentralMomentK(T ux, T uy,  MlLatticeNodeD2Q9& node_in_out);

	MLFUNC_TYPE void mlCentralMomentPro2F(T ux, T uy,  MlLatticeNodeD2Q9& node_in_out);
 

};
typedef mlEqStateCpu2D<REAL> mlEqStateCpu2f;
typedef mlEqStateCpu2D<REAL> mlEqStateCpu2d;




template<class T>
inline MLFUNC_TYPE void mlEqStateCpu2D<T>::mlComputeEqstate(T ux, T uy,  T H, T g, MlLatticeNodeD2Q9& node_in_out)
{
	REAL U =  ux;
	REAL V =  uy;
	REAL U2V2 = U * U + V * V;

	//for (int i = 0; i < 9; i++)
	//{
	//	cu = ex2d_cpu[i] * u.ux + ey2d_cpu[i] * u.uy; // c k*u
	//	node_in_out.f[i] = w2d_gpu[i] * rho.rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U2);
	//}
	T gH2 = g * H * H;
	node_in_out.f[0] = H - 5.0 * gH2 / 6.0 - 2.0 * H / 3.0 * U2V2;
	node_in_out.f[1] = gH2 / 6. + H / 3. * (ex2d_cpu[1] * U + ey2d_cpu[1] * V) + H / 2. * (ex2d_cpu[1] * U * ex2d_cpu[1] * U + ey2d_cpu[1] * V * ey2d_cpu[1] * V + 2. * ex2d_cpu[1] * U * ey2d_cpu[1] * V) - H / 6. * U2V2;
	node_in_out.f[2] = gH2 / 6. + H / 3. * (ex2d_cpu[2] * U + ey2d_cpu[2] * V) + H / 2. * (ex2d_cpu[2] * U * ex2d_cpu[2] * U + ey2d_cpu[2] * V * ey2d_cpu[2] * V + 2. * ex2d_cpu[2] * U * ey2d_cpu[2] * V) - H / 6. * U2V2;
	node_in_out.f[3] = gH2 / 6. + H / 3. * (ex2d_cpu[3] * U + ey2d_cpu[3] * V) + H / 2. * (ex2d_cpu[3] * U * ex2d_cpu[3] * U + ey2d_cpu[3] * V * ey2d_cpu[3] * V + 2. * ex2d_cpu[3] * U * ey2d_cpu[3] * V) - H / 6. * U2V2;
	node_in_out.f[4] = gH2 / 6. + H / 3. * (ex2d_cpu[4] * U + ey2d_cpu[4] * V) + H / 2. * (ex2d_cpu[4] * U * ex2d_cpu[4] * U + ey2d_cpu[4] * V * ey2d_cpu[4] * V + 2. * ex2d_cpu[4] * U * ey2d_cpu[4] * V) - H / 6. * U2V2;
	node_in_out.f[5] = gH2 / 24. + H / 12. * (ex2d_cpu[5] * U + ey2d_cpu[5] * V) + H / 8. * (ex2d_cpu[5] * U * ex2d_cpu[5] * U + ey2d_cpu[5] * V * ey2d_cpu[5] * V + 2. * ex2d_cpu[5] * U * ey2d_cpu[5] * V) - H / 24. * U2V2;
	node_in_out.f[6] = gH2 / 24. + H / 12. * (ex2d_cpu[6] * U + ey2d_cpu[6] * V) + H / 8. * (ex2d_cpu[6] * U * ex2d_cpu[6] * U + ey2d_cpu[6] * V * ey2d_cpu[6] * V + 2. * ex2d_cpu[6] * U * ey2d_cpu[6] * V) - H / 24. * U2V2;
	node_in_out.f[7] = gH2 / 24. + H / 12. * (ex2d_cpu[7] * U + ey2d_cpu[7] * V) + H / 8. * (ex2d_cpu[7] * U * ex2d_cpu[7] * U + ey2d_cpu[7] * V * ey2d_cpu[7] * V + 2. * ex2d_cpu[7] * U * ey2d_cpu[7] * V) - H / 24. * U2V2;
	node_in_out.f[8] = gH2 / 24. + H / 12. * (ex2d_cpu[8] * U + ey2d_cpu[8] * V) + H / 8. * (ex2d_cpu[8] * U * ex2d_cpu[8] * U + ey2d_cpu[8] * V * ey2d_cpu[8] * V + 2. * ex2d_cpu[8] * U * ey2d_cpu[8] * V) - H / 24. * U2V2;
}

template<class T>
inline MLFUNC_TYPE void mlEqStateCpu2D<T>::mlGetFirstOrderMoment(MlLatticeNodeD2Q9& node, T& h_out)
{
	h_out = T(0);
	for (int i = 0; i < 9; i++)
	{
		h_out += node.f[i];
	}
}

template<class T>
inline MLFUNC_TYPE void mlEqStateCpu2D<T>::mlGetSecondOrderMoment(MlLatticeNodeD2Q9& node, T &ux, T &uy)
{
	ux = uy = T(0);
	for (int i = 0; i < 9; i++)
	{
		ux += ex2d_cpu[i] * node.f[i];	uy += ey2d_cpu[i] * node.f[i];
	}
}

template<class T>
inline MLFUNC_TYPE void mlEqStateCpu2D<T>::mlComputeCentralMomentK(T ux, T uy, MlLatticeNodeD2Q9& node_in_out)
{
	MlLatticeNodeD2Q9 node = node_in_out;
	for (int i = 0; i < 9; i++)
	{
		node_in_out[i] = 0;
	}
	for (int k = 0; k < 9; k++)
	{
		REAL CX = ex2d_cpu[k] - ux;
		REAL CY = ey2d_cpu[k] - uy;
		REAL ftemp = node[k];
		node_in_out[0] += ftemp;
		node_in_out[1] += ftemp * CX;
		node_in_out[2] += ftemp * CY;
		node_in_out[3] += ftemp * (CX * CX + CY * CY);
		node_in_out[4] += ftemp * (CX * CX - CY * CY);
		node_in_out[5] += ftemp * CX * CY;
		node_in_out[6] += ftemp * CX * CX * CY;
		node_in_out[7] += ftemp * CX * CY * CY;
		node_in_out[8] += ftemp * CX * CX * CY * CY;
	}
}

template<class T>
inline MLFUNC_TYPE void mlEqStateCpu2D<T>::mlCentralMomentPro2F(T ux, T uy, MlLatticeNodeD2Q9& node_in_out)
{
	MlLatticeNodeD2Q9 node = node_in_out;
	REAL ux = ux; REAL  uy = uy;

	REAL k0, k1, k2, k3, k4, k5, k6, k7, k8;
	k0 = node_in_out.f[0];
	k1 = node_in_out.f[1];
	k2 = node_in_out.f[2];
	k3 = node_in_out.f[3];
	k4 = node_in_out.f[4];
	k5 = node_in_out.f[5];
	k6 = node_in_out.f[6];
	k7 = node_in_out.f[7];
	k8 = node_in_out.f[8];
	//rho = d[0];
	node_in_out.f[0] = -k0 * (-ux * ux * uy * uy + ux * ux + uy * uy - 1) - k1 * (-2 * ux * uy * uy + 2 * ux) - k2 * (-2 * uy * ux * ux + 2 * uy) + k3 / 2 * (-2 + ux * ux + uy * uy) + k4 / 2 * (-ux * ux + uy * uy) + 4 * k5 * ux * uy + 2 * k6 * uy + 2 * k7 * ux + k8;
	node_in_out.f[1] = -(k1 * (2 * ux + 1) * (uy * uy - 1)) / 2 - k2 * ux * uy * (ux + 1) - (k0 * ux * (uy * uy - 1) * (ux + 1)) / 2 + k3 / 4 * (1 - ux - ux * ux - uy * uy) + k4 / 4 * (1 + ux + ux * ux - uy * uy) + k5 * (-uy - 2 * ux * uy) - k6 * uy + k7 / 2 * (-1 - 2 * ux) - k8 / 2;
	node_in_out.f[2] = -(k2 * (ux * ux - 1) * (2 * uy + 1)) / 2 - k1 * ux * uy * (uy + 1) - (k0 * uy * (ux * ux - 1) * (uy + 1)) / 2 + k3 / 4 * (1 - uy - ux * ux - uy * uy) + k4 / 4 * (-1 - uy + ux * ux - uy * uy) + k5 * (-ux - 2 * ux * uy) + k6 / 2 * (-1 - 2 * uy) - k7 * ux - k8 / 2;
	node_in_out.f[3] = -(k1 * (2 * ux - 1) * (uy * uy - 1)) / 2 - k2 * ux * uy * (ux - 1) - (k0 * ux * (uy * uy - 1) * (ux - 1)) / 2 + k3 / 4 * (1 + ux - ux * ux - uy * uy) + k4 / 4 * (1 - ux + ux * ux - uy * uy) + k5 * (uy - 2 * ux * uy) - k6 * uy + k7 / 2 * (1 - 2 * ux) - k8 / 2;
	node_in_out.f[4] = -(k2 * (ux * ux - 1) * (2 * uy - 1)) / 2 - k1 * ux * uy * (ux - 1) - (k0 * uy * (ux * ux - 1) * (uy - 1)) / 2 + k3 / 4 * (1 + uy - ux * ux - uy * uy) + k4 / 4 * (-1 + uy + ux * ux - uy * uy) + k5 * (ux - 2 * ux * uy) + k6 / 2 * (1 - 2 * uy) - k7 * ux - k8 / 2;
	node_in_out.f[5] = (k2 * ux * (2 * uy + 1) * (ux + 1)) / 4 + (k1 * uy * (2 * ux + 1) * (uy + 1)) / 4 + (k0 * ux * uy * (ux + 1) * (uy + 1)) / 4 + k3 / 8 * (ux + uy + ux * ux + uy * uy) + k4 / 8 * (-ux + uy - ux * ux + uy * uy) + k5 / 4 * (1 + 2 * ux + 2 * uy + 4 * ux * uy) + k6 / 4 * (1 + 2 * uy) + k7 / 4 * (1 + 2 * ux) + k8 / 4;
	node_in_out.f[6] = (k2 * ux * (2 * uy + 1) * (ux - 1)) / 4 + (k1 * uy * (2 * ux - 1) * (uy + 1)) / 4 + (k0 * ux * uy * (ux - 1) * (uy + 1)) / 4 + k3 / 8 * (-ux + uy + ux * ux + uy * uy) + k4 / 8 * (ux + uy - ux * ux + uy * uy) + k5 / 4 * (-1 + 2 * ux - 2 * uy + 4 * ux * uy) + k6 / 4 * (1 + 2 * uy) + k7 / 4 * (-1 + 2 * ux) + k8 / 4;
	node_in_out.f[7] = (k2 * ux * (2 * uy - 1) * (ux - 1)) / 4 + (k1 * uy * (2 * ux - 1) * (uy - 1)) / 4 + (k0 * ux * uy * (ux - 1) * (uy - 1)) / 4 + k3 / 8 * (-ux - uy + ux * ux + uy * uy) + k4 / 8 * (ux - uy - ux * ux + uy * uy) + k5 / 4 * (1 - 2 * ux - 2 * uy + 4 * ux * uy) + k6 / 4 * (-1 + 2 * uy) + k7 / 4 * (-1 + 2 * ux) + k8 / 4;
	node_in_out.f[8] = (k2 * ux * (2 * uy - 1) * (ux + 1)) / 4 + (k1 * uy * (2 * ux + 1) * (uy - 1)) / 4 + (k0 * ux * uy * (ux + 1) * (uy - 1)) / 4 + k3 / 8 * (ux - uy + ux * ux + uy * uy) + k4 / 8 * (-ux - uy - ux * ux + uy * uy) + k5 / 4 * (-1 - 2 * ux + 2 * uy + 4 * ux * uy) + k6 / 4 * (-1 + 2 * uy) + k7 / 4 * (1 + 2 * ux) + k8 / 4;

}

 
#endif
