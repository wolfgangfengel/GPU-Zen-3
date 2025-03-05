#pragma once
#ifndef _MLEQSTATEGPU2DH_
#define _MLEQSTATEGPU2DH_

#include "cuda_runtime.h"
#include "../inc/mlCommon.h"
#include "../inc/mlLatticeNode.h"

#include "mlConstantParamGpu2D.h"
template<class T>
class mlEqStateGpu2D
{
public:
	MLFUNC_TYPE void mlComputeEqstate(T ux, T uy, T hei, T G, MlLatticeNodeD2Q9& node_in_out);
	MLFUNC_TYPE void mlGetFirstOrderMoment(MlLatticeNodeD2Q9& node, T& h_out);
	MLFUNC_TYPE void mlGetSecondOrderMoment(MlLatticeNodeD2Q9& node, T& ux, T& uy);

	MLFUNC_TYPE void mlComputeCentralMomentK(T ux, T uy, MlLatticeNodeD2Q9& node_in_out);
	MLFUNC_TYPE void mlComputeCentralMomentKeq(T ux, T uy, T hei, T G, MlLatticeNodeD2Q9& node_in_out);


	MLFUNC_TYPE void mlCentralMomentPro2F(T ux, T uy, MlLatticeNodeD2Q9& node_in_out);
	MLFUNC_TYPE void mlComputeForceProjection(T fx, T fy, MlLatticeNodeD2Q9& node_in_out);
	//MLFUNC_TYPE void mlCentralMomentPro2F_step1(mlVelocity2D<T> u, MlLatticeNodeD2Q9& node_in_out);
	//MLFUNC_TYPE void mlCentralMomentPro2F_step2(MlLatticeNodeD2Q9& node_in_out);
	MLFUNC_TYPE void mlComputeForceMomentK(T fx, T fy, MlLatticeNodeD2Q9& node_in_out);


	MLFUNC_TYPE void Cal2rdGradientZben(REAL& gradx, REAL& grady,
		REAL* data, int x, int y,
		int xSize, int ySize);

};
typedef mlEqStateGpu2D<REAL> mlEqStateGpu2f;
typedef mlEqStateGpu2D<REAL> mlEqStateGpu2d;

template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlComputeCentralMomentK(T ux, T uy, MlLatticeNodeD2Q9& node_in_out)
{
	MlLatticeNodeD2Q9 node = node_in_out;
#pragma unroll 
	for (int i = 0; i < 9; i++)
	{
		node_in_out[i] = 0;
	}
#pragma unroll 
	for (int k = 0; k < 9; k++)
	{
		REAL CX = ex2d_gpu[k] -  ux;
		REAL CY = ey2d_gpu[k] -  uy;
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
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlComputeCentralMomentKeq(T ux, T uy, T H, T g, MlLatticeNodeD2Q9& node_in_out)
{
	REAL cu, U2;
	REAL U =  ux;
	REAL V =  uy;
	T gH2 = g * H * H;
	node_in_out[0] = H;
	node_in_out[1] = 0;
	node_in_out[2] = 0;
	node_in_out[3] = gH2;
	node_in_out[4] = 0;
	node_in_out[5] = 0;
	node_in_out[6] = -0.5 * gH2 * V - H * U * U * V + 1.0 / 3.0 * H * V;
	node_in_out[7] = -0.5 * gH2 * U - H * U * V * V + 1.0 / 3.0 * H * U;
	node_in_out[8] = 1.0 / 6.0 * gH2 * (3 * (U * U + V * V) + 1) + H / 3.0 * (-U * U - V * V + 9 * U * U * V * V);

	//(H * (H * g - 2 * U ^ 2 - 2 * V ^ 2 + 18 * U ^ 2 * V ^ 2 + 3 * H * U ^ 2 * g + 3 * H * V ^ 2 * g)) / 6
}
template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlComputeForceMomentK(T fx, T fy, MlLatticeNodeD2Q9& node_in_out)
{
	node_in_out[0] = 0;
	node_in_out[1] = fx;
	node_in_out[2] = fy;
	node_in_out[3] = 0;
	node_in_out[4] = 0;
	node_in_out[5] = 0;
	node_in_out[6] = 1.0f / 3.0f * fy;
	node_in_out[7] = 1.0f / 3.0f * fx;
	node_in_out[8] = 0;
}


template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlCentralMomentPro2F(T ux, T uy, MlLatticeNodeD2Q9& node_in_out)
{
	MlLatticeNodeD2Q9 node = node_in_out;

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


template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlComputeEqstate(T ux, T uy, T H, T g, MlLatticeNodeD2Q9& node_in_out)
{
	REAL cu, U2;
	REAL U = ux;
	REAL V = uy;
	REAL U2V2 = U * U + V * V;

	//for (int i = 0; i < 9; i++)
	//{
	//	cu = ex2d_gpu[i] * u.ux + ey2d_gpu[i] * u.uy; // c k*u
	//	node_in_out.f[i] = w2d_gpu[i] * rho.rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U2);
	//}
	T gH2 = g * H * H;
	node_in_out.f[0] = H - 5.0 * gH2 / 6.0 - 2.0 * H / 3.0 * U2V2;
	node_in_out.f[1] = gH2 / 6. + H / 3. * (ex2d_gpu[1] * U + ey2d_gpu[1] * V) + H / 2. * (ex2d_gpu[1] * U * ex2d_gpu[1] * U + ey2d_gpu[1] * V * ey2d_gpu[1] * V + 2. * ex2d_gpu[1] * U * ey2d_gpu[1] * V) - H / 6. * U2V2;
	node_in_out.f[2] = gH2 / 6. + H / 3. * (ex2d_gpu[2] * U + ey2d_gpu[2] * V) + H / 2. * (ex2d_gpu[2] * U * ex2d_gpu[2] * U + ey2d_gpu[2] * V * ey2d_gpu[2] * V + 2. * ex2d_gpu[2] * U * ey2d_gpu[2] * V) - H / 6. * U2V2;
	node_in_out.f[3] = gH2 / 6. + H / 3. * (ex2d_gpu[3] * U + ey2d_gpu[3] * V) + H / 2. * (ex2d_gpu[3] * U * ex2d_gpu[3] * U + ey2d_gpu[3] * V * ey2d_gpu[3] * V + 2. * ex2d_gpu[3] * U * ey2d_gpu[3] * V) - H / 6. * U2V2;
	node_in_out.f[4] = gH2 / 6. + H / 3. * (ex2d_gpu[4] * U + ey2d_gpu[4] * V) + H / 2. * (ex2d_gpu[4] * U * ex2d_gpu[4] * U + ey2d_gpu[4] * V * ey2d_gpu[4] * V + 2. * ex2d_gpu[4] * U * ey2d_gpu[4] * V) - H / 6. * U2V2;
	node_in_out.f[5] = gH2 / 24. + H / 12. * (ex2d_gpu[5] * U + ey2d_gpu[5] * V) + H / 8. * (ex2d_gpu[5] * U * ex2d_gpu[5] * U + ey2d_gpu[5] * V * ey2d_gpu[5] * V + 2. * ex2d_gpu[5] * U * ey2d_gpu[5] * V) - H / 24. * U2V2;
	node_in_out.f[6] = gH2 / 24. + H / 12. * (ex2d_gpu[6] * U + ey2d_gpu[6] * V) + H / 8. * (ex2d_gpu[6] * U * ex2d_gpu[6] * U + ey2d_gpu[6] * V * ey2d_gpu[6] * V + 2. * ex2d_gpu[6] * U * ey2d_gpu[6] * V) - H / 24. * U2V2;
	node_in_out.f[7] = gH2 / 24. + H / 12. * (ex2d_gpu[7] * U + ey2d_gpu[7] * V) + H / 8. * (ex2d_gpu[7] * U * ex2d_gpu[7] * U + ey2d_gpu[7] * V * ey2d_gpu[7] * V + 2. * ex2d_gpu[7] * U * ey2d_gpu[7] * V) - H / 24. * U2V2;
	node_in_out.f[8] = gH2 / 24. + H / 12. * (ex2d_gpu[8] * U + ey2d_gpu[8] * V) + H / 8. * (ex2d_gpu[8] * U * ex2d_gpu[8] * U + ey2d_gpu[8] * V * ey2d_gpu[8] * V + 2. * ex2d_gpu[8] * U * ey2d_gpu[8] * V) - H / 24. * U2V2;
}

template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlGetFirstOrderMoment(MlLatticeNodeD2Q9& node, T& h_out)
{
	h_out = T(0);
#pragma unroll 
	for (int i = 0; i < 9; i++)
	{
		h_out += node.f[i];
	}
}

template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlGetSecondOrderMoment(MlLatticeNodeD2Q9& node, T &ux, T &uy)
{
	ux = uy = T(0);
#pragma unroll 
	for (int i = 0; i < 9; i++)
	{
		ux += ex2d_gpu[i] * node.f[i];	uy += ey2d_gpu[i] * node.f[i];
	}
}




template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::mlComputeForceProjection(T fx, T fy, MlLatticeNodeD2Q9& node_in_out)
{

	for (int i = 0; i < 9; i++)
	{
		node_in_out[i] = (ex2d_gpu[i] * fx + ey2d_gpu[i] * fy) / 6.0;
	}
}



template<class T>
inline MLFUNC_TYPE void mlEqStateGpu2D<T>::Cal2rdGradientZben(
	REAL& gradx, REAL& grady, REAL* data, int x, int y, int xSize, int ySize)
{

	REAL gx = 0;
	REAL gy = 0;
	REAL datatmp = 0;
	int curind = y * xSize + x;
	for (int i = 0; i < 9; i++)
	{
		int xn = (x + (int)ex2d_gpu[i] + xSize) % xSize;
		int yn = (y + (int)ey2d_gpu[i] + ySize) % ySize;
		int indtmp = xn + yn * xSize;
		datatmp = data[indtmp];
		gx += 3 / 1.0 * ex2d_gpu[i] * w2d_gpu[i] * (datatmp);
		gy += 3 / 1.0 * ey2d_gpu[i] * w2d_gpu[i] * (datatmp);
	}
	gradx = gx;
	grady = gy;
}


#endif
