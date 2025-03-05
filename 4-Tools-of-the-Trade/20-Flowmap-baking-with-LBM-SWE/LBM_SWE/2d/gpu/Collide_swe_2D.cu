#include "../../inc/mlcudaCommon.h"
#include "../UtilFuncGpu2D.h"
#include "Collide_swe_2D.h"


__global__ void mlCollide2DKernel
(
	mlFlow2D* mlflow,
	int sample_x,
	int sample_y,
	int sample_num
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int curind = y * sample_x + x;
	REAL g = mlflow->param->gy;
	if (
		(x >= 0 && x <= sample_x - 1)
		&&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if (mlflow[0].flag[curind] == ML_FLUID
			||
			mlflow[0].flag[curind] == ML_INLET
			||
			mlflow[0].flag[curind] == ML_OUTLET
			)
		{
			mlEqStateGpu2f mleqstate;
			mlLatticeNodeD2Q9f node;
#pragma unroll 
			for (int i = 0; i < 9; i++)
			{
				int f_ind = i * sample_num + curind;
				node.f[i] = mlflow[0].f[f_ind];
			}
			mlLatticeNodeD2Q9f CentralMomentNode = node;
			REAL h = mlflow[0].h[curind];
			REAL u0x = mlflow[0].ux[curind];
			REAL u0y = mlflow[0].uy[curind];
			float vis_shear = mlflow[0].vis_shear;

			mleqstate.mlComputeCentralMomentK(u0x, u0y, CentralMomentNode);
			mlLatticeNodeD2Q9f CentralMomenTEq;
 

			mleqstate.mlComputeCentralMomentKeq(u0x, u0y, h, g, CentralMomenTEq);
			long x1 = x - 1, x2 = x + 1;
			long y1 = y - 1, y2 = y + 1;

			if (x1 < 0) x1 = 0;
			if (x2 >= sample_x) x2 = sample_x - 1;

			if (y1 < 0) y1 = 0;
			if (y2 >= sample_y) y2 = sample_y - 1;

			long index3 = y1 * sample_x + x; //(x,y-1)
			long index4 = y2 * sample_x + x; //(x,y+1)
			long index5 = y * sample_x + x1; //(x-1,y)
			long index6 = y * sample_x + x2; //(x+1,y)
			float uyx = (mlflow->uy[index6] - mlflow->uy[index5]) / 2.0f;//(2.0f*mlflowvec[curscale]->param->delta_x);
			float uyy = (mlflow->uy[index4] - mlflow->uy[index3]) / 2.0f;//(2.0f*mlflowvec[curscale]->param->delta_x);
			float uxx = (mlflow->ux[index6] - mlflow->ux[index5]) / 2.0f;// (2.0f*mlflowvec[curscale]->param->delta_x);
			float uxy = (mlflow->ux[index4] - mlflow->ux[index3]) / 2.0f;// (2.0f*mlflowvec[curscale]->param->delta_x);
			float St2 = 2.0 * (uxx * uxx + (uxy + uyx) * (uyx + uxy) / 2.0 + uyy * uyy);

			float Rxx, Rxy, Ryx, Ryy;



			REAL newS = 0.1;
			REAL newSH = 0.1;

			float s[9];
			s[0] = s[1] = s[2] = 1;
			s[3] = 1;
			s[4] = s[5] = 1 / ((vis_shear) * 3 + 0.5f);
			s[6] = s[7] = 1 / ((newS) * 3 + 0.5f);
			s[8] = 1 / ((newSH) * 3 + 0.5f);

			mlLatticeNodeD2Q9f forceTerm;
			/*mleqstate.mlComputeForceProjection(mlflow[0].forcex[curind], mlflow[0].forcey[curind], forceTerm);
			mleqstate.mlComputeCentralMomentK(u0, forceTerm);*/
			mleqstate.mlComputeForceMomentK(mlflow[0].forcex[curind], mlflow[0].forcey[curind], forceTerm);
			mlLatticeNodeD2Q9f f_curind;
			for (int m = 0; m < 9; m++)
			{
				f_curind.f[m] = (1 - s[m]) * CentralMomentNode.f[m] + s[m] * CentralMomenTEq.f[m];
			}
			for (int m = 0; m < 9; m++)
			{
				f_curind.f[m] += (1 - s[m] / 2.0) * forceTerm.f[m];;
			}
			mleqstate.mlCentralMomentPro2F(u0x, u0y, f_curind);

			for (int m = 0; m < 9; m++)
			{
				int f_ind = m * sample_num + curind;
				mlflow[0].fPost[f_ind] = f_curind.f[m];
				if (mlflow[0].fPost[f_ind] < 0) mlflow[0].fPost[f_ind] = 0;
			}
		}
	}

}

void mlCollide2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samplesx;
	int sample_y = param->samplesy;
	int sample_num = sample_x * sample_y;
	dim3 threads(4, 4);
	dim3 grid(
		ceil(float(param->samplesx) / threads.x),
		ceil(float(param->samplesy) / threads.y)
	);
	mlCollide2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
		(mlflow, sample_x, sample_y, sample_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
