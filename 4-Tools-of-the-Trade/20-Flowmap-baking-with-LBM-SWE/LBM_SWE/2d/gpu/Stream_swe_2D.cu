#include "../../inc/mlcudaCommon.h"
#include "../mlConstantParamGpu2D.h"
#include "../UtilFuncGpu2D.h"
#include "Stream_swe_2D.h"

__global__ void mlStream2DKernel
(
    mlFlow2D* mlflow,
    float w1,
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
            REAL h = mlflow[0].h[curind];
            REAL u0x = 0.0;//mlflow[0].ux[curind];
            REAL u0y = 0.0;//mlflow[0].uy[curind];
            mlLatticeNodeD2Q9f CentralMomenTEq;
            mleqstate.mlComputeEqstate(u0x, u0y, h, g, CentralMomenTEq);
#pragma unroll 
            for (int i = 0; i < 9; i++)
            {
                int x1 = x - int(ex2d_gpu[i]);
                int y1 = y - int(ey2d_gpu[i]);
                int ind_back = y1 * sample_x + x1;
                if (
                    x1 >= 0 && x1 < sample_x &&
                    y1 >= 0 && y1 < sample_y
                    )
                {
                    if (
                        mlflow[0].flag[ind_back] == ML_WALL_LEFT || mlflow[0].flag[ind_back] == ML_WALL_RIGHT ||
                        mlflow[0].flag[ind_back] == ML_WALL_FOR || mlflow[0].flag[ind_back] == ML_WALL_BACK ||
                        mlflow[0].flag[ind_back] == ML_WALL_DOWN || mlflow[0].flag[ind_back] == ML_WALL_UP ||
                        mlflow[0].flag[ind_back] == ML_SOILD || (mlflow[0].h[ind_back] < mlflow[0].ZBed[ind_back] && mlflow[0].flag[ind_back] == ML_FLUID)
                        )
                    {
                        int f_ind1 = i * sample_num + curind;
                        int f_ind2 = index2dInv_gpu[i] * sample_num + curind;
                        mlflow[0].f[f_ind1] = (1 - w1) * mlflow[0].fPost[f_ind2] + w1 * CentralMomenTEq[index2dInv_gpu[i]];
                    }
                    else
                    {
                        int f_ind1 = i * sample_num + curind;
                        int f_ind2 = i * sample_num + ind_back;
                        mlflow[0].f[f_ind1] = mlflow[0].fPost[f_ind2];
                    }
                }
            }
        }
    }
}

void Stream_SWE_2DGpu(mlFlow2D* mlflow, float w1, MLFluidParam2D* param)
{
    int sample_x = param->samplesx;
    int sample_y = param->samplesy;
    int sample_num = sample_x * sample_y;
    dim3 threads(4, 4);
    dim3 grid(
        ceil(float(param->samplesx) / threads.x),
        ceil(float(param->samplesy) / threads.y)
    );
    mlStream2DKernel << <grid, threads >> >
        (mlflow, w1, sample_x, sample_y, sample_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}