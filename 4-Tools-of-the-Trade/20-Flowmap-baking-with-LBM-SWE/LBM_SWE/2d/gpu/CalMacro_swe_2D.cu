#include "../../inc/mlcudaCommon.h"
#include "../UtilFuncGpu2D.h"
#include "CalMacro_swe_2D.h"

__global__ void Inlet_SWE_2DKernel
(
    mlFlow2D* mlflow, float height,
    float ux, float uy,
    int sample_x,
    int sample_y,
    int sample_num
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int curind = y * sample_x + x;
    if (
        (x >= 0 && x <= sample_x - 1)
        &&
        (y >= 0 && y <= sample_y - 1)
        )
    {
        if (mlflow[0].flag[curind] == ML_INLET)
        {
            int x1 = x;
            int y1 = y + 1;
            int ind_for = y1 * sample_x + x1;
            mlEqStateGpu2f mleqstate;
            mlLatticeNodeD2Q9f CentralMomenTEq;
            REAL h = height;// 3.5;// mlflow[0].h[ind_for];// 0.0002;//0.0 * tpflow[0].pressure[ind_for];
            REAL g = mlflow->param->gy;
            mleqstate.mlComputeCentralMomentKeq(ux, uy, h, g, CentralMomenTEq);//0.00, 0.02,
            mleqstate.mlCentralMomentPro2F(ux, uy, CentralMomenTEq);

            for (int m = 0; m < 9; m++)
            {
                int f_ind = m * sample_num + curind;
                int f_ind_for = m * sample_num + ind_for;
                mlflow[0].fPost[f_ind] = CentralMomenTEq.f[m];
            }
            mleqstate.mlGetFirstOrderMoment(CentralMomenTEq, mlflow[0].h[curind]);


        }
    }
}

__global__ void Outlet_SWE_2DKernel
(
    mlFlow2D* mlflow, REAL hin,
    int sample_x,
    int sample_y,
    int sample_num
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int curind = y * sample_x + x;
    if (
        (x >= 0 && x <= sample_x - 1)
        &&
        (y >= 0 && y <= sample_y - 1)
        )
    {

        if (mlflow[0].flag[curind] == ML_OUTLET)
        {
            int x1 = x + 1;
            int y1 = y;
            int ind_for = y1 * sample_x + x1;
            mlEqStateGpu2f mleqstate;
            mlLatticeNodeD2Q9f CentralMomenTEq;
            REAL h = hin;// mlflow[0].h[ind_for];// 0.0002;//0.0 * tpflow[0].pressure[ind_for];
            REAL ux = 0.0;// mlflow[0].ux[ind_for];
            REAL uy = 0.0;//mlflow[0].uy[ind_for];
            REAL g = mlflow->param->gy;
            mleqstate.mlComputeCentralMomentKeq(ux, uy, h, g, CentralMomenTEq);
            mleqstate.mlCentralMomentPro2F(ux, uy, CentralMomenTEq);

            for (int m = 0; m < 9; m++)
            {
                int f_ind = m * sample_num + curind;
                int f_ind_for = m * sample_num + ind_for;
                mlflow[0].fPost[f_ind] = 1.0 * CentralMomenTEq.f[m];
            }
        }
    }
}

__global__ void CalMacro_SWE_2DKernel
(
    mlFlow2D* mlflow,
    int sample_x,
    int sample_y,
    int sample_num
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int curind = y * mlflow->param->samplesx + x;
    if (
        (x >= 0 && x <= sample_x - 1)
        &&
        (y >= 0 && y <= sample_y - 1)
        )
    {
        if (mlflow[0].flag[curind] == ML_FLUID
            ||
            mlflow[0].flag[curind] == ML_INLET
            )
        {
            mlEqStateGpu2f mleqstate;
            mlLatticeNodeD2Q9f node;
            for (int i = 0; i < 9; i++)
            {
                int f_ind = i * sample_num + curind;
                node.f[i] = mlflow[0].f[f_ind];
            }
            REAL& h = mlflow[0].h[curind];
            REAL u_curindx;
            REAL u_curindy;
            mleqstate.mlGetFirstOrderMoment(node, h);
            mleqstate.mlGetSecondOrderMoment(node, u_curindx, u_curindy);
            mlflow[0].ux[curind] = u_curindx / (h);
            mlflow[0].uy[curind] = u_curindy / (h);
        }
    }
}
__global__ void AddForcetoU_SWE_2DKernel
(
    mlFlow2D* mlflow,
    int sample_x,
    int sample_y,
    int sample_num
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int curind = y * mlflow->param->samplesx + x;
    if (
        (x >= 0 && x <= sample_x - 1)
        &&
        (y >= 0 && y <= sample_y - 1)
        )
    {
        if (mlflow[0].flag[curind] == ML_FLUID
            ||
            mlflow[0].flag[curind] == ML_INLET
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
            REAL h = mlflow[0].h[curind];
            if (isnan(h))
            {
                printf("%d,%d\n", x, y);
            }
            REAL u_curindx;
            REAL u_curindy;
            mleqstate.mlGetSecondOrderMoment(node, u_curindx, u_curindy);

            mlflow[0].ux[curind] = (u_curindx + 1 / 2.0 * mlflow[0].forcex[curind]) / (h);
            mlflow[0].uy[curind] = (u_curindy + 1 / 2.0 * mlflow[0].forcey[curind]) / (h);

        }
    }
}


void Inlet_SWE_2DGpu(mlFlow2D* mlflow, float height, float ux, float uy, MLFluidParam2D* param)
{
    int sample_x = param->samplesx;
    int sample_y = param->samplesy;
    int sample_num = sample_x * sample_y;
    dim3 threads(4, 4);
    dim3 grid(
        ceil(float(param->samplesx) / threads.x),
        ceil(float(param->samplesy) / threads.y)
    );
    Inlet_SWE_2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
        (mlflow, height, ux, uy, sample_x, sample_y, sample_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Outlet_SWE_2DGpu(mlFlow2D* mlflow, REAL hin, MLFluidParam2D* param)
{
    int sample_x = param->samplesx;
    int sample_y = param->samplesy;
    int sample_num = sample_x * sample_y;
    dim3 threads(4, 4);
    dim3 grid(
        ceil(float(param->samplesx) / threads.x),
        ceil(float(param->samplesy) / threads.y)
    );
    Outlet_SWE_2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
        (mlflow, hin, sample_x, sample_y, sample_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void CalMacro_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param)
{
    int sample_x = param->samplesx;
    int sample_y = param->samplesy;
    int sample_num = sample_x * sample_y;
    dim3 threads(4, 4);
    dim3 grid(
        ceil(float(param->samplesx) / threads.x),
        ceil(float(param->samplesy) / threads.y)
    );
    CalMacro_SWE_2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
        (mlflow, sample_x, sample_y, sample_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}



void AddForcetoU_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param)
{
    int sample_x = param->samplesx;
    int sample_y = param->samplesy;
    int sample_num = sample_x * sample_y;
    dim3 threads(4, 4);
    dim3 grid(
        ceil(float(param->samplesx) / threads.x),
        ceil(float(param->samplesy) / threads.y)
    );
    AddForcetoU_SWE_2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
        (mlflow, sample_x, sample_y, sample_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
