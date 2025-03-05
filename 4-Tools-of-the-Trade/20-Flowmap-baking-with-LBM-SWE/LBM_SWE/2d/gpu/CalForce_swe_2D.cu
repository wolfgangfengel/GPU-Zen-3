#include "../../inc/mlcudaCommon.h"
#include "../UtilFuncGpu2D.h"
#include "CalForce_swe_2D.h"

__global__ void CalForce_SWE_2DKernel
(
	mlFlow2D* mlflow,
	uint32_t sample_x,
	uint32_t sample_y,
	uint32_t sample_num
)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t curind = y * sample_x + x;
	mlEqStateGpu2f mleqstate;

	REAL g = mlflow->param->gy;
	if (
		(x >= 0 && x <= sample_x - 1)
		&&
		(y >= 0 && y <= sample_y - 1)
		)
	{
		if (mlflow[0].flag[curind] == ML_FLUID
			)
		{
			uint32_t x1 = x - 1, x2 = x + 1;
			uint32_t y1 = y - 1, y2 = y + 1;

			if (x1 < 0) x1 = 0;
			if (x2 >= sample_x) x2 = sample_x - 1;

			if (y1 < 0) y1 = 0;
			if (y2 >= sample_y) y2 = sample_y - 1;
			uint32_t index3 = y1 * sample_x + x; //(x,y-1)
			uint32_t index4 = y2 * sample_x + x; //(x,y+1)
			uint32_t index5 = y * sample_x + x1; //(x-1,y)
			uint32_t index6 = y * sample_x + x2; //(x+1,y)


			float mprate = 1.0; 
			REAL h_cur = mlflow[0].h[curind] * 1;
			REAL zbed_cur = mlflow[0].ZBed[curind] * 1;
			REAL ux_cur = mlflow[0].ux[curind];
			REAL uy_cur = mlflow[0].uy[curind];

			REAL h_avg_x = 0.5 * (mlflow[0].h[index5] + mlflow[0].h[index6]);
			REAL h_avg_y = 0.5 * (mlflow[0].h[index3] + mlflow[0].h[index4]);
			REAL zb_gx = 0;//1 * (mlflow[0].ZBed[index6] - mlflow[0].ZBed[index5]) / (2 * mprate);
			REAL zb_gy = 0;//1 * (mlflow[0].ZBed[index4] - mlflow[0].ZBed[index3]) / (2 * mprate);
			mleqstate.Cal2rdGradientZben(zb_gx, zb_gy, mlflow[0].ZBed, x, y, sample_x, sample_y);

			/*REAL force_x1 = h_cur >= zbed_cur ? -g * (h_cur)*zb_gx : 0;
			REAL force_y1 = h_cur >= zbed_cur ? -g * (h_cur)*zb_gy : 0;*/

			REAL force_x1 = h_cur >= zbed_cur ? -g * (h_cur)*zb_gx * mprate : 0;
			REAL force_y1 = h_cur >= zbed_cur ? -g * (h_cur)*zb_gy * mprate : 0;



			REAL U_norm = ux_cur * ux_cur + uy_cur * uy_cur;
			REAL cb = h_cur >= zbed_cur ? mlflow->param->MannC * mlflow->param->MannC * g / (powf(h_cur, 1.0 / 3.0)) : 0;
			REAL cb_u_norm = 1 * cb * powf(U_norm, 0.5);
			REAL bed_shear_stressx = -cb_u_norm * ux_cur;
			REAL bed_shear_stressy = -cb_u_norm * uy_cur;

			REAL ux_wind = 0;
			REAL uy_wind = 0;
			REAL U_wind_norm = ux_wind * ux_wind + uy_wind * uy_wind;
			REAL Cw = 0.0;
			REAL cw_uwind_norm = Cw * powf(U_wind_norm, 0.5);
			REAL wind_shear_stressx = -cw_uwind_norm * ux_wind;
			REAL wind_shear_stressy = -cw_uwind_norm * uy_wind;

			//Coriolis 2* 7.3 x 1e-5*sin(phi)
			REAL fc = 0;
			REAL Coriolis_x = fc * h_cur * uy_cur;
			REAL Coriolis_y = -fc * h_cur * ux_cur;

			mlflow[0].forcex[curind] = force_x1 + bed_shear_stressx + wind_shear_stressx + Coriolis_x;
			mlflow[0].forcey[curind] = force_y1 + bed_shear_stressy + wind_shear_stressy + Coriolis_y;
		}
	}
}


void CalForce_SWE_2DGpu(mlFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samplesx;
	int sample_y = param->samplesy;
	int sample_num = sample_x * sample_y;
	dim3 threads(4, 4);
	dim3 grid(
		ceil(float(param->samplesx) / threads.x),
		ceil(float(param->samplesy) / threads.y)
	);
	CalForce_SWE_2DKernel << <grid, threads >> >//mlCollide2DBGKKernel
		(mlflow, sample_x, sample_y, sample_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}