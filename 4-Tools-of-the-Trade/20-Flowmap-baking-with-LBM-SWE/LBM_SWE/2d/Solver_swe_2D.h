#pragma once
#ifndef SOLVERSWE2DH
#define SOLVERSWE2DH
#include "../inc/mlCoreWin.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "../3rdParty/helper_cuda.h"


#include "cpu/Init_swe_2D.h"
#include "gpu/CalForce_swe_2D.h"
#include "gpu/CalMacro_swe_2D.h"
#include "gpu/Collide_swe_2D.h"
#include "gpu/Stream_swe_2D.h"
#include "../inc/colorramp.h"


using namespace Mfree;
#include <vector>
#include <fstream>
#include<iostream>

class  Solver_swe_2D
{
public:
    void AttachLbmvecHost(std::vector<mlFlow2D* > lbmvec);
    void AttachLbmDevice(std::vector<mlFlow2D*> lbmvec_dev);
    void mlInitFig_3_4(const char* filename);
    void mlInitFig_3_5(const char* filename);
    void mlIterateGpu();



    void mlTransData2Gpu();
    void mlDeepCopy(mlFlow2D* mllbm_host, mlFlow2D* mllbm_dev, int i);
    void mlTransData2Host(int i);
    void mlClear();
    void mlSavePPM(const char* filename, REAL* data, int mWidth, int mHeight);
    void mlResamplCutSlice(long upw, long uph, int scaleNum, int frame);

    void saveVData(const char* filename);

public:
    std::vector<mlFlow2D* > lbmvec;
    std::vector<mlFlow2D*> lbm_dev_gpu;
    REAL L;
    int gpuId = 0;
    float w1 = 0.5;
    float hin = 3.01;
    float hout = 3.01;
    float uxinit = 0;
    float uyinit = 0;
    float MapRate = 1;
    float Vmag = 0;
    bool isboundary = false;
    bool addforce = false;
    bool changeInlet = false;
public:
    std::vector<REAL*>               f_dev_vec;
    std::vector<REAL*>               fPost_dev_vec;
    std::vector<REAL*>				h_dev_vec;
    std::vector<REAL*>		       ux_dev_vec;
    std::vector<REAL*>		       uy_dev_vec;
    std::vector<MLLATTICENODE_FLAG*>  flag_dev_vec;
    std::vector<MLFluidParam2D*>        param_dev_vec;
    std::vector<REAL*>               forcex_dev_vec;
    std::vector<REAL*>               forcey_dev_vec;

private:
    mlInitHandler2D mlInitHandler;

};

void Solver_swe_2D::AttachLbmvecHost(std::vector<mlFlow2D*> lbmvec)
{
    this->lbmvec = lbmvec;
}

inline void Solver_swe_2D::AttachLbmDevice(std::vector<mlFlow2D*> lbmvec_dev)
{
    this->lbm_dev_gpu = lbmvec_dev;
}

inline void Solver_swe_2D::mlInitFig_3_4(const char* filename)
{
    mlInitHandler.loadHeightMap(filename, MapRate);
    mlInitHandler.mlInitCascadeBoundaryCpu_Fig3_4(lbmvec, 0, L);
    mlInitHandler.mlInitCascadeCpu_Fig3_4(lbmvec, 0, L);
}


inline void Solver_swe_2D::mlInitFig_3_5(const char* filename)
{
    mlInitHandler.loadHeightMap(filename, MapRate);
    mlInitHandler.mlInitCascadeBoundaryCpu_Fig3_5(lbmvec, 0, L);
    mlInitHandler.mlInitCascadeCpu_Fig3_5(lbmvec, 0, L);
}

inline void Solver_swe_2D::mlIterateGpu()
{
    int scale = 0;
    checkCudaErrors(cudaSetDevice(gpuId));
    for (int i = 0; i < lbmvec.size(); i++)
    {
        if (changeInlet)
            Inlet_SWE_2DGpu(lbm_dev_gpu[i], hin, uxinit, uyinit, lbmvec[i]->param);
        Stream_SWE_2DGpu(lbm_dev_gpu[i], w1, lbmvec[i]->param);
        CalMacro_SWE_2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);
        if (addforce)
        {
            CalForce_SWE_2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);
            AddForcetoU_SWE_2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);
        }
        mlCollide2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);
        Outlet_SWE_2DGpu(lbm_dev_gpu[i], hout, lbmvec[i]->param);
    }
}

inline void Solver_swe_2D::mlTransData2Gpu()
{
    checkCudaErrors(cudaSetDevice(gpuId));
    for (int i = 0; i < lbmvec.size(); i++)
    {
        if (lbm_dev_gpu[i] != NULL)
        {
            mlClear();
            checkCudaErrors(cudaFree(lbm_dev_gpu[i]));
        }
        cudaMalloc((void**)&lbm_dev_gpu[i], sizeof(mlFlow2D));
        mlDeepCopy(lbmvec[i], lbm_dev_gpu[i], i);
    }
}

inline void Solver_swe_2D::mlDeepCopy(mlFlow2D* mllbm_host, mlFlow2D* mllbm_dev, int i)
{
    float* f_dev;
    float* fPost_dev;
    float* h_dev;
    float* ux_dev;
    float* uy_dev;
    MLLATTICENODE_FLAG* flag_dev;
    MLFluidParam2D* param_dev;
    float* forcex_dev;
    float* forcey_dev;
    float* zbed_dev;
#pragma region MallocData
    checkCudaErrors(cudaMalloc(&f_dev, 9 * mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fPost_dev, 9 * mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&h_dev, mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&ux_dev, mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&uy_dev, mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&flag_dev, mllbm_host->count * sizeof(MLLATTICENODE_FLAG)));
    checkCudaErrors(cudaMalloc(&param_dev, 1 * sizeof(MLFluidParam2D)));
    checkCudaErrors(cudaMalloc(&forcex_dev, mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&forcey_dev, mllbm_host->count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&zbed_dev, mllbm_host->count * sizeof(float)));

#pragma endregion

#pragma region MEMCPY
    checkCudaErrors(cudaMemcpy(f_dev, mllbm_host->f, 9 * mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(fPost_dev, mllbm_host->fPost, 9 * mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(h_dev, mllbm_host->h, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ux_dev, mllbm_host->ux, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(uy_dev, mllbm_host->uy, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(flag_dev, mllbm_host->flag, mllbm_host->count * sizeof(MLLATTICENODE_FLAG), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(param_dev, mllbm_host->param, 1 * sizeof(MLFluidParam2D), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(forcex_dev, mllbm_host->forcex, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(forcey_dev, mllbm_host->forcey, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(zbed_dev, mllbm_host->ZBed, mllbm_host->count * sizeof(float), cudaMemcpyHostToDevice));
#pragma endregion
    checkCudaErrors(cudaMemcpy(mllbm_dev, mllbm_host, 1 * sizeof(mlFlow2D), cudaMemcpyHostToDevice));

#pragma region DeepCOPY
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->f), &f_dev, sizeof(f_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->fPost), &fPost_dev, sizeof(fPost_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->h), &h_dev, sizeof(h_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->ux), &ux_dev, sizeof(ux_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->uy), &uy_dev, sizeof(uy_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->flag), &flag_dev, sizeof(flag_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->param), &param_dev, sizeof(param_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->forcex), &forcex_dev, sizeof(forcex_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->forcey), &forcey_dev, sizeof(forcey_dev), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(mllbm_dev->ZBed), &zbed_dev, sizeof(zbed_dev), cudaMemcpyHostToDevice));



#pragma endregion

    f_dev_vec.push_back(f_dev);
    fPost_dev_vec.push_back(fPost_dev);
    h_dev_vec.push_back(h_dev);
    ux_dev_vec.push_back(ux_dev);
    uy_dev_vec.push_back(uy_dev);
    flag_dev_vec.push_back(flag_dev);
    param_dev_vec.push_back(param_dev);
    forcex_dev_vec.push_back(forcex_dev);
    forcey_dev_vec.push_back(forcey_dev);
}

inline void Solver_swe_2D::mlTransData2Host(int i)
{
    mlFlow2D* mllbm_host = new mlFlow2D();
    checkCudaErrors(cudaSetDevice(gpuId));
    checkCudaErrors(cudaMemcpy(mllbm_host, lbm_dev_gpu[i], 1 * sizeof(mlFlow2D), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lbmvec[i]->h, (mllbm_host->h), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lbmvec[i]->ux, (mllbm_host->ux), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lbmvec[i]->uy, (mllbm_host->uy), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lbmvec[i]->forcex, (mllbm_host->forcex), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lbmvec[i]->forcey, (mllbm_host->forcey), lbmvec[i]->count * sizeof(float), cudaMemcpyDeviceToHost));

}

inline void Solver_swe_2D::mlClear()
{
}

inline void Solver_swe_2D::mlSavePPM(const char* filename, REAL* data, int mWidth, int mHeight)
{
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
    for (unsigned i = 0; i < mWidth * mHeight * 3; ++i) {
        ofs << (unsigned char)(data[i] * 255);
    }
    ofs.close();
}

inline void Solver_swe_2D::mlResamplCutSlice(long upw, long uph, int scaleNum, int frame)
{
    int upnum = 0;
    upw = lbmvec[0]->param->samplesx;  uph = lbmvec[0]->param->samplesy;
    float* numofcomV = new float[lbmvec[0]->param->samplesy * lbmvec[0]->param->samplesx];
    float* numofcomF = new float[lbmvec[0]->param->samplesy * lbmvec[0]->param->samplesx];
    float* numofcomH = new float[lbmvec[0]->param->samplesy * lbmvec[0]->param->samplesx];
    float* numofcomVor = new float[lbmvec[0]->param->samplesy * lbmvec[0]->param->samplesx];
    float minh = 100;
    float maxh = -100;
    //#pragma omp parallel  for  
    float sumHeight = 0;
    for (int y = 0; y < lbmvec[0]->param->samplesy; y++)
    {
        for (long x = 0; x < lbmvec[0]->param->samplesx; x += 1)
        {
            int ind = y * lbmvec[0]->param->samplesx + x;
            numofcomV[ind] = pow((pow(lbmvec[0]->ux[ind], 2) +
                pow(lbmvec[0]->uy[ind], 2)), 0.5);
            if (lbmvec[0]->flag[ind] == ML_INLET)
            {
                numofcomV[ind] = 1;
            }

            numofcomH[ind] = lbmvec[0]->h[ind];// +lbmvec[0]->ZBed[ind];
            if (numofcomH[ind] < minh) minh = numofcomH[ind];
            if (numofcomH[ind] > maxh) maxh = numofcomH[ind];
            numofcomF[ind] = pow((pow(lbmvec[0]->forcex[ind], 2) +
                pow(lbmvec[0]->forcey[ind], 2)), 0.5);
            //lbmvec[0]->error1[ind];
            long x1 = x - 1, x2 = x + 1;
            long y1 = y - 1, y2 = y + 1;
            if (x1 < 0) x1 = 0;
            if (x2 >= lbmvec[0]->param->samplesx) x2 = lbmvec[0]->param->samplesx - 1;
            if (y1 < 0) y1 = 0;
            if (y2 >= lbmvec[0]->param->samplesy) y2 = lbmvec[0]->param->samplesy - 1;
            long index3 = y1 * lbmvec[0]->param->samplesx + x; //(x,y-1)
            long index4 = y2 * lbmvec[0]->param->samplesx + x; //(x,y+1)
            long index5 = y * lbmvec[0]->param->samplesx + x1; //(x-1,y)
            long index6 = y * lbmvec[0]->param->samplesx + x2; //(x+1,y)
            float uyx = (lbmvec[0]->uy[index6] - lbmvec[0]->uy[index5]) / 2.0f;//(2.0f*mlflowvec[curscale]->param->delta_x);
            float uxy = (lbmvec[0]->ux[index4] - lbmvec[0]->ux[index3]) / 2.0f;// (2.0f*mlflowvec[curscale]->param->delta_x);
            numofcomVor[ind] = fabs(uyx - uxy);
            sumHeight += (lbmvec[0]->h[ind] - 0.01);
        }
    }
    std::cout << "minh:	" << minh << std::endl;
    std::cout << "maxh:	" << maxh << std::endl;
    std::ofstream mycout("../data2D_swe/heightsum.txt", std::ios::app);
    mycout << frame << "		" << sumHeight << std::endl;

    float* vertices = new float[upw * uph * 3];
    int num = 0;
    upnum = 0;
    //#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    for (int k = 0; k < uph; k++)
    {
        for (int j = 0; j < upw; j++)
        {
            vec3 color(0, 0, 0);

            ColorRamp color_m;
            int upnum = k * upw + j;
            double vv = numofcomV[upnum] / Vmag;
            color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);

            vertices[num++] = color.x;
            vertices[num++] = color.y;
            vertices[num++] = color.z;
        }
    }
    char filename[2048];
    sprintf(filename, "../data2D_swe/ppm_ve/static%05d.ppm", frame);
    mlSavePPM(filename, vertices, upw, uph);

    //sprintf(filename, "../data_sib/ve/velocity%05d.bin", frame);
    //saveVData(filename);

    num = 0;
    upnum = 0;
    float gamma = 1 / 1.0;
    //#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    for (int k = 0; k < uph; k++)
    {
        for (int j = 0; j < upw; j++)
        {
            vec3 color(0, 0, 0);

            ColorRamp color_m;
            int upnum = k * upw + j;
            double vv = (numofcomH[upnum] - 0.01) / 6.12;
            color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);

            color.x = pow(color.x, 1.0 / gamma);
            color.y = pow(color.y, 1.0 / gamma);
            color.z = pow(color.z, 1.0 / gamma);

            vertices[num++] = color.x;
            vertices[num++] = color.y;
            vertices[num++] = color.z;
        }
    }
    sprintf(filename, "../data2D_swe/ppm_h/static%05d.ppm", frame);
    mlSavePPM(filename, vertices, upw, uph);


    num = 0;
    upnum = 0;
    //#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    for (int k = 0; k < uph; k++)
    {
        for (int j = 0; j < upw; j++)
        {
            vec3 color(0, 0, 0);

            ColorRamp color_m;
            int upnum = k * upw + j;
            double vv = (numofcomF[upnum] - 0.0) / 1e-5;
            color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);

            color.x = pow(color.x, 1.0 / gamma);
            color.y = pow(color.y, 1.0 / gamma);
            color.z = pow(color.z, 1.0 / gamma);

            vertices[num++] = color.x;
            vertices[num++] = color.y;
            vertices[num++] = color.z;
        }
    }
    sprintf(filename, "../data2D_swe/ppm_F/static%05d.ppm", frame);
    mlSavePPM(filename, vertices, upw, uph);

    //num = 0;
    //upnum = 0;
    //gamma = 1 / 2.2;
    ////#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    //for (int k = 0; k < uph; k++)
    //{
    //	for (int j = 0; j < upw; j++)
    //	{
    //		vec3 color(0, 0, 0);

    //		ColorRamp color_m;
    //		int upnum = k * upw + j;
    //		double vv = numofcomVor[upnum] / 0.1;
    //		color_m.set_GLcolor(vv, COLOR__PLASMA, color, false);

    //		color.x = pow(color.x, 1.0 / gamma);
    //		color.y = pow(color.y, 1.0 / gamma);
    //		color.z = pow(color.z, 1.0 / gamma);

    //		vertices[num++] = color.x;
    //		vertices[num++] = color.y;
    //		vertices[num++] = color.z;
    //	}
    //}
    //sprintf(filename, "../data/ppm_vor/static%05d.ppm", frame);
    //mlSavePPM(filename, vertices, upw, uph);

    delete[] numofcomV;
    delete[] numofcomF;
    delete[] numofcomH;
    delete[] vertices;
    delete[] numofcomVor;
}


#endif // !SOLVERSWE2DH
