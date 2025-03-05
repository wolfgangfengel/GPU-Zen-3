#pragma once
#ifndef _MLTPINIT2DH_
#define _MLTPINIT2DH_

#include "../../flow/lwflow_swe2D.h"
#include "../UtilFuncCpu2D.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../3rdParty/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../3rdParty/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../3rdParty/stb_image_resize.h"

#include <algorithm>
#include <omp.h>
#include <vector>
class mlInitHandler2D
{
public:
    void loadHeightMap(const char* filename, float rate);
    void mlInitCascadeCpu_Fig3_4(std::vector<mlFlow2D*>  mlflowvec, int scale, float L);
    void mlInitCascadeBoundaryCpu_Fig3_4(std::vector<mlFlow2D*>  mlflowvec, int scale, float L);

    void mlInitCascadeCpu_Fig3_5(std::vector<mlFlow2D*>  mlflowvec, int scale, float L);
    void mlInitCascadeBoundaryCpu_Fig3_5(std::vector<mlFlow2D*>  mlflowvec, int scale, float L);

    void AttachMapping(MLMappingParam& mapping);
    MLMappingParam mparam;
    float uyPhy = 0.4;
    float R;
    float r;
    float Scale_length;
    unsigned char* hmap;
    int ow, oh, n;
    int iw, ih;
    int scaletime = 1;

};

inline void mlInitHandler2D::loadHeightMap(const char* filename, float rate)
{
    unsigned char* idata = stbi_load(filename, &iw, &ih, &n, 0);
    ow = iw * rate;
    oh = ih * rate;
    hmap = (unsigned char*)malloc(ow * oh * n);

    // ¸Ä±äÍ¼Æ¬³ß´ç
    stbir_resize(idata, iw, ih, 0, hmap, ow, oh, 0, STBIR_TYPE_UINT8, n, STBIR_ALPHA_CHANNEL_NONE, 0,
        STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
        STBIR_FILTER_BOX, STBIR_FILTER_BOX,
        STBIR_COLORSPACE_SRGB, nullptr
    );

}

void mlInitHandler2D::mlInitCascadeCpu_Fig3_4(std::vector<mlFlow2D*> mlflowvec, int scale, float L)
{
    int Nx = mlflowvec[scale]->param->samplesx;
    int Ny = mlflowvec[scale]->param->samplesy;
    float H = 2;
    Scale_length = mlflowvec[scale]->param->Scale_length;
    R = 2.5 / Scale_length * 4 * 2;
    REAL g = mlflowvec[scale]->param->gy;
    mlEqStateCpu2f mleqstate;
    //#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    for (long y = 0; y < mlflowvec[scale]->param->samplesy; y++)
    {
        for (long x = 0; x < mlflowvec[scale]->param->samplesx; x++)
        {
            int curind = y * mlflowvec[scale]->param->samplesx + x;
            //figure 3.4 setting
            mlflowvec[scale]->h[curind] = 0.01;
            mlflowvec[scale]->ux[curind] = 0.0;
            mlflowvec[scale]->uy[curind] = 0.0;
            mlflowvec[scale]->ZBed[curind] = (int)hmap[curind * n + 0] / 255.0 * 8 + 0.02;

            if (mlflowvec[scale]->flag[curind] == ML_INLET)
            {
                H = mlflowvec[scale]->ZBed[curind];
            }
            H = mlflowvec[scale]->ZBed[curind];
            if (mlflowvec[scale]->flag[curind] == ML_SOILD)
            {
                mlflowvec[scale]->ux[curind] = 0.0;
            }
            float mprate = 5.0 / Nx;
            mlLatticeNodeD2Q9f f_curind;
            mleqstate.mlComputeEqstate(mlflowvec[scale]->ux[curind], mlflowvec[scale]->uy[curind],
                mlflowvec[scale]->h[curind], g, f_curind);
            for (int i = 0; i < 9; i++)
            {
                int f_ind = i * mlflowvec[scale]->param->samplesx * mlflowvec[scale]->param->samplesy + curind;
                mlflowvec[scale]->f[f_ind] = f_curind.f[i];
                mlflowvec[scale]->fPost[f_ind] = f_curind.f[i];
            }

            //figure 3.5 setting
           /* mlflowvec[scale]->h[curind] = 0.01;
            mlflowvec[scale]->ux[curind] = 0.0;
            mlflowvec[scale]->uy[curind] = 0.0;
            mlflowvec[scale]->ZBed[curind] = fabs((int)hmap[curind * n + 0]) / 255.0 * 8.0 + 0.01;

            if (mlflowvec[scale]->flag[curind] == ML_INLET)
            {
                H = mlflowvec[scale]->ZBed[curind];
            }
            H = mlflowvec[scale]->ZBed[curind];
            if (H > 4.01) mlflowvec[scale]->flag[curind] = ML_SOILD;
            if (mlflowvec[scale]->flag[curind] == ML_SOILD)
            {
                mlflowvec[scale]->ux[curind] = 0.0;
            }
            float mprate = 5.0 / Nx;
            mlLatticeNodeD2Q9f f_curind;
            mleqstate.mlComputeEqstate(mlflowvec[scale]->ux[curind], mlflowvec[scale]->uy[curind],
                mlflowvec[scale]->h[curind], g, f_curind);
            for (int i = 0; i < 9; i++)
            {
                int f_ind = i * mlflowvec[scale]->param->samplesx * mlflowvec[scale]->param->samplesy + curind;
                mlflowvec[scale]->f[f_ind] = f_curind.f[i];
                mlflowvec[scale]->fPost[f_ind] = f_curind.f[i];
            }*/
        }
    }
}
inline void mlInitHandler2D::mlInitCascadeBoundaryCpu_Fig3_4(std::vector<mlFlow2D*> mlflowvec, int scale, float L)
{
    int Nx = mlflowvec[scale]->param->samplesx;
    int Ny = mlflowvec[scale]->param->samplesy;
#pragma omp parallel  for  
    for (long y = 0; y < mlflowvec[scale]->param->samplesy; y++)
    {
        for (long x = 0; x < mlflowvec[scale]->param->samplesx; x++)
        {
            int curind = y * mlflowvec[scale]->param->samplesx + x;

            if (mlflowvec[scale]->flag[curind] != ML_SOILD && mlflowvec[scale]->flag[curind] != ML_INVALID)
            {
                //figure 3.4 setting
                float _cellsize = mlflowvec[scale]->param->delta_x;
                if (y == 0)
                    mlflowvec[scale]->flag[curind] = ML_WALL_DOWN;
                if (x == 0)
                    mlflowvec[scale]->flag[curind] = ML_WALL_LEFT;
                if (x == Nx - 1)
                    mlflowvec[scale]->flag[curind] = ML_WALL_RIGHT;
                if (y == Ny - 1)
                    mlflowvec[scale]->flag[curind] = ML_WALL_UP;
                float g_h = L / (float)mlflowvec[scale]->param->domian_sizex;
                float x_L = g_h * x;
                float y_L = g_h * y;
                float X = g_h * (float)mlflowvec[scale]->param->domian_sizex;
                float Y = g_h * (float)mlflowvec[scale]->param->domian_sizey;
                float inlet_posX = 0.5 * L;
                float inlet_posY = 0.5 * L;
                float inlet_posZ = 0.5 * L;
                float inlet_posr = 0.02 * L;// 0.3*L;+ 0.5*g_h
                float inlet_posR = 0.3 * L;//0.05*L+ 0.5*g_h ;
                if (
                    x > 60 && x < 80 && y == 0
                    )
                {
                    mlflowvec[scale]->flag[curind] = ML_INLET;
                }

                //figure 3.5 setting
                //if (y == 0)
                //    mlflowvec[scale]->flag[curind] = ML_WALL_DOWN;
                //if (x <= 38)
                //    mlflowvec[scale]->flag[curind] = ML_WALL_LEFT;
                //if (x == Nx - 1)
                //    mlflowvec[scale]->flag[curind] = ML_WALL_RIGHT;
                //if (y == Ny - 1)
                //    mlflowvec[scale]->flag[curind] = ML_WALL_UP;
                //if (
                //    y > 330 * scaletime && y < 370 * scaletime && x == 38 * scaletime
                //    )
                //{
                //    mlflowvec[scale]->flag[curind] = ML_INLET;
                //}
                //if (
                //    //x == Nx - 1
                //    pow(y - 210 * scaletime, 2) + pow(x - 405 * scaletime, 2) < 15 * scaletime * 15 * scaletime
                //    )
                //{
                //    mlflowvec[scale]->flag[curind] = ML_OUTLET;
                //}


            }
        }
    }
}
inline void mlInitHandler2D::mlInitCascadeCpu_Fig3_5(std::vector<mlFlow2D*> mlflowvec, int scale, float L)
{
    int Nx = mlflowvec[scale]->param->samplesx;
    int Ny = mlflowvec[scale]->param->samplesy;
    float H = 2;
    Scale_length = mlflowvec[scale]->param->Scale_length;
    R = 2.5 / Scale_length * 4 * 2;
    REAL g = mlflowvec[scale]->param->gy;
    mlEqStateCpu2f mleqstate;
    //#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
    for (long y = 0; y < mlflowvec[scale]->param->samplesy; y++)
    {
        for (long x = 0; x < mlflowvec[scale]->param->samplesx; x++)
        {
            int curind = y * mlflowvec[scale]->param->samplesx + x;

            //figure 3.5 setting
            mlflowvec[scale]->h[curind] = 0.01;
            mlflowvec[scale]->ux[curind] = 0.0;
            mlflowvec[scale]->uy[curind] = 0.0;
            mlflowvec[scale]->ZBed[curind] = fabs((int)hmap[curind * n + 0]) / 255.0 * 8.0 + 0.01;

            if (mlflowvec[scale]->flag[curind] == ML_INLET)
            {
                H = mlflowvec[scale]->ZBed[curind];
            }
            H = mlflowvec[scale]->ZBed[curind];
            if (H > 4.01) mlflowvec[scale]->flag[curind] = ML_SOILD;
            if (mlflowvec[scale]->flag[curind] == ML_SOILD)
            {
                mlflowvec[scale]->ux[curind] = 0.0;
            }
            float mprate = 5.0 / Nx;
            mlLatticeNodeD2Q9f f_curind;
            mleqstate.mlComputeEqstate(mlflowvec[scale]->ux[curind], mlflowvec[scale]->uy[curind],
                mlflowvec[scale]->h[curind], g, f_curind);
            for (int i = 0; i < 9; i++)
            {
                int f_ind = i * mlflowvec[scale]->param->samplesx * mlflowvec[scale]->param->samplesy + curind;
                mlflowvec[scale]->f[f_ind] = f_curind.f[i];
                mlflowvec[scale]->fPost[f_ind] = f_curind.f[i];
            }
        }
    }
}
inline void mlInitHandler2D::mlInitCascadeBoundaryCpu_Fig3_5(std::vector<mlFlow2D*> mlflowvec, int scale, float L)
{
    int Nx = mlflowvec[scale]->param->samplesx;
    int Ny = mlflowvec[scale]->param->samplesy;
#pragma omp parallel  for  
    for (long y = 0; y < mlflowvec[scale]->param->samplesy; y++)
    {
        for (long x = 0; x < mlflowvec[scale]->param->samplesx; x++)
        {
            int curind = y * mlflowvec[scale]->param->samplesx + x;

            if (mlflowvec[scale]->flag[curind] != ML_SOILD && mlflowvec[scale]->flag[curind] != ML_INVALID)
            {
                //figure 3.5 setting
                if (y == 0)
                    mlflowvec[scale]->flag[curind] = ML_WALL_DOWN;
                if (x <= 38)
                    mlflowvec[scale]->flag[curind] = ML_WALL_LEFT;
                if (x == Nx - 1)
                    mlflowvec[scale]->flag[curind] = ML_WALL_RIGHT;
                if (y == Ny - 1)
                    mlflowvec[scale]->flag[curind] = ML_WALL_UP;
                if (
                    y > 330 * scaletime && y < 370 * scaletime && x == 38 * scaletime
                    )
                {
                    mlflowvec[scale]->flag[curind] = ML_INLET;
                }
                if (
                    //x == Nx - 1
                    pow(y - 210 * scaletime, 2) + pow(x - 405 * scaletime, 2) < 15 * scaletime * 15 * scaletime
                    )
                {
                    mlflowvec[scale]->flag[curind] = ML_OUTLET;
                }
            }
        }
    }
}
inline void mlInitHandler2D::AttachMapping(MLMappingParam& mapping)
{
    this->mparam.l0p = mapping.l0p;
    this->mparam.N = mapping.N;

    this->mparam.u0p = mapping.u0p;
    this->mparam.labma = mapping.labma;

    this->mparam.tp = mapping.tp;
    this->mparam.roup = mapping.roup;
}
#endif // !_MLINIT2DH_
