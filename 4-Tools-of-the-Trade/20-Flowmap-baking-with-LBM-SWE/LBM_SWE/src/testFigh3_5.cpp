#include "../2d/Solver_swe_2D.h"

int main()
{
    Solver_swe_2D solver2d;
    mlFlow2D* mlsmoke0 = new mlFlow2D();
    mlFlow2D* mlsmoke_dev0 = 0;
    std::vector<mlFlow2D* > lbmvec;
    std::vector<mlFlow2D*> lbmvec_gpu;
    lbmvec.push_back(mlsmoke0);
    lbmvec_gpu.push_back(mlsmoke_dev0);

    int scaletime = 1;
    float L = 15;
    int nx = 505 * scaletime;
    int ny = 505 * scaletime;

    float gx = 0.0;
    float gy = 9.81 * 1;
    lbmvec[0]->mlCreate(0, 0, nx, ny, 1,
        nx, ny, 1e-6 * 1, L, gy);
    lbmvec[0]->param->MannC = 0.02;
    std::cout << "gy:	" << lbmvec[0]->param->gy << std::endl;
    solver2d.AttachLbmvecHost(lbmvec);
    solver2d.AttachLbmDevice(lbmvec_gpu);
    solver2d.L = L;
    solver2d.isboundary = 1;
    solver2d.hin = 5.01f;
    solver2d.hout = 3.01f;
    solver2d.uxinit = 0.1f;
    solver2d.uyinit = 0.0f;
    solver2d.MapRate = 1.0f;
    solver2d.Vmag = 0.15;

    char filename[2048];
    sprintf(filename, "../canyon_heightmap/RiverTestHeightmap.png");
    solver2d.mlInitFig_3_5(filename);
    solver2d.mlTransData2Gpu();
    int numofiteration = 0;
    int numofframe = 0;

    
    solver2d.changeInlet = true;
    while (numofframe < 800)
    {
        for (int time = 0; time < 300 * scaletime; time++)
        {
            solver2d.mlIterateGpu();
        }
    
        {
            solver2d.mlTransData2Host(0);
            solver2d.mlResamplCutSlice(0, 0, 0, numofframe++);
        }
        if (numofframe == 20)
        {
            solver2d.addforce = true;
            solver2d.w1 = 0.05;
       
        }

        std::cout << "numofiteration: " << numofiteration++ << std::endl;
    }


    return 0;
}