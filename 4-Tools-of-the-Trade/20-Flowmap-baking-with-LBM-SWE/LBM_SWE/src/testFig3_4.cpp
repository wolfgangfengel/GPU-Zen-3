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

    float L = 10;
    int nx = 512;
    int ny = 512;

    float gx = 0.0;
    float gy = 9.81 * 1;
    lbmvec[0]->mlCreate(0, 0, nx, ny, 1,
        nx, ny, 1e-4, L, gy);
    lbmvec[0]->param->MannC = 0.02;
    solver2d.isboundary = 1;
    solver2d.hin = 3.5f;
    solver2d.uxinit = 0.0f;
    solver2d.uyinit = 0.02f;
    solver2d.MapRate = 0.5f;
    solver2d.Vmag = 0.082;

    solver2d.AttachLbmvecHost(lbmvec);
    solver2d.AttachLbmDevice(lbmvec_gpu);
    solver2d.L = L;


    char filename[2048];
    sprintf(filename, "../canyon_heightmap/old_method_terrain_higher.png");
    solver2d.mlInitFig_3_4(filename);
    solver2d.mlTransData2Gpu();
    int numofiteration = 0;
    int numofframe = 0;

  
    solver2d.addforce = true;
    solver2d.changeInlet = true;
    while (numofframe < 800)
    {
        for (int time = 0; time < 300; time++)
        {
            solver2d.mlIterateGpu();
        }
        {
            solver2d.mlTransData2Host(0);
            solver2d.mlResamplCutSlice(0, 0, 0, numofframe++);
        }
        std::cout << "numofiteration: " << numofiteration++ << std::endl;
    }


    return 0;
}