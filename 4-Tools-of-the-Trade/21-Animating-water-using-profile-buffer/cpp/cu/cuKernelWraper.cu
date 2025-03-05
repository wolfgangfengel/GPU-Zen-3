
#include "cuKernelWraper.cuh"
#include "cuKernelLauncher.cu"

#include "helper_cuda.h"

namespace SSWECUDA
{
    /* \brief Compute profile buffer
    \param d_PB_field_: profile buffer that stores the integration results
    \param t: current time
    \param k_min: lower bound of wave number
    \param k_max: higher bound of wave number
    \param scale: profile scale
    \param L: profile buffer length
    \param integration nodes: number of integration nodes
    */
    void Precompute_Profile_Buffer(real* d_PB_field_,
        real t, real k_min, real k_max, real scale, real L, int integration_nodes,
        Grid grid)
    {
        cuPrecompute_Profile_Buffer << < (PB_RESOLUTION / 512) + 1, 512 >> >
            (d_PB_field_,  // output
                t, k_min, k_max, scale, L, integration_nodes,
                grid);
    }

    void Compute_Water_Grid(real* water_field, real* terrain_field, real* water_normal_field,
        Grid grid, real* water_grid_pos, real* water_grid_norm, const real dx)
    {
        const real scaling = real(grid.res_v[0]) * dx;
        cuCompute_Water_Grid << < (grid.total_voxels / 512) + 1, 512 >> > (
            water_field, terrain_field, water_normal_field,
            grid, water_grid_pos, water_grid_norm,
            dx, scaling);
    }

    void Update_Layered_Data(real* water_field,
        real* flowmap_field,
        real* ampla_field,
        real* terrain_field,
        Grid grid, real* layered_data, real* depth_data, real* foam_data, const real dx)
    {
        const real scaling = real(grid.res_v[0]) * dx;
        cuUpdate_Layered_Data << < (grid.total_voxels / 512) + 1, 512 >> > (
            water_field,
            flowmap_field,
            ampla_field,
            terrain_field,
            grid, layered_data, depth_data, foam_data, scaling);
    }

    void Update_Fb_Norm_Field(real* pbs_field, real* pbf_field, real* pb_norm_field)
    {
        cuUpdate_Fb_Norm_Field << < (PB_RESOLUTION * 2 / 512) + 1, 512 >> > (pbs_field, pbf_field, pb_norm_field);
    }

    void Update_Fb_Offset_Field(real* pbs_field, real* pbf_field, real* pb_offset_field)
    {
        cuUpdate_Fb_Offset_Field << < (PB_RESOLUTION * 2 / 512) + 1, 512 >> > (pbs_field, pbf_field, pb_offset_field);
    }
}