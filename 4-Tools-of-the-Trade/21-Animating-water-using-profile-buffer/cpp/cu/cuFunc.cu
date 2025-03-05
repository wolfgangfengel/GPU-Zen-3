
#include "cuFunc.cuh"
#include "cuUtils.cuh"

#include <cstdio>

namespace SSWECUDA
{
    __global__ void cuCompute_Water_Grid(real* water_field, real* terrain_field, real* water_normal_field,
        Grid grid, real* water_grid_pos, real* water_grid_norm,
        const real dx, const real scaling)
    {
        int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx1D >= grid.total_voxels)
            return;

        int index[2] = { 0, 0 };
        grid.Voxel_Unflatten_Index(idx1D, index);

        water_grid_pos[idx1D * 3] = (index[0]) / float(grid.res_v[0]);
        water_grid_pos[idx1D * 3 + 2] = (index[1]) / float(grid.res_v[1]);
        if (index[0] == 0 || index[1] == 0 || index[0] == grid.res_v[0] - 1 || index[1] == grid.res_v[1] - 1 ||
            index[0] == 1 || index[1] == 1 || index[0] == int(grid.res_v[0]) - 2 || index[1] == int(grid.res_v[1]) - 2)
            water_grid_pos[idx1D * 3 + 1] = (real)0.f;
        else
            water_grid_pos[idx1D * 3 + 1] =  (water_field[idx1D] + terrain_field[idx1D]) / scaling - 1e-4f;


        int indexX[2] = { index[0] + 1, index[1] };
        int indexY[2] = { index[0], index[1] + 1 };

        int idx1DX = grid.Voxel_Flatten_ID(indexX);
        int idx1DY = grid.Voxel_Flatten_ID(indexY);

        water_grid_norm[idx1D * 3] = water_normal_field[idx1D];
        water_grid_norm[idx1D * 3 + 1] = water_normal_field[idx1D + grid.total_voxels];
        water_grid_norm[idx1D * 3 + 2] = water_normal_field[idx1D + grid.total_voxels * 2];

        if (index[0] == 0 || index[1] == 0 || index[0] == grid.res_v[0] - 1 || index[1] == grid.res_v[1] - 1)
        {
            water_grid_norm[idx1D * 3 + 0] = 0;
            water_grid_norm[idx1D * 3 + 1] = 1;
            water_grid_norm[idx1D * 3 + 2] = 0;
        }
    }

    __global__ void cuUpdate_Layered_Data(real* water_field,
        real* flowmap_field,
        real* ampla_field,
        real* terrain_field,
        Grid grid, real* layered_data, real* depth_data, real* foam_data, const real scaling)
    {
        int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx1D >= grid.total_voxels)
            return;

        int index[2] = { 0, 0 };

        index[1] = idx1D / grid.res_v[0];
        index[0] = idx1D - index[1] * grid.res_v[0];

        int newidx1D = grid.Voxel_Flatten_ID(index);

        if (index[0] == 0 || index[1] == 0 ||
            index[0] == grid.res_v[0] - 1 ||
            index[1] == grid.res_v[1] - 1)
        {
            layered_data[newidx1D * 4 + 0] = 0.0f;
            layered_data[newidx1D * 4 + 1] = 0.0f;
            layered_data[newidx1D * 4 + 2] = 0.5f;
            layered_data[newidx1D * 4 + 3] = 0.5f;
            depth_data[newidx1D * 4 + 0] = (18.0f - terrain_field[idx1D]) / scaling;
            foam_data[newidx1D * 4 + 0] = 0.0f;
        }
        else
        {
            // along i
            float vx = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0])];
            // along j
            float vz = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0]) + 1];

            float dangle = -1.f;
            float angle = atan2(vx, vz);
            if (angle < 0.f) angle += TAU;


            float vx_nl = flowmap_field[2 * (grid.res_v[0] * (index[1] - 1) + index[0])];
            float vz_nl = flowmap_field[2 * (grid.res_v[0] * (index[1] - 1) + index[0]) + 1];
            float anglel = atan2(vx_nl, vz_nl);
            if (anglel < 0.f) anglel += TAU;
            dangle = fmaxf(dangle, fminf(fabsf(anglel - angle), TAU - fabsf(anglel - angle)));


            float vx_nr = flowmap_field[2 * (grid.res_v[0] * (index[1] + 1) + index[0])];
            float vz_nr = flowmap_field[2 * (grid.res_v[0] * (index[1] + 1) + index[0]) + 1];
            float angler = atan2(vx_nr, vz_nr);
            if (angler < 0.f) angler += TAU;
            dangle = fmaxf(dangle, fminf(fabsf(angler - angle), TAU - fabsf(angler - angle)));


            float vx_nb = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0] - 1)];
            float vz_nb = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0] - 1) + 1];
            float angleb = atan2(vx_nb, vz_nb);
            if (angleb < 0.f) angleb += TAU;
            dangle = fmaxf(dangle, fminf(fabsf(angleb - angle), TAU - fabsf(angleb - angle)));


            float vx_nt = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0] + 1)];
            float vz_nt = flowmap_field[2 * (grid.res_v[0] * index[1] + index[0] + 1) + 1];
            float anglet = atan2(vx_nt, vz_nt);
            if (anglet < 0.f) anglet += TAU;
            dangle = fmaxf(dangle, fminf(fabsf(anglet - angle), TAU - fabsf(anglet - angle)));

            layered_data[newidx1D * 4 + 0] = ampla_field[grid.res_v[0] * index[1] + index[0]];
            layered_data[newidx1D * 4 + 1] = dangle;
            layered_data[newidx1D * 4 + 2] = vx * 0.5f + 0.5f;
            layered_data[newidx1D * 4 + 3] = vz * 0.5f + 0.5f;

            depth_data[newidx1D * 4 + 0] = (18.0f - terrain_field[idx1D]) / scaling;
            foam_data[newidx1D * 4 + 0] = sqrt(vx * vx + vz * vz) * 20.0f;
        }
    }

    __global__ void cuUpdate_Fb_Norm_Field(real* pbs_field, real* pbf_field, real* pb_norm_field)
    {
        int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx1D >= PB_RESOLUTION * 2)
            return;

        int index[2] = { 0, 0 };

        index[1] = idx1D / PB_RESOLUTION;
        index[0] = (idx1D - index[1] * PB_RESOLUTION);

        real pbs2 = pbs_field[index[0] + 2 * PB_RESOLUTION];
        real pbs3 = pbs_field[index[0] + 3 * PB_RESOLUTION];
        real pbf2 = pbf_field[index[0] + 2 * PB_RESOLUTION];
        real pbf3 = pbf_field[index[0] + 3 * PB_RESOLUTION];

        pb_norm_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 0] = (pbs2 * 0.1f + 1.0f) * 0.5f;
        pb_norm_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 1] = (pbs3 * 0.1f + 1.0f) * 0.5f;
        pb_norm_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 2] = (pbf2 * 0.1f + 1.0f) * 0.5f;
        pb_norm_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 3] = (pbf3 * 0.1f + 1.0f) * 0.5f;
    }

    __global__ void cuUpdate_Fb_Offset_Field(real* pbs_field, real* pbf_field, real* pb_offset_field)
    {
        int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx1D >= PB_RESOLUTION * 2)
            return;

        int index[2] = { 0, 0 };

        index[1] = idx1D / PB_RESOLUTION;
        index[0] = (idx1D - index[1] * PB_RESOLUTION);

        real pbs0 = pbs_field[index[0] + 0 * PB_RESOLUTION];
        real pbs1 = pbs_field[index[0] + 1 * PB_RESOLUTION];
        real pbf0 = pbf_field[index[0] + 0 * PB_RESOLUTION];
        real pbf1 = pbf_field[index[0] + 1 * PB_RESOLUTION];

        pb_offset_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 0] = (pbs0 * 0.1f + 1.0f) * 0.5f;
        pb_offset_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 1] = (pbs1 * 0.1f + 1.0f) * 0.5f;
        pb_offset_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 2] = (pbf0 * 0.1f + 1.0f) * 0.5f;
        pb_offset_field[(index[1] * PB_RESOLUTION + index[0]) * 4 + 3] = (pbf1 * 0.1f + 1.0f) * 0.5f;
    }

    __global__ void cuPrecompute_Profile_Buffer(real* PB_field,
        real t, real k_min, real k_max, real scale, real L, int integration_nodes,
        Grid grid)
    {
        int idx1D = blockIdx.x * blockDim.x + threadIdx.x; // global
        if (idx1D >= PB_RESOLUTION)
            return;

        SSWEShareCode::Precompute_Profile_Buffer(PB_field,   // output
            t, k_min, k_max, scale, L, integration_nodes, grid, idx1D);
    }
}