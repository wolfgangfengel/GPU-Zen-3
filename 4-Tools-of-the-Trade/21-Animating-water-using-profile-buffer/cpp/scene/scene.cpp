#include "scene/scene.h"

#include <cuda_runtime.h>
#include "cu/helper_cuda.h"
#include "cu/cuKernelWraper.cuh"
#include "cu/cuFunc.cuh"

#include <random>

/* \brief Linear interpolation
\param a Lower bound
\param b Higher bound
\param t Ratio
*/
real lint(real a, real b, real t) {
    return a + t * (b - a);
}
// \briefBilinear interpolation
real bilint(real q11, real q12, real q21, real q22, real xFrac, real yFrac) {
    real top = lint(q11, q12, xFrac);
    real bottom = lint(q21, q22, xFrac);
    return lint(top, bottom, yFrac);
}

#include "flowmap.cpp"
auto& flow_map = flow_data;
#include "heightmap.cpp"
auto& height_map = height_data;

Scene::~Scene()
{
#ifdef TRACK_WHERE
    std::cout << "~Scene" << std::endl;
#endif
    ReleaseCPUMemory();
    ReleaseGPUMemory();
}

void Scene::NullifyCPUPointers()
{
#ifdef TRACK_WHERE
    std::cout << "NullifyCPUPointers" << std::endl;
#endif
    h_height_ = nullptr;
    h_H_ = nullptr;

    // water
    h_hw_field_ = nullptr;  // water depth	// TOOPTIMIZE: could be optimized
    h_H_field_ = nullptr;
}

void Scene::NullifyGPUPointers()
{
#ifdef TRACK_WHERE
    std::cout << "NullifyGPUPointers" << std::endl;
#endif
    d_layered_data_ = nullptr;
    d_depth_data_ = nullptr;
    d_foam_data_ = nullptr;
    d_pb_norm_field_ = nullptr;
    d_pb_offset_field_ = nullptr;

    // water
    d_hw_field_ = nullptr;  // water depth

    d_H_field_ = nullptr;

    // Profile Buffer
    d_PBf_field_ = nullptr;
    d_PBs_field_ = nullptr;

    d_heightmap_field_ = nullptr;
    d_flowmap_field_ = nullptr;

    d_water_normal_field_ = nullptr;
}

void Scene::Configure(real* minc, real* maxc, int* N)
{
#ifdef TRACK_WHERE
    std::cout << "Configure" << std::endl;
#endif
    grid_ = Grid(minc, maxc, N);
#if 1
    InitCPUMemory();
#endif

#if 1
    InitGPUMemory();
#endif
}

void Scene::InitCPUMemory()
{
#ifdef TRACK_WHERE
    std::cout << "InitCPUMemory" << std::endl;
#endif
    if (h_height_)
        free(h_height_);
    h_height_ = (real*)malloc(sizeof(real) * grid_.total_voxels);
    memset(h_height_, 0, sizeof(real) * grid_.total_voxels);

    if (h_H_)
        free(h_H_);
    h_H_ = (real*)malloc(sizeof(real) * grid_.total_voxels);
    memset(h_H_, 0, sizeof(real) * grid_.total_voxels);

    if (h_hw_field_)
    {
        free(h_hw_field_);
    }
    h_hw_field_ = (real*)malloc(sizeof(real) * grid_.total_voxels);

    if (h_H_field_)
    {
        free(h_H_field_);
    }
    h_H_field_ = (real*)malloc(sizeof(real) * grid_.total_voxels);
}

void Scene::ReleaseCPUMemory()
{
#ifdef TRACK_WHERE
    std::cout << "ReleaseCPUMemory" << std::endl;
#endif
    if (h_height_)
        free(h_height_);
    if (h_H_)
        free(h_H_);

    if (h_hw_field_)
    {
        free(h_hw_field_);
    }

    if (h_H_field_)
    {
        free(h_H_field_);
    }

    NullifyCPUPointers();
}

// \brief Initialize GPU buffers and compute water height map, based on the selected scene. 
void Scene::InitGPUMemory()
{
#ifdef TRACK_WHERE
    std::cout << "InitGPUMemory" << std::endl;
#endif
    ReleaseGPUMemory();

    checkCudaErrors(cudaMalloc((void**)&(d_layered_data_), sizeof(real) * 4 * grid_.total_voxels));
    checkCudaErrors(cudaMalloc((void**)&(d_depth_data_), sizeof(real) * 4 * grid_.total_voxels));
    checkCudaErrors(cudaMalloc((void**)&(d_foam_data_), sizeof(real) * 4 * grid_.total_voxels));
    checkCudaErrors(cudaMalloc((void**)&(d_pb_norm_field_), sizeof(real) * 4 * 2 * PB_RESOLUTION));
    checkCudaErrors(cudaMalloc((void**)&(d_pb_offset_field_), sizeof(real) * 4 * 2 * PB_RESOLUTION));

    // allocate GPU memory
    checkCudaErrors(cudaMalloc((void**)&(d_H_field_), sizeof(real) * grid_.total_voxels));
    checkCudaErrors(cudaMalloc((void**)&(d_hw_field_), sizeof(real) * grid_.total_voxels * WAVE_DIM));

    // Profile Buffer
    checkCudaErrors(cudaMalloc((void**)&(d_PBf_field_), sizeof(real) * PB_RESOLUTION * WAVE_DIM));
    checkCudaErrors(cudaMalloc((void**)&(d_PBs_field_), sizeof(real) * PB_RESOLUTION * WAVE_DIM));

    checkCudaErrors(cudaMalloc((void**)&(d_water_normal_field_), sizeof(real) * grid_.total_voxels * 3));
    std::cout << "name: " << name_ << std::endl;
    if (name_ == "River Scene")
    {
        // design field
        real* flowmap_tempHostArray = new real[grid_.total_voxels * 2];
        real* heightmap_tempHostArray = new real[grid_.total_voxels];
        for (int i = 0; i < GRID_W; i++) {
            for (int j = 0; j < GRID_L; j++) {
                // Scale i and j down to old dimensions
                real x = (real)i / (GRID_W - 1.f) * 511.f;
                real y = (real)j / (GRID_L - 1.f) * 511.f;

                int xInt = (int)x;
                int yInt = (int)y;

                real xFrac = x - xInt;
                real yFrac = y - yInt;

                // Ensure you don't run over the boundary of the array
                int xNext = (int)std::min(real(xInt) + 1.f, 511.f);
                int yNext = (int)std::min(real(yInt) + 1.f, 511.f);

                {
                    real q11 = flow_map[(yInt * 512 + xInt) * 2];
                    real q12 = flow_map[(yInt * 512 + xInt) * 2];
                    real q21 = flow_map[(yNext * 512 + xNext) * 2];
                    real q22 = flow_map[(yNext * 512 + xNext) * 2];
                    flowmap_tempHostArray[((i + 1) * grid_.res_v[1] + (j + 1)) * 2 + 1] = bilint(q11, q12, q21, q22, xFrac, yFrac);

                    q11 = flow_map[(yInt * 512 + xInt) * 2 + 1];
                    q12 = flow_map[(yNext * 512 + xInt) * 2 + 1];
                    q21 = flow_map[(yInt * 512 + xNext) * 2 + 1];
                    q22 = flow_map[(yNext * 512 + xNext) * 2 + 1];
                    flowmap_tempHostArray[((i + 1) * grid_.res_v[1] + (j + 1)) * 2] = bilint(q11, q12, q21, q22, xFrac, yFrac);
                }

                {
                    real q11 = height_map[yInt * 512 + xInt];
                    real q12 = height_map[yNext * 512 + xInt];
                    real q21 = height_map[yInt * 512 + xNext];
                    real q22 = height_map[yNext * 512 + xNext];
                    heightmap_tempHostArray[(i + 1) * grid_.res_v[1] + (j + 1)] = bilint(q11, q12, q21, q22, xFrac, yFrac);
                }
            }
        }

        checkCudaErrors(cudaMalloc((void**)&(d_flowmap_field_), sizeof(real) * grid_.total_voxels * 2));
        checkCudaErrors(cudaMemcpy(d_flowmap_field_, flowmap_tempHostArray, sizeof(real) * grid_.total_voxels * 2, cudaMemcpyHostToDevice));
        delete[] flowmap_tempHostArray;

        checkCudaErrors(cudaMalloc((void**)&(d_heightmap_field_), sizeof(real) * grid_.total_voxels));
        checkCudaErrors(cudaMemcpy(d_heightmap_field_, heightmap_tempHostArray, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
        delete[] heightmap_tempHostArray;
    }
    else if (name_ == "Fixed Direction Scene")
    {
        real* flowmap_tempHostArray = new real[grid_.total_voxels * 2];
        real* heightmap_tempHostArray = new real[grid_.total_voxels];
        for (int i = 0; i < GRID_W; i++) {
            for (int j = 0; j < GRID_L; j++) {
                flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2] = .00f;                // i
                flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2 + 1] = .002f;             // j
                heightmap_tempHostArray[(j + 1) * grid_.res_v[0] + (i + 1)] = 1.0f;
            }
        }

        checkCudaErrors(cudaMalloc((void**)&(d_flowmap_field_), sizeof(real) * grid_.total_voxels * 2));
        checkCudaErrors(cudaMemcpy(d_flowmap_field_, flowmap_tempHostArray, sizeof(real) * grid_.total_voxels * 2, cudaMemcpyHostToDevice));
        delete[] flowmap_tempHostArray;

        checkCudaErrors(cudaMalloc((void**)&(d_heightmap_field_), sizeof(real) * grid_.total_voxels));
        checkCudaErrors(cudaMemcpy(d_heightmap_field_, heightmap_tempHostArray, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
        delete[] heightmap_tempHostArray;
    }
    else if (name_ == "Opposite Direction Scene")
    {
        real* flowmap_tempHostArray = new real[grid_.total_voxels * 2];
        real* heightmap_tempHostArray = new real[grid_.total_voxels];
        for (int i = 0; i < GRID_W; i++) {
            for (int j = 0; j < GRID_L; j++) {
                if (j < GRID_L / 2)
                    flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2 + 1] = .002f;
                else
                    flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2 + 1] = -.002f;

                flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2] = 0.0f;
                heightmap_tempHostArray[(j + 1) * grid_.res_v[0] + (i + 1)] = 1.0f;
            }
        }
        checkCudaErrors(cudaMalloc((void**)&(d_flowmap_field_), sizeof(real) * grid_.total_voxels * 2));
        checkCudaErrors(cudaMemcpy(d_flowmap_field_, flowmap_tempHostArray, sizeof(real) * grid_.total_voxels * 2, cudaMemcpyHostToDevice));
        delete[] flowmap_tempHostArray;

        checkCudaErrors(cudaMalloc((void**)&(d_heightmap_field_), sizeof(real) * grid_.total_voxels));
        checkCudaErrors(cudaMemcpy(d_heightmap_field_, heightmap_tempHostArray, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
        delete[] heightmap_tempHostArray;
    }
    else if (name_ == "Swirl Scene")
    {
        // design field
        real* flowmap_tempHostArray = new real[grid_.total_voxels * 2];
        for (int i = 0; i < grid_.total_voxels * 2; ++i)
        {
            flowmap_tempHostArray[i] = 0.f;
        }
        real* heightmap_tempHostArray = new real[grid_.total_voxels];
        for (int i = 0; i < GRID_W; i++) {
            for (int j = 0; j < GRID_L; j++) {
                real x = (real)i / (GRID_W - 1.f);
                real y = (real)j / (GRID_L - 1.f);

                real rx = x - 0.5f;
                real ry = y - 0.5f;
                real radius = sqrtf(rx * rx + ry * ry);
                //if (radius > 0.001f)
                {
                    flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2] = .002f * ry;
                    flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2 + 1] = .002f * -rx;
                
                }
                heightmap_tempHostArray[(j + 1) * grid_.res_v[0] + (i + 1)] = 1.0f;
            }
        }
        checkCudaErrors(cudaMalloc((void**)&(d_flowmap_field_), sizeof(real) * grid_.total_voxels * 2));
        checkCudaErrors(cudaMemcpy(d_flowmap_field_, flowmap_tempHostArray, sizeof(real) * grid_.total_voxels * 2, cudaMemcpyHostToDevice));
        delete[] flowmap_tempHostArray;

        checkCudaErrors(cudaMalloc((void**)&(d_heightmap_field_), sizeof(real) * grid_.total_voxels));
        checkCudaErrors(cudaMemcpy(d_heightmap_field_, heightmap_tempHostArray, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
        delete[] heightmap_tempHostArray;
    }
    else if (name_ == "Outward Scene")
    {
        // design field
        real* flowmap_tempHostArray = new real[grid_.total_voxels * 2];
        real* heightmap_tempHostArray = new real[grid_.total_voxels];
        for (int i = 0; i < GRID_W; i++) {
            for (int j = 0; j < GRID_L; j++) {
                real x = (real)i / (GRID_W - 1.f);
                real y = (real)j / (GRID_L - 1.f);

                real rx = x - 0.5f;
                real ry = y - 0.5f;
                real radius = sqrtf(rx * rx + ry * ry);
                flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2 + 1] = .002f * rx / radius;
                flowmap_tempHostArray[((j + 1) * grid_.res_v[0] + (i + 1)) * 2] = .002f * ry / radius;
                heightmap_tempHostArray[(j + 1) * grid_.res_v[0] + (i + 1)] = 1.0f;
            }
        }
        checkCudaErrors(cudaMalloc((void**)&(d_flowmap_field_), sizeof(real) * grid_.total_voxels * 2));
        checkCudaErrors(cudaMemcpy(d_flowmap_field_, flowmap_tempHostArray, sizeof(real) * grid_.total_voxels * 2, cudaMemcpyHostToDevice));
        delete[] flowmap_tempHostArray;

        checkCudaErrors(cudaMalloc((void**)&(d_heightmap_field_), sizeof(real) * grid_.total_voxels));
        checkCudaErrors(cudaMemcpy(d_heightmap_field_, heightmap_tempHostArray, sizeof(real) * grid_.total_voxels, cudaMemcpyHostToDevice));
        delete[] heightmap_tempHostArray;
    }
}

void Scene::ReleaseGPUMemory()
{
#ifdef TRACK_WHERE
    std::cout << "ReleaseGPUMemory" << std::endl;
#endif
    if (d_layered_data_)
        checkCudaErrors(cudaFree(d_layered_data_));

    if (d_depth_data_)
        checkCudaErrors(cudaFree(d_depth_data_));

    if (d_foam_data_)
        checkCudaErrors(cudaFree(d_foam_data_));

    if (d_pb_norm_field_)
        checkCudaErrors(cudaFree(d_pb_norm_field_));

    if (d_pb_offset_field_)
        checkCudaErrors(cudaFree(d_pb_offset_field_));

    // water
    if (d_hw_field_)
        checkCudaErrors(cudaFree(d_hw_field_));
    if (d_H_field_)
        checkCudaErrors(cudaFree(d_H_field_));
    NullifyGPUPointers();
}

void Scene::PrepareGridWorldPos(real* cudaWaterGridPos, real* cudaWaterGridNorm)
{
    SSWECUDA::Update_Layered_Data(d_hw_field_, d_flowmap_field_, d_heightmap_field_,
        d_H_field_, grid_, d_layered_data_, d_depth_data_, d_foam_data_, grid_.dx[0]);
    SSWECUDA::Compute_Water_Grid(d_hw_field_, d_H_field_, d_water_normal_field_,
        grid_, cudaWaterGridPos, cudaWaterGridNorm, grid_.dx[0]);
}

void Scene::PrepareFbNormFieldResource()
{
    SSWECUDA::Update_Fb_Norm_Field(d_PBs_field_, d_PBf_field_, d_pb_norm_field_);
}

void Scene::PrepareFbOffsetFieldResource()
{
    SSWECUDA::Update_Fb_Offset_Field(d_PBs_field_, d_PBf_field_, d_pb_offset_field_);
}

real* Scene::GetHeight()
{
    checkCudaErrors(cudaMemcpy(h_height_, d_hw_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyDeviceToHost));
    return h_height_;
}

real* Scene::TransferTerrainHeightToCPU()
{
    checkCudaErrors(cudaMemcpy(h_H_, d_H_field_, sizeof(real) * grid_.total_voxels, cudaMemcpyDeviceToHost));
    return h_H_;
}

real* Scene::GetTerrainHeight()
{
    return h_H_;
}

// \brief Initialize terrain and water height fields
void Scene::Initialize_Fields()
{
#ifdef TRACK_WHERE
    std::cout << "Initialize_Fields" << std::endl;
#endif
    Initialize_Solid_Field();
    Initialize_Water_Depth_Field();
}

// \brief Compute two profile buffers using two different profile scales, s_fast and s_slow, respectively.
void Scene::ProfileBufferStep()
{
#ifdef TRACK_WHERE
    std::cout << "Scene::ProfileBufferStep" << std::endl;
#endif
    SSWECUDA::Precompute_Profile_Buffer(d_PBf_field_,
        t, k_min, k_max, s_fast, L, integration_nodes,
        grid_);
    SSWECUDA::Precompute_Profile_Buffer(d_PBs_field_,
        t, k_min, k_max, s_slow, L, integration_nodes,
        grid_);
    t += dt;
}
// \brief Compute the group speed to determine time step size.
void Scene::PrecomputeGroupSpeeds()
{
#ifdef TRACK_WHERE
    std::cout << "Scene::PrecomputeGroupSpeeds" << std::endl;
#endif
    real group_speed[2];
    SSWEShareCode::Precompute_Group_Speeds(group_speed,   // output
        integration_nodes, k_min, k_max, grid_);
    gv = 3.f * group_speed[0] / group_speed[1];
}