#pragma once
#include "common/options.h"
#include "common/Defines.h"

#include "config_type.h"
#include "common/SSWE.ShareCode.h"

// Linear interpolation
real lint(real a, real b, real t);
// Bilinear interpolation
real bilint(real q11, real q12, real q21, real q22, real xFrac, real yFrac);

class Scene {
public:
	using VectorD = Eigen::Matrix<real, 3, 1>;
	using VectorI = Eigen::Matrix<int, 3, 1>;

	Scene()
	{
		NullifyCPUPointers();
		NullifyGPUPointers();
	}

	Scene(const std::string& name) : name_(name), options_()
	{
		NullifyCPUPointers();
		NullifyGPUPointers();
	}

	~Scene();

	virtual void Initialize() {}

	const std::string name() const { return name_; }
	const Options& options() const { return options_; }

	void Configure(real* minc, real* maxc, int* N);

	void PrecomputeGroupSpeeds();
	void ProfileBufferStep();

	real GetPeriodicity() { return L; }
	real GetAmpla() { return ampla; }


	// CPU functions
	void ReleaseCPUMemory();

	// GPU functions. 
	void InitGPUMemory();
	void ReleaseGPUMemory();

	void TransferCudaToGrid();

	real* TransferTerrainHeightToCPU();

	// interop
	void PrepareGridWorldPos(real* cudaWaterGridPos, real* cudaWaterGridNorm);

	void PrepareFbNormFieldResource();
	void PrepareFbOffsetFieldResource();

	real* GetCudaLayeredData() { return d_layered_data_; }
	real* GetCudaDepthData() { return d_depth_data_; }
	real* GetCudaFoamData() { return d_foam_data_; }
	real* GetCudaPbNormField() { return d_pb_norm_field_; }
	real* GetCudaPbOffsetField() { return d_pb_offset_field_; }

	VectorXI GetGridRes() { return options_.GetVectorIOptions("gridRes"); }
	void	 GetGridRes(int* rtnRes) { rtnRes[0] = grid_.res_v[0]; rtnRes[1] = grid_.res_v[1]; }
	real* GetHeight();
	real* GetTerrainHeight();
	Grid	 GetGrid() { return grid_; }

	// Simulation functions
	void Initialize_Fields();
	virtual void Initialize_Solid_Field() {}
	virtual void Initialize_Water_Depth_Field() {}
	virtual void Initialize_Water_Depth_Field_With_Divide() {}
protected:

	std::string			name_;
	Options				options_;

	// Grid
	Grid	grid_;

	//*********************
	//* Sim Config
	//*********************
	real dt;

	//*********************
	//* Water Field
	//*********************
	real hw_init;

	//*********************
	//* Solid Field
	//*********************
	real H_init;
	real dH;

	//*************************************************************************************
	//* CPU Variables
	//*************************************************************************************
	real* h_height_;	// output to display
	real* h_H_;		// output to display

	// water
	real* h_hw_field_; 	// water depth	
	real* h_H_field_;		// terrain height
	//*************************************************************************************
	//* GPU Variables
	//*************************************************************************************

	real* d_layered_data_;
	real* d_depth_data_;
	real* d_foam_data_;
	real* d_pb_norm_field_;
	real* d_pb_offset_field_;

	real* d_hw_field_; 	// water depth	
	real* d_H_field_;		// terrain height

	int currentBufferID = 0;
	int cur_frame = 0;
	int substeps = 0;

	//*************************************************************************************
	//* Profile Buffer
	//*************************************************************************************
	real* d_heightmap_field_;	// 32
	real* d_PBf_field_;			// 4
	real* d_PBs_field_;			// 4
	real* d_flowmap_field_;
	real* d_water_normal_field_;

	real t;
	real gv;

	real L;
	int integration_nodes = 100;
	int N;
	real k_min;
	real k_max;
	real s_fast;
	real s_slow;
	real ampla;

private:
	void NullifyGPUPointers();
	void NullifyCPUPointers();
	void InitCPUMemory();
};
