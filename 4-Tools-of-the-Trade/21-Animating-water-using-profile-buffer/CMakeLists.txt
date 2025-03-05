cmake_minimum_required(VERSION 3.1)

project(CuSSWE)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)
include_directories(
	"C:/Program Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v10.1/include"
	"C:/ProgramData/NVIDIA\ Corporation/CUDA\ Samples/v10.1/common/inc")
link_directories("C:/Program Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v10.1/lib/x64")

link_directories("lib/freeglut/lib/x64")
link_directories("lib/glew/lib")

set(CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA) 

#add_definitions("-std=c++14 -Wall")
add_definitions("-D_ENABLE_EXTENDED_ALIGNED_STORAGE" -DWIN32 -DEIGEN_TEST_CUDA=ON)

# Eigen.
include_directories("lib/eigen-3.3.7")

# glut.
include_directories("lib/freeglut/include")

# glew.
include_directories("lib/glew/include")

# OpenMP.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

include_directories("cpp/")
file(GLOB_RECURSE CPP_HEADER "cpp/*/*h")
file(GLOB_RECURSE HPP_HEADER "cpp/*hpp")
file(GLOB_RECURSE CPP_FILE "cpp/*/*.cpp")
file(GLOB_RECURSE CUDA_HEADER "cpp/*.cuh")
file(GLOB_RECURSE CUDA_FILE "cpp/*.cu")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
message("arch=61")

# Create the GFluid library.
add_library(CuSSWE_core STATIC ${CPP_HEADER} ${CPP_FILE} ${CUDA_FILE} ${CUDA_HEADER} ${HPP_HEADER})
if(OpenMP_CXX_FOUND)
    target_link_libraries(CuSSWE_core PUBLIC OpenMP::OpenMP_CXX)
else(OpenMP_CXX_FOUND)
    target_link_libraries(CuSSWE_core)
endif()

include_directories("render/")
file(GLOB_RECURSE CPP_HEADER_RENDER "render/*h")
file(GLOB_RECURSE HPP_HEADER_RENDER "render/*hpp")

# Main program.
add_executable(CuSSWE "render/main.cpp" "render/visualizer.cpp")

target_include_directories(CuSSWE PUBLIC ${CPP_HEADER_RENDER} ${HPP_HEADER_RENDER})

target_link_libraries(CuSSWE PUBLIC CuSSWE_core cudart cudadevrt freeglut glew32)
