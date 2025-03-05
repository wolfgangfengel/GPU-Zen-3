#ifndef __SYSTEM_CUDA_DEVICE_H_
#define __SYSTEM_CUDA_DEVICE_H_

#include <string>
#include <unordered_map>
#include <driver_types.h>

#include <cuda_runtime.h>

#include "Singleton.h"
#include "CudaExecutionPolicy.h"


using KernelFunc = const void*;

struct KernelConfig {       ///< static kernel attrib, could contain run-time debugger setting(error checking/ time recording etc...)
    KernelFunc      func;
    cudaFuncAttributes  attribs;
    cudaFuncCache   cachePreference;
    bool            waveFashion;        ///< general fashion or loop fashion
    int             maxOccBlockSize;    ///< condition: use no shared memory
    explicit KernelConfig(KernelFunc f = nullptr, cudaFuncCache cacheConfig = cudaFuncCachePreferNone, bool isWave = false);
};

class CudaDevice : public ManagedSingleton<CudaDevice> {
public:
    CudaDevice();
    ~CudaDevice();

    static void registerKernel(std::string tag, KernelFunc f, cudaFuncCache cacheConfig = cudaFuncCachePreferL1, bool waveFashion = true);
    static const KernelConfig& findKernel(std::string name);

    int generalGridSize(int& threadNum, int& blockSize) const;
    int waveGridSize(int& threadNum, int& blockSize) const;
    static int evalOptimalBlockSize(cudaFuncAttributes attribs, cudaFuncCache cachePreference, size_t smemBytes = 0);
    ExecutionPolicy launchConfig(std::string kernelName, int threadNum, bool sync = false, size_t smemSize = 0, cudaStream_t sid = nullptr) const;

    static void reportMemory(std::string msg) {
        size_t free_byte;
        size_t total_byte;
        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

        if (cudaSuccess != cuda_status) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
            exit(1);
        }

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;
        printf("GPU memory usage (%s): used = %f MB, free = %f MB, total = %f MB\n",
            msg.data(), used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
    }

    /// Launching Kernels
private:
    static cudaDeviceProp*  _akDeviceProps;
    static int              _iDevID;    ///< selected cuda device
    static std::unordered_map<std::string, KernelConfig>
        _kFuncTable;
};

#endif
