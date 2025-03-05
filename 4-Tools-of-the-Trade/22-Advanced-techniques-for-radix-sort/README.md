## Example Code for Advanced Techniques for Radix Sort 
This implementation contains as follows: 
- Classic radix sort
- Onesweep with our extension

### Environment
- Windows 10/11

Note that CUDA SDK 12.0 or later is required on NVIDIA platform

### Build

```
premake5 vs2022
```

Note that you can optionally use --cuda for cuda environment like 'premake5 vs2022 --cuda'
Open build/GPUZen3Onesweep.sln and build/run.

### Dependencies 
- Orochi. This sample code uses https://github.com/GPUOpen-LibrariesAndSDKs/Orochi/tree/release/hip6.0_cuda12.2
    - It will have future updates on the radix sort implementation on the repository
- nlohmann/json. at https://github.com/nlohmann/json
