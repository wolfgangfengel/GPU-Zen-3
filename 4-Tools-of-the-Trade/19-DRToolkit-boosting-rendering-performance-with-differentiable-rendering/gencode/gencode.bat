rem working directory

set workdir=%~dp0
cd %workdir%

BatchCommand.exe gen_glsl_h.dsl > %workdir%../bin/Debug/net6.0/shaderlib/glsl_autogen.h
BatchCommand.exe gen_hlsl_lib_numpy_swizzle.dsl > %workdir%../bin/Debug/net6.0/shaderlib/hlsl_lib_numpy_swizzle.py
BatchCommand.exe gen_hlsl_lib_torch_swizzle.dsl > %workdir%../bin/Debug/net6.0/shaderlib/hlsl_lib_torch_swizzle.py

BatchCommand.exe gen_glsl_h.dsl > %workdir%../bin/Debug/mac/shaderlib/glsl_autogen.h
BatchCommand.exe gen_hlsl_lib_numpy_swizzle.dsl > %workdir%../bin/Debug/mac/shaderlib/hlsl_lib_numpy_swizzle.py
BatchCommand.exe gen_hlsl_lib_torch_swizzle.dsl > %workdir%../bin/Debug/mac/shaderlib/hlsl_lib_torch_swizzle.py
