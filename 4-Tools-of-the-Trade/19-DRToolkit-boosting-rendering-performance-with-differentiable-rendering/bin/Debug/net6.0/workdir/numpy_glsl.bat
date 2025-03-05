rem working directory

set workdir=%~dp0
cd %workdir%
cd ..

Hlsl2Python -shadertoy -ngl -notorch -gendsl -vectorizebranch -vectorization %workdir%test.glsl

cd %workdir%

