rem working directory

set workdir=%~dp0
cd %workdir%
cd ..

Hlsl2Python -shadertoy -gl -torch -gendsl -vectorizebranch -vectorization %workdir%test.glsl

cd %workdir%

