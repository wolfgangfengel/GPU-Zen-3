rem working directory

set workdir=%~dp0
cd %workdir%
cd ..

Hlsl2Python -gendsl -ngl -notorch -vectorizebranch -vectorization %workdir%hlsl_test.hlsl

cd %workdir%

