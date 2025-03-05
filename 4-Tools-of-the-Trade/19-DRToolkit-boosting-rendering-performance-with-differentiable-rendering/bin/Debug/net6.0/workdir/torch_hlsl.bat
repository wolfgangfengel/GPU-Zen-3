rem working directory

set workdir=%~dp0
cd %workdir%
cd ..

Hlsl2Python -gendsl -gl -torch -vectorizebranch -vectorization %workdir%hlsl_test.hlsl

cd %workdir%

