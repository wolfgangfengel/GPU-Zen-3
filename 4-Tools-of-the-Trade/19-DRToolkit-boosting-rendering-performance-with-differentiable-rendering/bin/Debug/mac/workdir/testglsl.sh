#!/bin/bash

cur_dir=$(pwd)
script_dir=$(cd $(dirname $0); pwd)
echo "cur dir:" $cur_dir "script dir:" $script_dir

cd $script_dir
cd ..

./Hlsl2PythonMac -shadertoy -ngl -notorch -gendsl -vectorizebranch -vectorization workdir/test.glsl

cd $cur_dir

