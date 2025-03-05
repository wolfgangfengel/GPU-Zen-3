#!/bin/bash

cur_dir=$(pwd)
script_dir=$(cd $(dirname $0); pwd)
echo "cur dir:" $cur_dir "script dir:" $script_dir

cd $script_dir

cp Hlsl2PythonMac.* ../mac/
cp -rf Hlsl2PythonMac ../mac/
cp Dsl.* ../mac/

cd $cur_dir
