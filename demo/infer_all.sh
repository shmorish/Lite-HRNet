#!/bin/bash


command_name=infer.py

# check conda env
conda activate hrnet

python3 ${command_name}

video_path="/home/morish/ダウンロード/nturgb+d_rgb/"

for i in $(ls ${video_path}); do
    echo ${i}
    python3 ${command_name} --video_path ${video_path}${i}
done
