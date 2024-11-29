#!/bin/bash

infer_file_name=infer.py

# check conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hrnet

video_path="/home/morish/nturgbd_rgb_dataset/"

# # signal handler
trap 'echo "Stop infer_all.sh"; exit 1' 2

file_num=`ls ${video_path} | wc -l`

for file in $(ls ${video_path}); do
    echo ${video_path}${file}
    python3 ${infer_file_name} ${video_path}${file}
done
 
curl=`cat <<EOS
curl
    --verbose
    -X POST
    https://discord.com/api/webhooks/1311956163373301812/KpHo7xPGku2yn9-wLLR4KtKs96Iipbqz3aM_WiXhsm57d9euhGjKJkRUy6xzGlz-pwMF
    -H 'Content-Type: application/json'
    --data '{"content": "推論終了通知 : ${file_num}個のファイルを推論しました"}'
EOS`
eval ${curl}
