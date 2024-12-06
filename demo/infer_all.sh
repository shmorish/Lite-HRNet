#!/bin/bash

infer_file_name=infer.py

# check conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hrnet

video_path="/home/morish/nturgbd_rgb_dataset/"
result_path="/home/morish/Lite-HRNet/demo/result/"

# webhook_url="https://discord.com/api/webhooks/1311956163373301812/KpHo7xPGku2yn9-wLLR4KtKs96Iipbqz3aM_WiXhsm57d9euhGjKJkRUy6xzGlz-pwMF"
webhook_url="https://discord.com/api/webhooks/1311956421885296713/NJxNixoPBjekqN8q7lgO8U3K9JjEhSX0v7WqxxVYvo5KjEkRDfN2O2W0UZetSiPqKJ9o"

file_num=`ls ${video_path} | wc -l`
current_inference_count=`ls ${result_path} | grep json | wc -l`

success_curl=`cat <<EOS
curl --verbose -X POST ${webhook_url} -H 'Content-Type: application/json'
    --data '{"content": "推論正常終了通知 : ${file_num}個のファイルを推論しました"}'
EOS`

error_curl=`cat <<EOS
curl --verbose -X POST ${webhook_url} -H 'Content-Type: application/json'
    --data '{"content": "推論異常終了通知 : ${current_inference_count}/${file_num}個のファイルを推論しました\nエラーが発生しました"}'
EOS`

# signal handler
function signal_handler() {
    echo "error"
    eval ${error_curl}
    exit 1
}
trap signal_handler 2

for file in $(ls ${video_path}); do
    if [ -e ${result_path}${file}.json ]; then
        echo "${file}.json is already exist."
        continue
    fi
    echo ${video_path}${file}
    python3 ${infer_file_name} ${video_path}${file}
    current_inference_count=`ls ${result_path} | grep *.json | wc -l`
done
 
eval ${success_curl}
