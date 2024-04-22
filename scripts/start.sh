#! /bin/bash

export http_proxy=http://100.68.161.73:3128 
export https_proxy=http://100.68.161.73:3128

# for img-txt
# python ./cal_diffusion_metric.py  --cal_IS False --cal_FID False --cal_CLIP True \
#     --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
#     --real_path ./examples/imgs1 --fake_path ./examples/prompt

# for jsonl
python ./cal_diffusion_metric.py --cal_IS --cal_FID --cal_CLIP \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 --fake_flag img\
    --jsonl_path /home/lzh/code/todo/T2I-Metrics/msr-vtt_frame_speed.jsonl
