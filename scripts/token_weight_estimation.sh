#!/bin/zsh

cd ..

model_name_1=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B
model_name_2=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B
input_dir="/home/hungpv/projects/DSKD-DPO/datasets/data100"
output_dir="/home/hungpv/projects/DSKD-DPO/generated-data/ultra-feedback-tisdpo"
model1_template="normal"
model2_template="normal"
batch_size=16
num_gpus=1
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python /home/hungpv/projects/DSKD-DPO/token_weight_estimation.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 