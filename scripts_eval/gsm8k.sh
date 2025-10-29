#!/bin/bash
set -e
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES="0,1,2,3" # adapt

export LM_EVAL_LOGLEVEL=DEBUG
export VLLM_LOGLEVEL=INFO

# Define variables
MODEL_NAME=""
TASKS="gsm8k"                # Replace with your desired tasks
TP_SIZE=4                             # Number of GPUs for tensor parallelism
                             # Number of model replicas
DTYPE="auto"                          # Data type (e.g., auto, float16)
GPU_UTIL=0.9                          # GPU memory utilization
BATCH_SIZE="auto:4"
MAX_LEN=4096                     

# Construct model arguments
MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"
#MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL}"

# Execute lm_eval
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --num_fewshot=5 \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_NAME}/${TASKS}" \
        --wandb_args project=KD-tis-dpo,name=base_14B_gsm,job_type=eval \
        --log_samples \
        2>&1 | tee /tmp/lm_eval_debug.log