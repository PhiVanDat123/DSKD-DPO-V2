#!/bin/bash
set -e

cd ..


python3 -u /home/hungpv/projects/DSKD-DPO/train.py \
  model=KD_tisdpo \
  model.policy_name_or_path=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B \
  model.reference_name_or_path=/home/hungpv/projects/DSKD-DPO/models/Qwen/Qwen2.5-7B \
  model.teacher_tokenizer_name_or_path=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B \
  model.student_tokenizer_name_or_path=/home/hungpv/projects/DSKD-DPO/models/Qwen/Qwen2.5-7B \
  model.teacher_name_or_path=/home/hungpv/projects/DSKD-DPO/models/Qwen/Qwen2.5-7B \
  model.student_name_or_path=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B \
  model.original_policy_name=/home/hungpv/projects/DSKD-DPO/models/meta-llama/Llama-3.2-1B \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=LlamaDecoderLayer \
  loss=KD_tisdpo \
  log_dir=KDPO_stdAllV1 \
  policy_mode=student \
  reference_mode=teacher \
  loss.beta=0.1 \
  loss.label_smoothing=0 \
  loss.average_log_prob=false \
  n_epochs=1 \
  max_grad_norm=1.0 \
  gradient_accumulation_steps=1 batch_size=1 eval_batch_size=1 \
  total_steps=7082 warmup_steps=708 eval_every=566 \
  lr=1e-6 scheduler=cosine \
  transform.method=origin \
  trainer=FSDPTrainer sample_during_eval=false \
  datasets=generated-data/ultra-feedback-tisdpo \