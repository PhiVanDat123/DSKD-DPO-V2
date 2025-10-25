import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from criterions.dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from distiller import Distiller
from preference_datasets import DistillDPODataset, CustomCollate
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def tisdpo_loss(chosen_logps_margin: torch.FloatTensor,
                rejected_logps_margin: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                beta: float, alpha: float = 0.5, token_level: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)  
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False, token_level: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # import ipdb; ipdb.set_trace()
    if token_level:
        weights = weights[:, 1:].clone()
        batch_logps = (per_token_logps * loss_mask * weights).sum(-1)
    else:
        batch_logps = (per_token_logps * loss_mask).sum(-1)
    
    if average_log_prob:
        return batch_logps/loss_mask.sum(-1)
    else:
        return batch_logps


def _get_batch_logps_tisdpo(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor, weights: torch.FloatTensor=None, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weights: Weights for each token. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per 
        (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If True, return the log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    labels[labels == -100] = 0

    vocab_ps = logits.softmax(-1)
    vocab_logps = vocab_ps.log()
    print(f"[DEBUG] vocab_logps.shape = {vocab_logps.shape}")

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)

    labels = labels.long()
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps
    weights = weights[:, 1:].clone()
    
    if average_log_prob:
        return (logps_margin * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * weights * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * weights * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * weights * loss_mask).sum(-1), \
            (per_position_kl * weights * loss_mask).sum(-1), \
            (per_token_logps * weights * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    #max_length = max(batch[f'chosen_{mode}_input_ids'].shape[1], batch[f'rejected_{mode}_input_ids'].shape[1])
    max_length = max(v.shape[1] for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim == 2)
    concatenated_batch = {}

    print("[DEBUG] Batch keys:", batch.keys())
    print("[DEBUG] Before concatenation:")
    for k, v in batch.items():
        print(k, type(v))
        if isinstance(v, torch.Tensor):
            print(v[:, :10])
            print(f"  {k}.shape = {v.shape}, numel={v.numel()}, batch_size={v.shape[0]}")

    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)

    # --- Debug print: kiểm tra batch size sau khi concat ---
    for k, v in concatenated_batch.items():
        print(f"[DEBUG] {k}.shape = {v.shape}, numel={v.numel()}, batch_size={v.shape[0]}")

    return concatenated_batch



class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
            
            If multiple GPUs are present, naively splits the model across them, effectively
            offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir
        self.distiller = Distiller(config, device = self.rank)
        self.DSKD = DualSpaceKDWithCMA(config, self.distiller, padding_id=-100)

        teacher_tokenizer_name_or_path = (
            config.model.teacher_tokenizer_name_or_path or config.model.teacher_name_or_path
        )
        rank0_print(f"Loading teacher tokenizer {teacher_tokenizer_name_or_path}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name_or_path)
        student_tokenizer_name_or_path = (
            config.model.student_tokenizer_name_or_path or config.model.student_name_or_path
        )
        rank0_print(f"Loading student tokenizer {student_tokenizer_name_or_path}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name_or_path)
        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id
            # self.teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token_id = self.student_tokenizer.eos_token_id
            # self.student_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        self.tokenizer = {
            "teacher": self.teacher_tokenizer,
            "student": self.student_tokenizer,
        }

        self.policy = policy
        self.reference_model = reference_model

        self.train_dataset = DistillDPODataset(
            local_file_path = "/home/hungpv/projects/DSKD-DPO/generated-data/ultra-feedback-tisdpo/train_prefs.jsonl",
            student_tokenizer = self.student_tokenizer,
            teacher_tokenizer = self.teacher_tokenizer,
        )
        self.eval_dataset = DistillDPODataset(
            local_file_path = "/home/hungpv/projects/DSKD-DPO/generated-data/ultra-feedback-tisdpo/test_prefs.jsonl",
            student_tokenizer = self.student_tokenizer,
            teacher_tokenizer = self.teacher_tokenizer,
        )
        
        self.train_iterator = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            # batch_size=1,
            shuffle=True,
            collate_fn=CustomCollate(self.tokenizer),
            pin_memory=True,
            num_workers=2,
            drop_last=True,
        )
        rank0_print("Loaded train data iterator")
        # rank0_print("[DEBUG_TRAIN_ITER] Train iterator length:", self.train_iterator[0]['chosen_student_input_ids'].shape)

        self.eval_iterator = DataLoader(
            self.eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=True,
            # collate_fn=get_collate_fn(self.tokenizer),
            collate_fn=CustomCollate(self.tokenizer),
            pin_memory=True,
            num_workers=2,
            drop_last=True,
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
    
    
    def tisdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                  batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # print("[DEBUG_INSIDE_TISDPO] Batch keys:", batch.keys())
        print("[DEBUG_INSIDE_TISDPO] Batch size:", batch['chosen_student_input_ids'].shape)
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_student_input_ids'],
                           attention_mask=concatenated_batch['concatenated_student_attention_mask']).logits.to(torch.float32)
        
        '''
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch[
                                                       'concatenated_attention_mask']).logits.to(torch.float32)
        '''
        reference_all_logits = self.DSKD.compute_t2s_logits(concatenated_batch, self.distiller, self.policy, self.reference_model)
        all_logps_margin, all_position_kl, all_logps = _get_batch_logps_tisdpo(all_logits, reference_all_logits, concatenated_batch['concatenated_student_labels'], concatenated_batch['concatenated_student_weight'], average_log_prob=False)

        chosen_logps_margin = all_logps_margin[:batch['chosen_student_input_ids'].shape[0]]
        rejected_logps_margin = all_logps_margin[batch['chosen_student_input_ids'].shape[0]:]
        chosen_position_kl = all_position_kl[:batch['chosen_student_input_ids'].shape[0]]
        rejected_position_kl = all_position_kl[batch['chosen_student_input_ids'].shape[0]:]

        chosen_logps = all_logps[:batch['chosen_student_input_ids'].shape[0]].detach()
        rejected_logps = all_logps[batch['chosen_student_input_ids'].shape[0]:].detach()

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, \
            chosen_logps, rejected_logps


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        print("[DEBUG_GET_BATCH_METRICS] Batch size:", batch['chosen_student_input_ids'].shape)
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'KD_tisdpo':
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps\
                = self.tisdpo_concatenated_forward(self.policy, self.reference_model, batch)
            loss_tisdpo, chosen_rewards, rejected_rewards = tisdpo_loss(chosen_logps_margin, rejected_logps_margin,
                                                                 chosen_position_kl, rejected_position_kl,
                                                                 beta=loss_config.beta, alpha=loss_config.alpha, token_level=loss_config.token_level)
            loss_dtw = self.DSKD.compute_dtw_loss(batch, self.distiller, self.policy, self.reference_model)
            losses = self.config.tis_dpo_rate * loss_tisdpo + self.config.dtw_rate * loss_dtw
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False, token_level=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    print("[DEBUG_EVAL] Eval batch size:", eval_batch['chosen_student_input_ids'].shape)
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                print("[DEBUG_TRAIN] Batch size:", batch['chosen_student_input_ids'].shape)
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                print("[DEBUG_TRAIN] Global microbatch size:", global_microbatch['chosen_student_input_ids'].shape)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                print("[DEBUG_TRAIN] Local microbatch size:", local_microbatch['chosen_student_input_ids'].shape)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()
                break 

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy and tokenizer to disk."""
        if output_dir is None:
            model_save_dir = os.path.join(self.run_dir, f'LATEST')
        else:
            model_save_dir = output_dir
            
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Save model using transformers save_pretrained
        self.policy.save_pretrained(model_save_dir)
        rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
        
        # Save tokenizer alongside the model
        self.student_tokenizer.save_pretrained(model_save_dir)
        
        # Save metrics separately
        if metrics is not None:
            metrics_file = os.path.join(model_save_dir, "training_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump({"step": self.example_counter, "metrics": metrics}, f)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.policy_block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.policy_block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy and tokenizer state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            # Save model using transformers save_pretrained
            if output_dir is None:
                model_save_dir = self.run_dir
            else:
                model_save_dir = output_dir
            
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Get the original model class and instantiate it directly
            from transformers import AutoModelForCausalLM
            model_name = self.config.model.policy_name_or_path
            unwrapped_model = AutoModelForCausalLM.from_pretrained(model_name)
            unwrapped_model.load_state_dict(policy_state_dict)
            
            # Save using transformers save_pretrained
            unwrapped_model.save_pretrained(model_save_dir)
            rank0_print(f"Model saved to {model_save_dir} using save_pretrained")
            del unwrapped_model
            
            # Save tokenizer alongside the model
            self.student_tokenizer.save_pretrained(model_save_dir)
            
            # Save metrics separately
            if metrics is not None:
                metrics_file = os.path.join(model_save_dir, "training_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"step": self.example_counter, "metrics": metrics}, f)
            
        del policy_state_dict
        dist.barrier()
        