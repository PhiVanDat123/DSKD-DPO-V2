import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import socket
import resource
from typing import Optional, Set

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import torch.multiprocessing as mp
import transformers
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, build_exp_name
import trainers
from transform_config import TransformConfig, get_transform_config


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver(
    "build_exp_name",
    lambda loss_name, policy_model_name, datasets, reverse_dataset:
        build_exp_name(loss_name, policy_model_name, datasets, reverse_dataset),
)


def _print_param_meta(model, name="model"):
    meta_found = False
    for n, p in model.named_parameters():
        if getattr(p, "is_meta", False):
            print(f"[DEBUG] {name} param {n}: is_meta=True")
            meta_found = True
    return meta_found


def worker_main(rank: int, world_size: int, config: DictConfig, policy_path: str, reference_path: Optional[str] = None):
    # ---------------- Init distributed ----------------
    if 'FSDP' in config.trainer:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_idx = rank % max(1, n_gpus)
            torch.cuda.set_device(gpu_idx)
        init_distributed(rank, world_size, port=config.fsdp_port)

    device = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu")
    print(f"[rank {rank}] Using device: {device}")

    # ---------------- W&B ----------------
    if config.debug:
        wandb.init = lambda *a, **kw: None
        wandb.log = lambda *a, **kw: None
    elif rank == 0 and getattr(config, "wandb", None) and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    # ---------------- Load Policy ----------------
    print(f"[rank {rank}] Loading policy model from: {policy_path}")
    policy_dtype = getattr(torch, config.model.policy_dtype)

    if 'FSDP' in config.trainer and world_size > 1:
        # FSDP cần real tensors, không meta
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            policy_path,
            torch_dtype=policy_dtype,
            device_map="cpu",
            low_cpu_mem_usage=False
        )
    else:
        # Single GPU: load vào CPU rồi .to(device)
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            policy_path,
            torch_dtype=policy_dtype,
            device_map="cpu",
            low_cpu_mem_usage=False
        )
        policy.to(device)

    disable_dropout(policy)
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    if _print_param_meta(policy, "policy"):
        raise RuntimeError(f"[rank {rank}] Policy has meta params after load.")

    print(f"[rank {rank}] ✅ Policy model loaded successfully.")

    # ---------------- Load Reference Model (CPU only) ----------------
    reference_model = None
    if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
        if reference_path is None:
            raise RuntimeError("reference_path required for DPO-style losses")

        print(f"[rank {rank}] Loading reference model (CPU-only) from: {reference_path}")
        ref_dtype = getattr(torch, config.model.reference_dtype)

        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            reference_path,
            torch_dtype=ref_dtype,
            device_map={"": "cpu"},      # ✅ tuyệt đối không lên GPU
            low_cpu_mem_usage=False
        )

        disable_dropout(reference_model)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

        if _print_param_meta(reference_model, "reference_model"):
            raise RuntimeError(f"[rank {rank}] Reference model has meta params after load.")

        print(f"[rank {rank}] ✅ Reference model loaded on CPU and frozen.")

    # ---------------- Create Trainer ----------------
    TrainerClass = getattr(trainers, config.trainer)
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )
    if getattr(trainer, "reference_model", None) is None and reference_model is not None:
        trainer.reference_model = reference_model

    # ---------------- Train ----------------
    if torch.cuda.is_available():
        try:
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
        except Exception:
            pass

    print(f"[rank {rank}] 🚀 Starting training")
    try:
        trainer.train()
        trainer.save()
        print(f"[rank {rank}] ✅ Training finished successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[rank {rank}] ❌ CUDA OOM: {e}")
            print("Try reducing batch_size or using fp16.")
        raise


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    if OmegaConf.missing_keys(config):
        raise ValueError("Missing keys in config")

    if config.eval_every % config.batch_size != 0:
        config.eval_every -= config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        config.fsdp_port = get_open_port()

    os.makedirs(config.local_run_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.local_run_dir, 'config.yaml'))

    print('=' * 80)
    print(f"Running on {socket.gethostname()} | Output dir: {config.local_run_dir}")
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.output_dir)

    available_cuda = torch.cuda.device_count()
    print(f"Detected {available_cuda} CUDA devices")

    policy_path = config.model.policy_name_or_path
    reference_path = (
        config.model.reference_name_or_path
        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}
        else None
    )

    if 'FSDP' in config.trainer and available_cuda > 1:
        world_size = available_cuda
        print(f"🚀 Launching FSDP with {world_size} processes")

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except Exception:
            pass

        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy_path, reference_path),
            join=True
        )
    else:
        print("🚀 Launching single-process training")
        worker_main(0, 1, config, policy_path, reference_path)


if __name__ == '__main__':
    main()





