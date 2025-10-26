import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


# Register resolvers used by Hydra config
OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver(
    "build_exp_name",
    lambda loss_name, policy_model_name, datasets, reverse_dataset:
        build_exp_name(loss_name, policy_model_name, datasets, reverse_dataset),
)


def worker_main(rank: int, world_size: int, config: DictConfig, policy_path: str, reference_path: Optional[str] = None):
    """
    Worker entrypoint for distributed (FSDP) or single-GPU training.
    """

    # ---------------- Initialize distributed ----------------
    if 'FSDP' in config.trainer:
        torch.cuda.set_device(rank)
        init_distributed(rank, world_size, port=config.fsdp_port)

    has_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if has_cuda else "cpu")
    print(f"[rank {rank}] Using device: {device}")

    # ---------------- W&B Setup ----------------
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and getattr(config, "wandb", None) and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    # ---------------- Load Policy Model (on CPU first) ----------------
    print(f"[rank {rank}] Loading policy model from: {policy_path}")
    policy = None
    try:
        policy_dtype = getattr(torch, config.model.policy_dtype)

        # ✅ Load model on CPU to avoid OOM during initialization
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            policy_path,
            torch_dtype=policy_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        disable_dropout(policy)

        # Enable gradient checkpointing (saves VRAM)
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()

        print(f"[rank {rank}] Policy loaded successfully on CPU ({type(policy)})")
    except Exception as e:
        print(f"[rank {rank}] ❌ ERROR loading policy: {e}")
        raise

    # ---------------- Load Reference Model (CPU only) ----------------
    reference_model = None
    if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}:
        if reference_path is None:
            raise RuntimeError("reference_path must be provided for DPO-style losses")
        print(f"[rank {rank}] Loading reference model from: {reference_path}")

        try:
            ref_dtype = getattr(torch, config.model.reference_dtype)
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                reference_path,
                torch_dtype=ref_dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            disable_dropout(reference_model)
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad = False
            print(f"[rank {rank}] Reference model ready on CPU")
        except Exception as e:
            print(f"[rank {rank}] ❌ ERROR loading reference model: {e}")
            raise

    # ---------------- Create Trainer ----------------
    TrainerClass = getattr(trainers, config.trainer)
    print(f"[rank {rank}] Creating trainer...")

    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )

    # If trainer doesn’t attach reference_model automatically
    if reference_model is not None and getattr(trainer, "reference_model", None) is None:
        trainer.reference_model = reference_model
        print(f"[rank {rank}] Injected reference_model into trainer post-init")

    # ---------------- Debug GPU Memory ----------------
    if torch.cuda.is_available():
        try:
            print(f"[rank {rank}] CUDA memory summary before training:")
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
        except Exception:
            pass

    # ---------------- Train ----------------
    print(f"[rank {rank}] 🚀 Starting training")
    trainer.train()
    trainer.save()
    print(f"[rank {rank}] ✅ Training finished")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('⚠️ eval_every must be divisible by batch_size. Adjusting.')
        config.eval_every -= config.eval_every % config.batch_size
        print(f'Adjusted eval_every = {config.eval_every}')

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('No FSDP port specified; using open port:', free_port)
        config.fsdp_port = free_port

    os.makedirs(config.local_run_dir, exist_ok=True)
    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Running on {socket.gethostname()} | Output dir: {config.local_run_dir}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.output_dir)

    world_size = torch.cuda.device_count() if 'FSDP' in config.trainer else 1
    print(f"Detected {world_size} CUDA devices")

    policy_path = config.model.policy_name_or_path
    reference_path = (
        config.model.reference_name_or_path
        if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'KD_tisdpo'}
        else None
    )

    if 'FSDP' in config.trainer and world_size > 1:
        print(f"🚀 Launching FSDP with {world_size} processes")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Set RLIMIT_NOFILE soft limit to {hard} from {soft}")

        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy_path, reference_path),
            join=True
        )
    else:
        print("🚀 Launching single-process training (no FSDP)")
        worker_main(0, 1, config, policy_path, reference_path)


if __name__ == '__main__':
    main()



