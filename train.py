import os
# Giữ expandable_segments nhưng thêm max_split_size_mb để giảm phân mảnh (tùy hệ thống)
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

# Nếu bạn dùng Linux cluster đôi khi cần set start method 'spawn' sớm
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # start method đã set trước đó
    pass

# Register resolvers used by Hydra config
OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver(
    "build_exp_name",
    lambda loss_name, policy_model_name, datasets, reverse_dataset:
        build_exp_name(loss_name, policy_model_name, datasets, reverse_dataset),
)


def _print_param_meta(model, name="model"):
    """Debug helper: show if any param is meta or has no device."""
    meta_found = False
    for n, p in model.named_parameters():
        is_meta = getattr(p, "is_meta", False)
        dev = getattr(p, "device", None)
        if is_meta or dev is None:
            print(f"[DEBUG] {name} param {n}: is_meta={is_meta}, device={dev}, numel={p.numel()}")
            meta_found = True
            # don't break, print all suspicious params
    return meta_found


def worker_main(rank: int, world_size: int, config: DictConfig, policy_path: str, reference_path: Optional[str] = None):
    """
    Worker entrypoint for distributed (FSDP) or single-GPU training.
    """

    # ---------------- Initialize distributed ----------------
    if 'FSDP' in config.trainer:
        # đảm bảo device mapping ổn định
        # torch.cuda.set_device may raise if rank >= cuda_count; check first
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            # map rank to gpu index defensively
            gpu_idx = rank % max(1, n_gpus)
            torch.cuda.set_device(gpu_idx)
        init_distributed(rank, world_size, port=config.fsdp_port)

    has_cuda = torch.cuda.is_available()
    # map device same như set_device ở trên
    device = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}" if has_cuda else "cpu")
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

    # ---------------- Load Policy Model ----------------
    print(f"[rank {rank}] Loading policy model from: {policy_path}")
    policy = None
    try:
        policy_dtype = getattr(torch, config.model.policy_dtype)

        # Khi chạy FSDP (multi-GPU): KHÔNG dùng low_cpu_mem_usage=True (tránh meta tensors)
        if 'FSDP' in config.trainer and world_size > 1:
            # Load fully on CPU (real tensors) so FSDP có thể wrap/distribute an toàn.
            policy = transformers.AutoModelForCausalLM.from_pretrained(
                policy_path,
                torch_dtype=policy_dtype,
                low_cpu_mem_usage=False,   # IMPORTANT: avoid meta tensors when using FSDP
                device_map="cpu",          # load on CPU then FSDP will shard/dispatch
            )
            print(f"[rank {rank}] Policy loaded on CPU for FSDP (no low_cpu_mem_usage).")
        else:
            # Single-process/single-GPU: let HF dispatch automatically if GPU available
            if torch.cuda.is_available():
                policy = transformers.AutoModelForCausalLM.from_pretrained(
                    policy_path,
                    torch_dtype=policy_dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=False,
                )
                # if from_pretrained dispatched to gpu, don't call .to(device) again
            else:
                policy = transformers.AutoModelForCausalLM.from_pretrained(
                    policy_path,
                    torch_dtype=policy_dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=False,
                )
                policy.to(device)
            print(f"[rank {rank}] Policy loaded (auto dispatch if possible).")

        disable_dropout(policy)

        # Enable gradient checkpointing if available
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()

        # Debug: kiểm tra metadata parameters (phát hiện meta tensors sớm)
        if _print_param_meta(policy, "policy"):
            raise RuntimeError(f"[rank {rank}] Detected meta parameters in policy after load. Aborting to avoid storage errors.")

    except RuntimeError as e:
        # Nếu OOM hoặc meta issue, in ra thông tin hữu ích
        print(f"[rank {rank}] ❌ ERROR loading policy (RuntimeError): {e}")
        raise
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

            # same rule: avoid low_cpu_mem_usage when we will use FSDP
            if 'FSDP' in config.trainer and world_size > 1:
                reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                    reference_path,
                    torch_dtype=ref_dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=False,
                )
            else:
                # for single-gpu, let HF dispatch to GPU automatically
                if torch.cuda.is_available():
                    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                        reference_path,
                        torch_dtype=ref_dtype,
                        device_map="auto",
                        low_cpu_mem_usage=False,
                    )
                else:
                    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                        reference_path,
                        torch_dtype=ref_dtype,
                        device_map="cpu",
                        low_cpu_mem_usage=False,
                    )
                    reference_model.to(device)

            disable_dropout(reference_model)
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad = False

            if _print_param_meta(reference_model, "reference_model"):
                raise RuntimeError(f"[rank {rank}] Detected meta parameters in reference_model after load. Aborting to avoid storage errors.")

            print(f"[rank {rank}] Reference model ready.")
        except RuntimeError as e:
            print(f"[rank {rank}] ❌ ERROR loading reference model (RuntimeError): {e}")
            raise
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

    # Nếu trainer không tự attach reference_model
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
    try:
        print(f"[rank {rank}] 🚀 Starting training")
        trainer.train()
        trainer.save()
        print(f"[rank {rank}] ✅ Training finished")
    except RuntimeError as e:
        # Catch common CUDA OOM and give actionable hints
        if 'out of memory' in str(e).lower():
            print(f"[rank {rank}] ❌ CUDA out of memory during training: {e}")
            print("Suggestions: reduce batch_size, use mixed precision (torch_dtype=float16),"
                  " use 8-bit loading (bitsandbytes) or enable gradient checkpointing.")
        raise


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

    # kiểm tra number of visible CUDA devices
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

        # tăng file descriptors để tránh issues khi spawn nhiều tiến trình
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"Set RLIMIT_NOFILE soft limit to {hard} from {soft}")
        except Exception as e:
            print("Could not set RLIMIT_NOFILE:", e)

        # Optional: nếu muốn ép mapping GPU cụ thể, set CUDA_VISIBLE_DEVICES trước spawn
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"   # set phù hợp với cluster của bạn

        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy_path, reference_path),
            join=True
        )
    else:
        # single-process
        print("🚀 Launching single-process training (no FSDP)")
        worker_main(0, 1, config, policy_path, reference_path)


if __name__ == '__main__':
    main()




