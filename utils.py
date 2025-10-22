import os
import getpass
from datetime import datetime, timezone, timedelta
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
from typing import Dict, Union, Type, List

import time
import logging
import torch.nn as nn
from datetime import timedelta
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_constant_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup
)


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def build_exp_name(loss_name: str, model_name: str, datasets: Union[str, List[str]], reverse_dataset: bool) -> str:
    """Build experiment name by combining loss name, model name, and dataset name(s)."""
    # Extract the model name without path
    model_short_name = model_name.split('/')[-1]
    
    dataset_part = '_'.join(datasets)
    
    # Add 'reverse' suffix if loss is dpo and reverse_dataset is True
    if loss_name == "dpo" and reverse_dataset:
        return f"{loss_name}_{model_short_name}_{dataset_part}_reverse"
        

    # import ipdb; ipdb.set_trace()
    if loss_name == "tisdpo":
        return f"{loss_name}_{model_short_name}_{dataset_part}"
        
    return f"{loss_name}_{model_short_name}_{dataset_part}"



def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(path: str) -> str:
    """Return the path to the cache directory."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
    

def get_local_run_dir(exp_name: str, local_dir: str) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now(timezone(timedelta(hours=8)))  # China Standard Time (UTC+8)
    timestamp = now.strftime("%m-%d_%H-%M")
    run_dir = f"{get_local_dir(local_dir)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

'''
def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    print("[DEBUG_SLICE_AND_MOVE_BATCH_FOR_DEVICE] Before slice Batch size:", batch['chosen_student_input_ids'].shape)
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    print("[DEBUG_SLICE_AND_MOVE_BATCH_FOR_DEVICE] After slice and move batch size:", on_device['chosen_student_input_ids'].shape)
    return on_device
'''

# ...existing code...
def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device.
    If total samples < world_size, replicate the full batch on every rank to avoid empty slices.
    """
    first = next(iter(batch.values()))
    if isinstance(first, torch.Tensor):
        total = first.size(0)
    elif isinstance(first, (list, tuple)):
        total = len(first)
    else:
        raise ValueError("Unsupported batch element type for slicing")

    print("[DEBUG_SLICE_AND_MOVE_BATCH_FOR_DEVICE] Before slice Batch size:", total, "rank:", rank, "world_size:", world_size)

    if total == 0:
        raise ValueError("slice_and_move_batch_for_device received an empty batch (total==0)")

    # If too few samples to split across ranks, replicate the whole batch on each rank (safe for eval)
    if total < world_size:
        sliced = {k: v for k, v in batch.items()}
    else:
        start = (total * rank) // world_size
        end = (total * (rank + 1)) // world_size
        sliced = {k: (v[start:end] if isinstance(v, (torch.Tensor, list, tuple)) else v) for k, v in batch.items()}

    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}

    # safe debug printing
    first_after = next(iter(on_device.values()))
    try:
        size_after = first_after.shape if isinstance(first_after, torch.Tensor) else len(first_after)
    except Exception:
        size_after = "unknown"
    print("[DEBUG_SLICE_AND_MOVE_BATCH_FOR_DEVICE] After slice and move batch size:", size_after, "rank:", rank)
    return on_device

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(71)
        np.random.seed(71)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)

#dskd-utils
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s]  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Logging
def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def log_rank(content, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        logging.info(content)


# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


# Initialize
def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

'''
def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    log_rank(f"Using world size: {args.world_size}")
    
    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()

    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=30))
'''

'''
def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    if args.model_parallel:
        raise NotImplementedError

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save_dir != None:
        os.makedirs(args.save_dir, exist_ok=True)
'''

# Load and save model
def get_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path)
    
    st_time = time.time()
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        dtype = torch.float32 if args.fp32 else torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=dtype
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=torch.float32
            )
            model = model.half()
        
        if args.peft is not None:
            if args.peft == "lora":
                model.enable_input_require_grads()
                if args.peft_path is not None:
                    if args.do_train:
                        _model = PeftModel.from_pretrained(model, args.peft_path)
                        state_dict = dict(_model.state_dict().items())
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM, 
                            inference_mode=(not args.do_train), 
                            r=args.peft_lora_r, 
                            lora_alpha=args.peft_lora_alpha, 
                            lora_dropout=args.peft_lora_dropout
                        )
                        model = get_peft_model(model, peft_config)
                        model.load_state_dict(state_dict)
                        
                        del _model
                        del state_dict
                    else:
                        model = PeftModel.from_pretrained(model, args.peft_path)
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, 
                        inference_mode=(not args.do_train), 
                        r=args.peft_lora_r, 
                        lora_alpha=args.peft_lora_alpha, 
                        lora_dropout=args.peft_lora_dropout
                    )
                    model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                log_rank(' > number of parameters: {:n}'.format(
                    sum([p.nelement() for p in model.parameters()])))
        # model = DDP(model)
        # NOTE: no need for DDP since deepspeed has done
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    log_rank(f"Model load time: {ed_time - st_time}s")
    
    return model


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: 
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=torch.float16
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=torch.float32
            )
            model = model.half()
        
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                log_rank(' > number of parameters of the teacher model: {:n}'.format(
                    sum([p.nelement() for p in model.parameters()])))

    model.eval()
    
    return model


def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_type=="qwen":
        tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    log_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters
        )
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min
        )
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5
        )
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler