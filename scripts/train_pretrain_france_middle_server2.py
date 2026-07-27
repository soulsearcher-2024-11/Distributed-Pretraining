#!/usr/bin/env python3
# coding: utf-8
"""
pretrain_gpt_1B_pytorch.py
Single-GPU (RTX 4090) pretraining script using PyTorch + Hugging Face transformers.
Tailored defaults for a ~1B-parameter GPT-like causal LM trained on local txt files.
Key features/choices for single 24GB GPU:
 - Model: create-from-config GPT-2 like (default: ~24 layers, 2048 hidden -> ~target 1B params)
 - Memory saving: gradient_checkpointing, mixed precision via torch.amp (fp16/bf16)
 - Streaming data via an IterableDataset to avoid concatenating entire corpus in memory
 - Small device batch size (1) + gradient_accumulation to get effective batch sizes
 - Checkpointing and resume

IMPORTANT:
 - The chosen config is an *approximation* toward ~1B params. Always verify with
   `model_size = sum(p.numel() for p in model.parameters())` after building model and
   adjust n_layer / n_embd / n_head accordingly.
 - Tune gradient_accumulation_steps to get desired effective batch size.
 - This script purposely avoids distributed training code (DDP/DeepSpeed) since you
   requested single-GPU RTX4090.

Usage example:
  python pretrain_gpt_1B_pytorch.py \
    --train_folder /path/to/txt_dir \
    --output_dir ./checkpoints/exp1 \
    --max_steps 20000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --fp16
"""


import os
import sys
import math
import time
import argparse
import inspect
from pathlib import Path
from typing import List, Iterator, Dict, Optional, Any
import json
import re
import io
import random
from types import MethodType
import gc
import traceback
import grpc
import threading
import queue
import numpy as np
import pickle


# Ensure local test_train modules take precedence over repo root.
# We keep repo root available for shared modules (configuration/tokenization/etc.).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Do NOT insert PROJECT_ROOT at position 0 (it would shadow test_train/modeling_MandelbrotV1.py).
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(1 if len(sys.path) > 0 else 0, PROJECT_ROOT)


import mandelbrot_service_pb2_grpc
import mandelbrot_service_pb2

import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# FSDP (Zero-3) Support
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision as FSDPMixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# torch.autograd.set_detect_anomaly(True)

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import tokenization_MandelbrotV1_fast as tokenizer_module
from tokenization_MandelbrotV1_fast import MandelbrotV1TokenizerFast, MandelbrotV1TokenizerManager,global_vars
from configuration_MandelbrotV1 import MandelbrotV1Config
import modeling_MandelbrotV1 as modeling_module
from modeling_MandelbrotV1 import MandelbrotV1ForCausalLM,MandelbrotV1DecoderBlock,nccl_mgr

import psutil
import gc
from torchviz import make_dot
import numpy as np
from collections import defaultdict
from BaseTextDataset import StreamingTextDataset, ActivationCacheDataset, PairedLineTextDataset, worker_init_fn,_worker_context,AsyncPrefetchDataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def global_worker_init_fn(worker_id,dataset_obj=None):
    # Read Configuration From Environment Variables
    
    dataset_obj = dataset_obj if dataset_obj is not None else get_worker_info().dataset

    ddp_enabled = dataset_obj.ddp_enabled
    if ddp_enabled:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        worker_seed = int(dataset_obj.seed) + local_rank * 10000 + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    # Client mode initialize tokenizer and stub
    
    worker_init_fn(worker_id,dataset_obj)   # Import from BaseTextDataset, which reads Tokenizer_NAME and so on inside

def _compute_mlp_layer_dims(config):
    """Reproduce MandelbrotV1MLP layer_dim construction to get in/out dims per FFN layer."""
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise ValueError(f"Invalid config.hidden_size={hidden_size}")
    use_dimensionality_elevation = bool(getattr(config, "use_dimensionality_elevation", False))

    layer_num = int(getattr(config, "layer_num", 0) or 0)
    if layer_num <= 0:
        layer_num = 3

    layer_dimdiff = getattr(config, "layer_dimdiff", None)
    if not isinstance(layer_dimdiff, dict):
        layer_dimdiff = {str(i): 2 for i in range(layer_num)}
    else:
        layer_dimdiff = {str(k): int(v) for k, v in layer_dimdiff.items()}

    layer_dim = {0: hidden_size}
    for i in range(1, len(layer_dimdiff) + 1):
        k = i - 1
        if i == 1:
            if use_dimensionality_elevation:
                layer_dim1 = hidden_size + 1
                layer_dim[0] = layer_dim1
            else:
                layer_dim1 = hidden_size
        else:
            layer_dim1 = layer_dim.get(k)
        diff = int(layer_dimdiff.get(str(k), 2))
        if i == 1 and use_dimensionality_elevation:
            layer_dim[i] = layer_dim1 + diff - 1
        else:
            layer_dim[i] = layer_dim1 + diff
    return layer_dim


def _extract_sub_state_dict(state_dict, prefix: str):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out

def _build_lightweight_delta_state_dict(model) -> dict:
    """Export trained params in full-model key space.

    This allows later merging into a full MandelbrotV1 checkpoint without having
    loaded the full model during training.
    """

    # idx = int(model.last_layer_idx) if hasattr(model, "last_layer_idx") else 0  
    delta = {}

    # for k, v in model.ffn.state_dict().items():
    #     delta[f"model.blocks.0.mlp.layers.{idx}.{k}"] = v.detach().cpu()
    # for k, v in model.ffn_layernorm.state_dict().items():
    #     delta[f"model.blocks.0.mlp.layernorm.{idx}.{k}"] = v.detach().cpu()
    # for k, v in model.final_norm.state_dict().items():
    #     delta[f"model.norm.{idx}.{k}"] = v.detach().cpu()

    # if model.use_layer_embeddings:
    #     for k, v in model.proj_to_hidden.state_dict().items():
    #         delta[f"lm_head.{idx}.{k}"] = v.detach().cpu()
    # else:
    #     for k, v in model.lm_head.state_dict().items():
    #         delta[f"lm_head.{k}"] = v.detach().cpu()

    return delta



def _get_block_last_layer_idx(config, block_idx: int) -> int:
    layer_num = int(getattr(config, "layer_num", 0) or 0)
    if layer_num <= 0:
        raise ValueError(f"Invalid config.layer_num={layer_num}")
    return int(block_idx) * layer_num + (layer_num - 1)


def _resolve_block_cache_start_idx(requested_start_block_idx: int, cache_after_block_idx: Optional[int]) -> int:
    start_block_idx = int(requested_start_block_idx or 0)
    if start_block_idx == 0 and cache_after_block_idx is not None:
        inferred_start = int(cache_after_block_idx) + 1
        if inferred_start > 0:
            start_block_idx = inferred_start
    return start_block_idx


def _get_block_input_hidden_size(config, block_idx: int) -> int:
    from modeling_MandelbrotV1 import MandelbrotV1DecoderBlock

    if block_idx <= 0:
        return int(config.hidden_size)

    current_hidden_size = int(config.hidden_size)
    for prev_block_idx in range(block_idx):
        prev_block = MandelbrotV1DecoderBlock(config, prev_block_idx, current_hidden_size)
        current_hidden_size = int(prev_block.getLastLayerDim())
    return current_hidden_size




try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("⚠️ Pynvml is not available, GPU UTILZATION will be limited")


global_dict = globals()

# ===== Training Performance Profiler =====
class TrainingPerformanceProfiler:
    """Performance monitors for training processes: FLOPs, throughput, GPU UTILZATION"""
    
    def __init__(self, model, config, device='cuda', enable_gpu_monitor=False, grad_accum_steps=1):
        self.model = model
        self.config = config
        self.device = device
        self.enable_gpu_monitor = enable_gpu_monitor
        self.step_metrics = []
        self.epoch_metrics = []
        self.grad_accum_steps = grad_accum_steps
        
    def estimate_training_flops(self, batch_size, seq_len):
        """Estimating the FLOPs (forward + backward) of a single training step"""
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_hidden_layers
        intermediate_size = self.config.intermediate_size
        num_attention_heads = self.config.num_attention_heads
        vocab_size = self.config.vocab_size
        
        # Forward pass FLOPs
        # 1. Embedding
        embedding_flops = batch_size * seq_len * hidden_size * 2
        
        # 2. Transformer layers
        per_layer_flops = 0
        
        # Attention
        qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
        attn_scores = batch_size * num_attention_heads * seq_len * seq_len * (hidden_size // num_attention_heads)
        attn_out = batch_size * num_attention_heads * seq_len * seq_len * (hidden_size // num_attention_heads)
        out_proj = batch_size * seq_len * hidden_size * hidden_size
        
        # FFN or MoE
        ffn_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
        if hasattr(self.config, 'n_routed_experts') and self.config.n_routed_experts:
            num_experts = self.config.n_routed_experts
            experts_per_tok = getattr(self.config, 'num_experts_per_tok', 2)
            routing_flops = batch_size * seq_len * hidden_size * num_experts
            expert_flops = experts_per_tok * batch_size * seq_len * hidden_size * intermediate_size * 2
            ffn_flops = routing_flops + expert_flops
        
        per_layer_flops = qkv_flops + attn_scores + attn_out + out_proj + ffn_flops
        transformer_flops = num_layers * per_layer_flops
        
        # 3. LM head
        lm_head_flops = batch_size * seq_len * hidden_size * vocab_size
        
        forward_flops = embedding_flops + transformer_flops + lm_head_flops
        
        # Backward pass ≈ 2x forward pass
        backward_flops = 2 * forward_flops
        
        total_flops = forward_flops + backward_flops
        
        return {
            'total_flops': total_flops,
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
        }
    
    def get_gpu_stats(self):
        """Get GPU utilization and GPU memory usage"""
        gpu_stats = {}
        
        if torch.cuda.is_available():
            # Collect details only when GPU monitoring is enabled
            if self.enable_gpu_monitor and NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_stats['gpu_util'] = utilization.gpu
                    gpu_stats['memory_util'] = utilization.memory
                    gpu_stats['memory_used_gb'] = memory.used / (1024**3)
                    gpu_stats['memory_total_gb'] = memory.total / (1024**3)
                    gpu_stats['power_watts'] = power
                    gpu_stats['temperature_c'] = temp
                except:
                    pass
            
            # PyTorch GPU-memory statistics (always collected)
            gpu_stats['torch_allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_stats['torch_reserved_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_stats['torch_max_allocated_gb'] = torch.cuda.max_memory_allocated(0) / (1024**3)
        
        return gpu_stats
    
    def record_step(self, batch_size, seq_len, elapsed_time, loss, global_step, world_size=1):
        """Document the performance of individual training steps"""
        
        # 1. Calculated cards FLOPs
        flops_info = self.estimate_training_flops(batch_size, seq_len)
        
        # 2. Acquisition of GPU status (require only on the master card or when needed, avoid multiple-card printing)
        gpu_stats = self.get_gpu_stats()
        
        # Core amendments: calculation of validity time
        # If the gradient accumulation is enabled, it takes N micro-step time to complete a valid parameter update
        effective_elapsed_time = elapsed_time
        
        # Core revision: calculation of global throughput
        # Token token processed by singlecard
        single_card_tokens = batch_size * seq_len
        # Number of tokens processed by cluster = single card * card count
        total_tokens_processed = single_card_tokens * world_size
        
        # THROUGHPUHPUT(Tokens/s) = Total Token/ (one step takes time * cumulative steps)
        # Note: here assumes that the average time taken is a single micro-step
        if effective_elapsed_time > 0:
            tokens_per_sec = total_tokens_processed / effective_elapsed_time
            # Effective Arithmetic (TFLOPS) = (single-card FLOPs *cards) / (single-step time * accumulated steps)
            tflops_per_sec = (flops_info['total_flops'] * world_size / 1e12) / effective_elapsed_time
        else:
            tokens_per_sec = 0
            tflops_per_sec = 0

        metrics = {
            'step': global_step,
            'loss': loss,
            'elapsed_time_s': elapsed_time, # Recording original microsteps takes time
            'batch_size': batch_size * world_size, # Record Global Batch Size
            'seq_len': seq_len,
            'tokens_processed': total_tokens_processed,
            'tokens_per_sec': tokens_per_sec,
            'ms_per_token': (effective_elapsed_time * 1000) / total_tokens_processed if total_tokens_processed > 0 else 0,
            'total_tflops': flops_info['total_flops'] * world_size / 1e12,
            'tflops_per_sec': tflops_per_sec,
        }
        
        metrics.update(gpu_stats)
        self.step_metrics.append(metrics)
        
        return metrics
    
    def get_average_metrics(self, last_n_steps=100):
        """Obtaining the average indicator for the recent N step"""
        if not self.step_metrics:
            return {}
        
        recent = self.step_metrics[-last_n_steps:]
        avg_metrics = {}
        
        numeric_keys = ['tokens_per_sec', 'tflops_per_sec', 'ms_per_token', 
                       'gpu_util', 'memory_used_gb', 'power_watts', 'temperature_c']
        
        for key in numeric_keys:
            values = [m[key] for m in recent if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'max_{key}'] = np.max(values)
                avg_metrics[f'min_{key}'] = np.min(values)
        
        return avg_metrics
    
    def print_summary(self, last_n_steps=100):
        """Print Performance Summary"""
        if not self.step_metrics:
            print("No performance data available")
            return
        
        avg = self.get_average_metrics(last_n_steps)
        latest = self.step_metrics[-1]
        
        print(f"\n{'='*70}")
        print(f"Training Performance Report (last {min(last_n_steps, len(self.step_metrics))} steps)")
        print(f"{'='*70}")
        
        print(f"\n Throughput:")
        print(f"   Avg: {avg.get('avg_tokens_per_sec', 0):.2f} tokens/s")
        print(f"   Max: {avg.get('max_tokens_per_sec', 0):.2f} tokens/s")
        print(f"   Current: {latest.get('tokens_per_sec', 0):.2f} tokens/s")
        
        print(f"\n Compute Performance:")
        print(f"   Avg Compute: {avg.get('avg_tflops_per_sec', 0):.2f} TFLOP/s")
        print(f"   Peak Compute: {avg.get('max_tflops_per_sec', 0):.2f} TFLOP/s")
        print(f"   Latency per token: {avg.get('avg_ms_per_token', 0):.2f} ms")
        
        if 'avg_gpu_util' in avg:
            print(f"\n GPU Utilization:")
            print(f"   Avg GPU Util: {avg.get('avg_gpu_util', 0):.1f}%")
            print(f"   Peak GPU Util: {avg.get('max_gpu_util', 0):.1f}%")
            print(f"   Avg Memory Used: {avg.get('avg_memory_used_gb', 0):.2f} GB")
            print(f"   Peak Memory Used: {avg.get('max_memory_used_gb', 0):.2f} GB")
            
        if 'avg_power_watts' in avg:
            print(f"\n Power:")
            print(f"   Avg Power: {avg.get('avg_power_watts', 0):.1f} W")
            print(f"   Peak Power: {avg.get('max_power_watts', 0):.1f} W")
            print(f"   Avg Temperature: {avg.get('avg_temperature_c', 0):.1f}°C")
        
        print(f"{'='*70}\n")

# ── GPU Monitor (communication expense + GPU utilization) -
class GPUMonitor:
    """Monitors GPU memory, utilization, NCCL and gRPC communication.
    Wraps torch.distributed collectives AND intercepts all gRPC unary calls
    to track distributed communication bytes and timing.
    """

    def __init__(self):
        self.comm_bytes = 0
        self.comm_calls = 0
        self.comm_cpu_us = 0.0
        self.orig_td_funcs = {}
        self.orig_grpc_unary_unary = None
        self.active = False

    @staticmethod
    def _first_tensor(*args, **kwargs):
        for a in args:
            if isinstance(a, torch.Tensor):
                return a
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _make_td_wrapper(self, orig_fn):
        mon = self
        @functools.wraps(orig_fn)
        def traced(*args, **kwargs):
            tensor = self._first_tensor(*args, **kwargs)
            if tensor is not None:
                t0 = time.perf_counter()
                result = orig_fn(*args, **kwargs)
                dt_us = (time.perf_counter() - t0) * 1e6
                mon._record_bytes(tensor.numel() * tensor.element_size(), dt_us)
                return result
            else:
                mon.comm_calls += 1
                return orig_fn(*args, **kwargs)
        return traced

    def _setup_grpc_trace(self, orig_fn):
        """Create a traced version of unary_unary by wrapping the returned callable."""
        mon = self
        def _patched_unary_unary(channel_self, method, request_serializer, response_deserializer, **kwargs):
            multi_callable = orig_fn(channel_self, method, request_serializer, response_deserializer, **kwargs)
            _orig_call = multi_callable.__call__

            @functools.wraps(_orig_call)
            def _traced_call(request, *a, **kw):
                t0 = time.perf_counter()
                try:
                    req_bytes = request.ByteSize()
                except Exception:
                    try:
                        req_bytes = len(request.SerializeToString())
                    except Exception:
                        req_bytes = 0
                result = _orig_call(request, *a, **kw)
                dt_us = (time.perf_counter() - t0) * 1e6
                mon._record_bytes(req_bytes, dt_us)
                return result

            multi_callable.__call__ = _traced_call
            return multi_callable
        return _patched_unary_unary

    def start(self):
        if self.active:
            return
        self.active = True

        # ── 1. Wrap torch.distributed ──
        td_names = {
            'all_reduce': 'all_reduce',
            'all_gather_into_tensor': 'all_gather_into_tensor',
            'all_gather': 'all_gather',
            'broadcast': 'broadcast',
            'reduce_scatter_tensor': 'reduce_scatter_tensor',
        }
        for attr_name, key in td_names.items():
            if hasattr(dist, attr_name):
                self.orig_td_funcs[key] = getattr(dist, attr_name)
                setattr(dist, attr_name, self._make_td_wrapper(self.orig_td_funcs[key]))

        # ── 2. Wrap gRPC (both abstract base AND concrete implementation) ──
        try:
            import grpc
            import grpc._channel

            self.orig_grpc_unary_unary = (
                grpc.Channel.unary_unary,
                grpc._channel.Channel.unary_unary,
            )

            grpc.Channel.unary_unary = self._setup_grpc_trace(self.orig_grpc_unary_unary[0])
            grpc._channel.Channel.unary_unary = self._setup_grpc_trace(self.orig_grpc_unary_unary[1])
        except Exception:
            pass

    def stop(self):
        if not self.active:
            return
        self.active = False

        restore_map = {
            'all_reduce': 'all_reduce',
            'all_gather_into_tensor': 'all_gather_into_tensor',
            'all_gather': 'all_gather',
            'broadcast': 'broadcast',
            'reduce_scatter_tensor': 'reduce_scatter_tensor',
        }
        for key, attr in restore_map.items():
            if key in self.orig_td_funcs:
                setattr(dist, attr, self.orig_td_funcs[key])

        if self.orig_grpc_unary_unary is not None:
            try:
                import grpc
                import grpc._channel
                grpc.Channel.unary_unary = self.orig_grpc_unary_unary[0]
                grpc._channel.Channel.unary_unary = self.orig_grpc_unary_unary[1]
            except Exception:
                pass

    def stats(self):
        torch.cuda.synchronize()
        return {
            "calls": self.comm_calls,
            "mb": self.comm_bytes / (1024 * 1024),
            "comm_ms": self.comm_cpu_us / 1000,
        }

    def _record_bytes(self, byte_count, cpu_time_us):
        self.comm_calls += 1
        self.comm_bytes += byte_count
        self.comm_cpu_us += cpu_time_us

    @staticmethod
    def gpu_mem(dev=0): return torch.cuda.max_memory_allocated(dev)/(1024*1024)
    @staticmethod
    def gpu_reserved(dev=0): return torch.cuda.memory_reserved(dev)/(1024*1024)
    @staticmethod
    def reset(): torch.cuda.reset_peak_memory_stats()
    @staticmethod
    def gpu_util(dev=None):
        try:
            import pynvml
            pynvml.nvmlInit()
            if dev is not None:
                return pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(dev)).gpu
            count = pynvml.nvmlDeviceGetCount()
            if count == 0: return -1
            return sum(pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu for i in range(count)) / count
        except Exception:
            return -1

# ===== weight-matrix rank Analyzer =====
class WeightRankAnalyzer:
    """
    Analyze the rank of weight matrices during training
    Monitoring objectives:
    1. Detect model degradation (rank reduction)
    2. Evaluate expressiveness (full rank vs. low rank)
    3. Identify signs of overfitting (a sudden rank drop)
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.rank_history = {}  # {step: {param_name: rank_info}}

    def get_last_block_ffn_target_patterns(self):
        """Returns the list of regular mode that matches only the FFN-related weight matrix in the last Block."""
        import re

        block_indices = []
        has_model_prefix = False
        has_plain_prefix = False

        for name, _ in self.model.named_parameters():
            m1 = re.match(r"^model\.blocks\.(\d+)\.mlp\.", name)
            if m1:
                block_indices.append(int(m1.group(1)))
                has_model_prefix = True
                continue

            m2 = re.match(r"^blocks\.(\d+)\.mlp\.", name)
            if m2:
                block_indices.append(int(m2.group(1)))
                has_plain_prefix = True

        if not block_indices:
            return None

        last_block_idx = max(block_indices)
        prefixes = []
        if has_model_prefix:
            prefixes.append(f"model.blocks.{last_block_idx}.mlp.")
        if has_plain_prefix:
            prefixes.append(f"blocks.{last_block_idx}.mlp.")

        return [rf"^{re.escape(prefix)}.*\.weight$" for prefix in prefixes]
        
    def compute_matrix_rank(self, weight_matrix, rtol=1e-5):
        """
        Calculate the numerical rank of a matrix
        
        Parameters:
            Weight_matrix: weight matrix (may be 2D or higher)
            rtol: Relative tolerance, used to determine whether singular values is 0
        
        Return:
            rank_info: {
                'rank': matrix rank,
                'full_rank': full-rank size,
                'rank_ratio': rank / full rank,
                'condition_number': condition number,
                'top_singular_values': top 10 singular values
            }
        """
        # Ensure calculation on CPU (avoiding GPU memory deficiencies)
        if weight_matrix.is_cuda:
            weight_matrix = weight_matrix.cpu()
        
        # Convert to float32 to improve numerical stability
        weight_matrix = weight_matrix.float()
        
        # If it's a high-dimensional, reshape is 2D.
        original_shape = weight_matrix.shape
        if len(original_shape) > 2:
            # For example (num_heads, seq_len, Hidden_dim) - (num_heads*seq_len, Hidden_dim)
            weight_matrix = weight_matrix.reshape(-1, original_shape[-1])
        
        try:
            # Calculate singular values decomposition
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            
            # Calculate rank as the number of nonzero singular values
            max_singular_value = S[0].item()
            tolerance = max_singular_value * max(weight_matrix.shape) * rtol
            rank = torch.sum(S > tolerance).item()
            
            # FULL rank Size
            full_rank = min(weight_matrix.shape)
            
            # CONDISION NUMBER: Max singular values/ Minimum non-zero singular values
            nonzero_singular_values = S[S > tolerance]
            if len(nonzero_singular_values) > 0:
                condition_number = (nonzero_singular_values[0] / nonzero_singular_values[-1]).item()
            else:
                condition_number = float('inf')
            
            # Top 10 singular values
            top_singular_values = S[:10].tolist()
            
            return {
                'rank': rank,
                'full_rank': full_rank,
                'rank_ratio': rank / full_rank if full_rank > 0 else 0.0,
                'condition_number': condition_number,
                'top_singular_values': top_singular_values,
                'shape': original_shape,
                'effective_shape': weight_matrix.shape
            }
        except Exception as e:
            print(f"⚠️ rank calculation failed: {e}")
            return None
    
    def analyze_model_ranks(self, global_step, target_layers=None, sample_rate=1.0):
        """
        Analyze the rank of all weight matrices in the model
        
        Parameters:
            Global_step: Current training steps
            Target_layers: Specifies the layer name to be analysed (regular expression list), and None means all layers
            sample_rate: sampling rate, avoid calculating all layers (range 0-1)
        
        Return:
            rank_stats: rank statistics
        """
        import re
        
        rank_data = {}
        print(f"\n{'='*80}")
        print(f"WEIGHT-matrix rank Analysis (step) {global_step})")
        print(f"{'='*80}")
        
        # Collect parameters to be analysed
        params_to_analyze = []
        for name, param in self.model.named_parameters():
            # Analysis of weight matrix only and above 2D (exceeding 1D parameters like bias)
            if param.dim() >= 2:
                # If target level is specified, filter
                if target_layers is not None:
                    if not any(re.search(pattern, name) for pattern in target_layers):
                        continue
                
                # Sample
                if np.random.rand() <= sample_rate:
                    params_to_analyze.append((name, param))
        
        print(f" Analysis {len(params_to_analyze)} weight matrices...")
        
        # Calculate the rank of each matrix
        for name, param in params_to_analyze:
            rank_info = self.compute_matrix_rank(param.data)
            if rank_info is not None:
                rank_data[name] = rank_info
        
        # Save to History
        self.rank_history[global_step] = rank_data
        
        # Calculate statistical information
        rank_stats = self._compute_rank_statistics(rank_data)
        
        return rank_stats
    
    def _compute_rank_statistics(self, rank_data):
        """Calculate statistical information for rank"""
        if not rank_data:
            return {}
        
        rank_ratios = [info['rank_ratio'] for info in rank_data.values()]
        condition_numbers = [info['condition_number'] for info in rank_data.values() 
                           if info['condition_number'] != float('inf')]
        
        stats = {
            'num_matrices': len(rank_data),
            'avg_rank_ratio': np.mean(rank_ratios),
            'min_rank_ratio': np.min(rank_ratios),
            'max_rank_ratio': np.max(rank_ratios),
            'std_rank_ratio': np.std(rank_ratios),
            'full_rank_count': sum(1 for r in rank_ratios if r >= 0.99),
            'low_rank_count': sum(1 for r in rank_ratios if r < 0.5),
        }
        
        if condition_numbers:
            stats['avg_condition_number'] = np.mean(condition_numbers)
            stats['max_condition_number'] = np.max(condition_numbers)
        
        return stats
    
    def print_rank_summary(self, global_step, top_n=10):
        """Print the rank Analysis Abstract"""
        if global_step not in self.rank_history:
            print("No rank data for this step")
            return
        
        rank_data = self.rank_history[global_step]
        if not rank_data:
            print("No rank data available for this step (perhaps not matched to target level)")
            return

        stats = self._compute_rank_statistics(rank_data)
        
        print(f"\nOverall statistics:")
        print(f"   Analysis of MARTIX: {stats['num_matrices']}")
        print(f"   Average rank RATIO: {stats['avg_rank_ratio']:.4f}")
        print(f"   rank-ratio range: [{stats['min_rank_ratio']:.4f}, {stats['max_rank_ratio']:.4f}]")
        print(f"   Full-rank matrices: {stats['full_rank_count']} ({stats['full_rank_count']/stats['num_matrices']*100:.1f}%)")
        print(f"   Low-rank matrices: {stats['low_rank_count']} (rank ratio <0.5)")
        
        if 'avg_condition_number' in stats:
            print(f"   Average condition number: {stats['avg_condition_number']:.2e}")
            print(f"   Max. CONDISION NUMBER: {stats['max_condition_number']:.2e}")
        
        # Show the lowest-rank matrices (possible signs of degradation)
        sorted_by_rank = sorted(rank_data.items(), key=lambda x: x[1]['rank_ratio'])
        print(f"\nrank Minimum {min(top_n, len(sorted_by_rank))} matrices:")
        for i, (name, info) in enumerate(sorted_by_rank[:top_n], 1):
            print(f"   {i}. {name}")
            print(f"       rank: {info['rank']}/{info['full_rank']} ({info['rank_ratio']:.4f})")
            print(f"      Shape: {info['shape']}")
            print(f"       condition number: {info['condition_number']:.2e}")
        
        # Show the highest-rank matrices
        print(f"\nrank THE HIGHEST {min(top_n, len(sorted_by_rank))} matrices:")
        for i, (name, info) in enumerate(reversed(sorted_by_rank[-top_n:]), 1):
            print(f"   {i}. {name}")
            print(f"       rank: {info['rank']}/{info['full_rank']} ({info['rank_ratio']:.4f})")
            print(f"      Shape: {info['shape']}")
        
        print(f"{'='*80}\n")
        
        return stats
    
    def save_rank_history(self, filepath):
        """Save rank History to File"""
        import json
        
        # Convert to Sequencable Format
        serializable_history = {}
        for step, rank_data in self.rank_history.items():
            serializable_history[step] = {}
            for name, info in rank_data.items():
                serializable_history[step][name] = {
                    'rank': info['rank'],
                    'full_rank': info['full_rank'],
                    'rank_ratio': info['rank_ratio'],
                    'condition_number': info['condition_number'],
                    'shape': list(info['shape']),
                    'top_singular_values': info['top_singular_values'][:5]  # Only the top five.
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"✅ rank History Saved: {filepath}")

# ===== Neuron activation Tracker =====
class NeuronActivationTracker:
    """
    Track which neurons each sequence activates in every layer
    Record Format: {step: {layer_name: {token_idx:} [activated_neuron_indexes}
    """
    
    def __init__(self, model, tokenizer=None, max_records=100, save_interval=10):
        """
        Args:
            model: The model to track
            tokenizer: tokenizer decode token ID to text
            max_records: how many step records can be kept (avoiding overflows from memory)
            Save_interval: Keep detailed records at every step
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_records = max_records
        self.save_interval = save_interval
        
        # Store activation records as {step: {layer_name: {token_idx: [neuron_indices]}}}
        self.activation_records = {}
        # Store token text: {step: [token_texts]
        self.token_texts = {}
        self.current_step = 0
        self.current_layer_activations = {}  # Temporarily store activations from the current forward pass
        self.current_input_ids = None  # Save input_ids of the current step temporarily
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activated neurons"""
        
        def forward_hook(module, input, output, layer_name):
            """Record which neurons are activated"""
            with torch.no_grad():
                # Process output
                if isinstance(output, torch.Tensor):
                    acts = output.detach()
                elif isinstance(output, tuple):
                    acts = output[0].detach() if len(output) > 0 else None
                else:
                    return
                
                if acts is None:
                    return
                
                # Reshaping as (batch_size* seq_len, num_neurons)
                original_shape = acts.shape
                if acts.dim() > 2:
                    acts = acts.reshape(-1, acts.shape[-1])
                elif acts.dim() == 1:
                    acts = acts.unsqueeze(0)
                
                # Find neurons activated by each token (activation > threshold)
                # acts.shape = (num_tokens, num_neurons)
                activated_mask = acts.abs() > 1e-8  # (num_tokens, num_neurons)
                
                # Record the indices of activated neurons for each token
                token_activations = {}
                for token_idx in range(acts.shape[0]):
                    activated_neurons = torch.nonzero(activated_mask[token_idx], as_tuple=False).squeeze(-1)
                    if activated_neurons.numel() > 0:
                        token_activations[token_idx] = activated_neurons.cpu().tolist()
                
                # Store activation records for the current layer
                if layer_name not in self.current_layer_activations:
                    self.current_layer_activations[layer_name] = {}
                self.current_layer_activations[layer_name] = token_activations
        
        # Registering books for each MandelbrotV1Layer of MoE specialists
        hook_count = 0
        for name, module in self.model.named_modules():
            if type(module).__name__ == 'MandelbrotV1Layer':
                if 'experts' in name or 'mlp.layers' in name:
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: forward_hook(m, i, o, n)
                    )
                    self.hooks.append(hook)
                    hook_count += 1
        
        if hook_count > 0:
            print(f"✅ Yes. {hook_count} layers for neuron activation tracking hooks")
    
    def start_new_step(self, step, input_ids=None, tokenizer_manager=None, layer_idx=0,use_dimensionality_reduction=True):
        """Start a new training step. Prepare the record.
        
        Args:
            Step: Current training steps
            input_ids: input token IDs (tensor) - can be local or global IDs
            Tokenizer_manager: Multilayer tokenizer Manager (optional)
            layer_idx: current layer index (default: 0, meaning input_ids are local IDs for Layer 0)
        """
        self.current_step = step
        self.current_layer_activations = {}
        self.current_input_ids = input_ids
        self.current_layer_idx = layer_idx  # Record the currently used layers
        
        # tokenizer and input_ids, decode token text if available
        if input_ids is not None:
            try:
                # Flatten input_ids and convert them to a list
                token_ids = input_ids.view(-1).cpu().tolist()
                token_texts = []
                
                # Deal with multi-layer Tokenetzer
                if tokenizer_manager is not None:
                    # Find out if input_ids are global or local.
                    # If layer_idx is specified (not None), the IDs are local
                    if layer_idx is not None:
                        # Use the specified layer to decode local ID
                        for idx, local_id in enumerate(token_ids):
                            try:
                                text = tokenizer_manager.tokenizers[layer_idx].decode([local_id], skip_special_tokens=False)
                                # Limit Text Length
                                if len(text) > 30:
                                    text = text[:30] + "..."
                                token_texts.append(f"[L{layer_idx}]{text}")
                            except Exception as e:
                                # Show detailed error information if decoding fails
                                if idx < 3:  # Only the top 3 token printing errors
                                    print(f"    ⚠️ Token {idx} Decoding failed: lid={local_id}, error={str(e)[:50]}")
                                token_texts.append(f"<L{layer_idx}:id:{local_id}>")
                    else:
                        # Use global decoding for tokenizer_manager (assuming global ID)
                        for idx, global_id in enumerate(token_ids):
                            try:
                                # Find out what level this global ID belongs to.
                                try:
                                    
                                    if use_dimensionality_reduction:
                                        layer_id = tokenizer_manager.get_layer_from_token_id(global_id)
                                    else:
                                        layer_id,_ = tokenizer_manager.get_old_layer_from_token_id(layer_idx,global_id,True)
                            
                                except Exception:
                                    layer_id = next(
                                        layer
                                        for layer, (start, end) in enumerate(tokenizer_manager.old_layer_ranges)
                                        if start <= global_id < end
                                    )
                                    local_id = global_id - tokenizer_manager.old_offsets[layer_id]
                                # Use tokenizer decoding for the corresponding layer
                                text = tokenizer_manager.tokenizers[layer_id].decode([local_id], skip_special_tokens=False)
                                # Limit Text Length
                                if len(text) > 30:
                                    text = text[:30] + "..."
                                token_texts.append(f"[L{layer_id}]{text}")
                            except Exception as e:
                                # Show detailed error information if decoding fails
                                if idx < 3:  # Only the top 3 token printing errors
                                    print(f"    ⚠️ Token {idx} Decoding failed: gid={global_id}, error={str(e)[:50]}")
                                token_texts.append(f"<gid:{global_id}>")
                
                # Use single tokenizer
                elif self.tokenizer is not None:
                    for token_id in token_ids:
                        try:
                            text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            token_texts.append(text)
                        except:
                            token_texts.append(f"<id:{token_id}>")
                else:
                    # No tokenizer, only IDs
                    token_texts = [f"<id:{tid}>" for tid in token_ids]
                
                self.token_texts[step] = token_texts
            except Exception as e:
                print(f"⚠️ Error decoding token: {e}")
                self.token_texts[step] = None
    
    def finalize_step(self):
        """Finish Current Step Record"""
        if self.current_layer_activations:
            # Only keep detailed records at the specified interval
            if self.current_step % self.save_interval == 0:
                self.activation_records[self.current_step] = self.current_layer_activations.copy()
                
                # Limit the number of records to prevent overflows of memory
                if len(self.activation_records) > self.max_records:
                    # Remove Oldest Record
                    oldest_step = min(self.activation_records.keys())
                    del self.activation_records[oldest_step]
                    # Remove corresponding totoken_texts
                    if oldest_step in self.token_texts:
                        del self.token_texts[oldest_step]
        
        self.current_layer_activations = {}
        self.current_input_ids = None
    
    def get_step_summary(self, step):
        """Get the activation summary for a specified step"""
        if step not in self.activation_records:
            return None
        
        summary = {}
        for layer_name, token_acts in self.activation_records[step].items():
            total_activations = sum(len(neurons) for neurons in token_acts.values())
            num_tokens = len(token_acts)
            summary[layer_name] = {
                'num_tokens': num_tokens,
                'total_activations': total_activations,
                'avg_activations_per_token': total_activations / num_tokens if num_tokens > 0 else 0
            }
        return summary
    
    def print_activation_details(self, step, max_tokens=5, max_neurons=10, show_all_tokens=False, group_by_layer=False):
        """Print details of the specified step
        
        Args:
            Step: Number of training steps
            max_tokens: how many tokens per extert (if show_all_tokens=False)
            max_neurons: maximum neuron indices displayed per token
            Show_all_tokens: Whether to show all tokens of the extert layer (neglecting max_tokens limits)
            Group_by_layer: Whether token groups are shown by tokenizer layers (L0, L1, L2)
        """
        if step not in self.activation_records:
            print(f"⚠️ Steps {step} No activation records")
            return
        
        print(f"\n{'='*80}")
        print(f"Steps {step}  neuron activation details (tracking the input sequence)")
        
        # Check whether layer_idx information is available
        if hasattr(self, 'current_layer_idx') and self.current_layer_idx is not None:
            print(f"Note: Current data set uses Layer {self.current_layer_idx} tokenizer Encoding (PlID)")
            print(f"     Show all token as [L]{self.current_layer_idx}It's normal.")
        elif hasattr(self, 'current_layer_idx') and self.current_layer_idx is None:
            print(f"Note: The layer distribution of the model input sequence is shown")
            print(f"     model internals recode and use multilayer token (see bottom 'model internal token information)")
        
        print(f"{'='*80}")
        
        # Fetch token text (if any)
        token_texts = self.token_texts.get(step, None)
        
        for layer_name, token_acts in sorted(self.activation_records[step].items()):
            # Get the total number of neurons in this layer
            total_neurons = None
            if token_acts:
                try:
                    all_neurons = set()
                    for neuron_list in token_acts.values():
                        all_neurons.update(neuron_list)
                    if all_neurons:
                        total_neurons = max(all_neurons) + 1
                except:
                    pass
            
            print(f"\n📍 {layer_name}")
            if total_neurons:
                print(f"   Total neurons: {total_neurons}")
            print(f"   activated tokens: {len(token_acts)}")
            
            # If you need to group by layer
            if group_by_layer and token_texts:
                layer_groups = {}  # {layer_id: [(token_idx, neuron_list, text), ...]}
                for token_idx, neuron_list in sorted(token_acts.items()):
                    if token_idx < len(token_texts):
                        text = token_texts[token_idx]
                        # Extract layer marker [L2], [L1], [L2]
                        if text.startswith('[L') and ']' in text:
                            layer_id = text[2:text.index(']')]
                            if layer_id not in layer_groups:
                                layer_groups[layer_id] = []
                            layer_groups[layer_id].append((token_idx, neuron_list, text))
                
                # Print statistics for each layer
                for layer_id in sorted(layer_groups.keys()):
                    tokens_in_layer = layer_groups[layer_id]
                    
                    # Calculate neuron statistics for tokens in this layer
                    neuron_sets = [set(neuron_list) for _, neuron_list, _ in tokens_in_layer]
                    if neuron_sets:
                        # Compute the intersection (neurons activated by every token)
                        intersection = set.intersection(*neuron_sets)
                        # Compute the union (neurons activated by at least one token)
                        union = set.union(*neuron_sets)
                        # Average number of activated neurons
                        avg_active = sum(len(s) for s in neuron_sets) / len(neuron_sets)
                        
                        overlap_info = f" | intersection:{len(intersection)} The following are the examples:{len(union)} Average activation:{avg_active:.0f}"
                        if total_neurons:
                            activation_rate = avg_active / total_neurons * 100
                            overlap_rate = len(intersection) / avg_active * 100 if avg_active > 0 else 0
                            overlap_info += f" | activation rate:{activation_rate:.1f}% overlap rate:{overlap_rate:.1f}%"
                    else:
                        overlap_info = ""
                    
                    print(f"\n   [Layer {layer_id}] - {len(tokens_in_layer)} tokens{overlap_info}")
                    display_count = len(tokens_in_layer) if show_all_tokens else min(max_tokens, len(tokens_in_layer))
                    for i, (token_idx, neuron_list, text) in enumerate(tokens_in_layer[:display_count]):
                        neuron_str = str(neuron_list[:max_neurons])
                        if len(neuron_list) > max_neurons:
                            neuron_str = neuron_str[:-1] + f"- I'm not sure.{len(neuron_list)}]"
                        if total_neurons:
                            act_rate = len(neuron_list) / total_neurons * 100
                            print(f"      Token-{token_idx} ({repr(text)}): activation  {len(neuron_list)}/{total_neurons}({act_rate:.1f}%) {neuron_str}")
                        else:
                            print(f"      Token-{token_idx} ({repr(text)}): activated neurons  {neuron_str}")
                    if not show_all_tokens and len(tokens_in_layer) > max_tokens:
                        print(f"      And... {len(tokens_in_layer) - max_tokens} L{layer_id}Tokens")
            else:
                # Original display (intoken index order)
                display_count = len(token_acts) if show_all_tokens else max_tokens
                for i, (token_idx, neuron_list) in enumerate(sorted(token_acts.items())):
                    if i >= display_count:
                        print(f"   And... {len(token_acts) - display_count} tokens")
                        break
                    
                    # Fetch token text
                    token_text = ""
                    if token_texts and token_idx < len(token_texts):
                        text = token_texts[token_idx]
                        # Process special characters and long text
                        if len(text) > 20:
                            text = text[:20] + "..."
                        text = repr(text)  # Transscription Special Characters
                        token_text = f" ({text})"
                    
                    neuron_str = str(neuron_list[:max_neurons])
                    if len(neuron_list) > max_neurons:
                        neuron_str = neuron_str[:-1] + f"- I'm not sure.{len(neuron_list)}]"
                    print(f"   Token-{token_idx}{token_text}: activated neurons  {neuron_str}")
        
        print(f"{'='*80}\n")
    
    def save_to_file(self, filepath, step=None):
        """Save activation records to JSON files"""
        import json
        
        if step is not None:
            # Only record of specified steps
            data = {step: self.activation_records.get(step, {})}
        else:
            # Save all records
            data = self.activation_records
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ activation records saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save activation records: {e}")
    
    def get_neuron_activation_frequency(self, layer_name=None):
        """Calculate the activation frequency of each neuron"""
        neuron_freq = {}
        
        for step, layers in self.activation_records.items():
            for lname, token_acts in layers.items():
                if layer_name is not None and lname != layer_name:
                    continue
                
                if lname not in neuron_freq:
                    neuron_freq[lname] = {}
                
                # Count the activations of each neuron
                for token_idx, neuron_list in token_acts.items():
                    for neuron_idx in neuron_list:
                        if neuron_idx not in neuron_freq[lname]:
                            neuron_freq[lname][neuron_idx] = 0
                        neuron_freq[lname][neuron_idx] += 1
        
        return neuron_freq
    
    def remove_hooks(self):
        """Remove all hooks"""
        if not self.hooks:
            return
        try:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            print("✅ neuron activation tracking hooks removed")
        except Exception as e:
            print(f"⚠️ Error removing trackers: {e}")
    
    def clear_records(self):
        """Clear all records"""
        self.activation_records.clear()
        self.current_layer_activations = {}

# ===== dead-neuron Detector =====
class DeadNeuronChecker:
    """
    dead-neuron detection - built-in version
    Test content:
    1. neurons whose activations are always zero (completely dead)
    2. neurons with vanishing gradients
    """
    
    def __init__(self, model, threshold_dead=0.0):
        self.model = model
        self.threshold_dead = threshold_dead  # 0.0 means total death (activation rate is 0)
        
        # Store activation statistics for each layer
        self.activation_stats = {}
        self.gradient_stats = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registered forforward andbackwardbooks"""
        
        def forward_hook(module, input, output, layer_name):
            """Record activations"""
            with torch.no_grad():
                # Process output
                if isinstance(output, torch.Tensor):
                    acts = output.detach()
                elif isinstance(output, tuple):
                    acts = output[0].detach() if len(output) > 0 else None
                else:
                    return
                
                if acts is None:
                    return
                
                # Reshaping as (batch_size* seq_len, num_neurons)
                if acts.dim() > 2:
                    acts = acts.reshape(-1, acts.shape[-1])
                elif acts.dim() == 1:
                    acts = acts.unsqueeze(0)
                
                batch_samples = acts.shape[0]
                num_neurons = acts.shape[-1]
                
                # Initialization or updating of statistics
                if layer_name not in self.activation_stats:
                    self.activation_stats[layer_name] = {
                        'total_samples': 0,
                        'nonzero_count_per_neuron': torch.zeros(num_neurons, device=acts.device),
                        'activation_sum': torch.zeros(num_neurons, device=acts.device)
                    }
                
                stats = self.activation_stats[layer_name]
                nonzero_mask = (acts.abs() > 1e-8).float()
                # Use non-in-place ops and clone to avoid CUDA graph memory reuse issues
                stats['nonzero_count_per_neuron'] = stats['nonzero_count_per_neuron'] + nonzero_mask.sum(dim=0).clone()
                stats['activation_sum'] = stats['activation_sum'] + acts.abs().sum(dim=0).clone()
                stats['total_samples'] = stats['total_samples'] + batch_samples
        
        def backward_hook(module, grad_input, grad_output, layer_name):
            """Record gradient"""
            if hasattr(module, 'weight') and module.weight.grad is not None:
                with torch.no_grad():
                    grad_norm = module.weight.grad.norm().item()
                    if layer_name not in self.gradient_stats:
                        self.gradient_stats[layer_name] = {'grad_norm': []}
                    self.gradient_stats[layer_name]['grad_norm'].append(grad_norm)
        
        # Registering books for each MandelbrotV1Layer of MoE specialists
        hook_count = 0
        for name, module in self.model.named_modules():
            if type(module).__name__ == 'MandelbrotV1Layer':
                if 'experts' in name or 'mlp.layers' in name:
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: forward_hook(m, i, o, n)
                    )
                    self.hooks.append(hook)
                    hook_count += 1
                    
                    hook = module.register_full_backward_hook(
                        lambda m, gi, go, n=name: backward_hook(m, gi, go, n)
                    )
                    self.hooks.append(hook)
        
        if hook_count == 0:
            print("⚠️ Warning: No MandelbrotV1Layer found to register hooks")
        else:
            print(f"✅ Yes. {hook_count} layers for neuron monitoring hooks")
    
    def reset_stats(self):
        """Reset Statistical Information"""
        self.activation_stats.clear()
        self.gradient_stats.clear()
    
    def remove_hooks(self):
        """Remove all hooks"""
        if not self.hooks:
            return
        try:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            print("✅ Neuron detector hooks removed")
        except Exception as e:
            print(f"⚠️ Error removing hooks: {e}")
    
    def compute_dead_neuron_stats(self):
        """Calculate dead-neuron statistics"""
        results = {}
        
        for layer_name, stats in self.activation_stats.items():
            # Checking the validity of statistics
            total_samples = stats.get('total_samples', 0)
            if total_samples == 0:
                continue
            
            nonzero_per_neuron = stats.get('nonzero_count_per_neuron')
            if nonzero_per_neuron is None:
                continue
            
            try:
                nonzero_counts = nonzero_per_neuron.cpu().numpy()
                activation_sums = stats['activation_sum'].cpu().numpy()
            except Exception as e:
                print(f"⚠️ Processing {layer_name} Error: {e}")
                continue
            
            # Calculate activation rate
            activation_rate = nonzero_counts / max(total_samples, 1)
            
            # Identify completely dead neurons (activation rate <= threshold_dead)
            dead_mask = activation_rate <= self.threshold_dead
            
            num_neurons = len(activation_rate)
            num_dead = dead_mask.sum()
            
            results[layer_name] = {
                'num_neurons': num_neurons,
                'num_dead': int(num_dead),
                'dead_ratio': float(num_dead / num_neurons),
                'activation_rate_mean': float(activation_rate.mean()),
                'activation_rate_std': float(activation_rate.std()),
                'activation_rate_min': float(activation_rate.min()),
                'activation_rate_max': float(activation_rate.max()),
                'dead_neuron_indices': np.where(dead_mask)[0].tolist(),
                'avg_activation': float(activation_sums.mean()),
            }
        
        return results
    
    def print_summary(self):
        """Print the dead-neuron detection summary"""
        results = self.compute_dead_neuron_stats()
        
        if not results:
            print("⚠️ No activation statistics collected")
            return
        
        print("\n" + "="*80)
        print("dead-neuron detection Report")
        print("="*80)
        
        total_neurons = 0
        total_dead = 0
        
        for layer_name, stats in sorted(results.items()):
            total_neurons += stats['num_neurons']
            total_dead += stats['num_dead']
            
            status = "✓ Health"
            if stats['dead_ratio'] > 0.5:
                status = "✗ Serious"
            elif stats['dead_ratio'] > 0.2:
                status = "⚠ Warning"
            elif stats['dead_ratio'] > 0.05:
                status = "⚡ Attention."
            
            print(f"\n{status} {layer_name}")
            print(f"  Total neurons: {stats['num_neurons']}")
            print(f"  Dead neurons: {stats['num_dead']} ({stats['dead_ratio']*100:.2f}%)")
            print(f"   activation rate: {stats['activation_rate_mean']:.4f} ± {stats['activation_rate_std']:.4f}")
            print(f"  Average activation: {stats['avg_activation']:.4f}")
        
        print("\n" + "-"*80)
        print("Overall statistics:")
        print(f"  Total neurons: {total_neurons}")
        print(f"  Dead neurons: {total_dead} ({total_dead/total_neurons*100:.2f}%)")
        print(f"  Healthy neurons: {total_neurons - total_dead} "
              f"({(total_neurons - total_dead)/total_neurons*100:.2f}%)")
        print("="*80 + "\n")
        
        # gradient Statistics
        if self.gradient_stats:
            print("\ngradient Statistics:")
            for layer_name, grad_stats in sorted(self.gradient_stats.items()):
                if grad_stats.get('grad_norm'):
                    grad_norms = grad_stats['grad_norm']
                    print(f"  {layer_name}:")
                    print(f"    gradient Parameter: Average ={np.mean(grad_norms):.6f}, "
                          f"Min ={np.min(grad_norms):.6f}, max ={np.max(grad_norms):.6f}")
    
    def get_detailed_report(self):
        """Access to detailed reports"""
        return {
            'activation_stats': self.compute_dead_neuron_stats(),
            'gradient_stats': dict(self.gradient_stats),
        }

# ===== intrinsic dimension (ID) Analyzer =====
class IntrinsicDimensionAnalyzer:
    """
    Analyze the intrinsic dimension (ID) of each model block's FFN/MLP output before the residual connection
    Enhanced intrinsic-dimension analyzer supporting multiple analysis modes:
    
    Analysis mode:
    - ffn_output: analyze FFN/MLP output before the residual connection
    - Block_input: Analyse input for each Block
    - Block_output: Analyse output of each Block
    - both_io: Analyse input and output
    Goal: quantify the true complexity of useful information in activation space and determine whether the model has learned more fundamental patterns
    Method: PCA characteristic decomposition method, with the minimum dimension required to explain 95% of the variance
    """

    def __init__(
        self,
        model,
        device='auto',
        max_buffer_size=1000,
        *,
        energy_threshold: float = 0.95,
        mode: str = "ffn_output",  
    ):
        self.model = model
        self.device = device
        # Configure Parameters
        self.max_buffer_size_total = int(max_buffer_size)
        self.max_buffer_size_per_ffn = max(32, int(max_buffer_size))
        
        self.energy_threshold = float(energy_threshold)
        self.mode = mode  # "ffn_output", "block_input", "block_output", "both_io"
        
        #  activation data buffers 
        self.activation_buffers = {}  # FFN Output
        self.block_input_buffers = {}  # Block input
        self.block_output_buffers = {}  # Block Output
        
        # Historical data records
        self.id_history = {}  # {step: {block_idx: id_value}}
        self.effdim_history = {}  # ✅ Add this line.
        # Analysis of objectives
        self.targets = []  # FFN/MLP module
        self.block_input_targets = []  # Block input
        self.block_output_targets = []  # Block Output
        self.hook_handles = []

        blocks = None
        blocks_prefix = None
        if hasattr(model, 'model') and hasattr(model.model, 'blocks') and isinstance(model.model.blocks, (list, torch.nn.ModuleList)):
            blocks = model.model.blocks
            blocks_prefix = "model.blocks"
        elif hasattr(model, 'blocks') and isinstance(model.blocks, (list, torch.nn.ModuleList)):
            blocks = model.blocks
            blocks_prefix = "blocks"

        if blocks is None or len(blocks) == 0:
            print("⚠️ Could not detect mdel.blocks, ID analysis not available")
            return

        # Set the target according to the analysis mode
        if self.mode in ["ffn_output", "both_io"]:
            for idx, block in enumerate(blocks):
                if hasattr(block, 'mlp') and block.mlp is not None:
                    name = f"{blocks_prefix}.{idx}.mlp"
                    self.targets.append((idx, name, block.mlp))
        if self.mode in ["block_input", "both_io"]:
            for idx, block in enumerate(blocks):
                # Add Block Input Analysis Target (before Block handles)
                self.block_input_targets.append((idx, f"{blocks_prefix}.{idx}.input", block))
        if self.mode in ["block_output", "both_io"]:
            for idx, block in enumerate(blocks):
                # Add Block Output Analysis Target
                self.block_output_targets.append((idx, f"{blocks_prefix}.{idx}.output", block))
        
        if not self.targets and not self.block_input_targets and not self.block_output_targets:
            print("⚠️ No target module found for ID analysis")
            return
        
        # Print Profile Information
        print(f"🔍 intrinsic dimension analyser configuration:")
        print(f"  - Analysis mode: {self.mode}")
        print(f"  - Energy threshold: {self.energy_threshold}")
        print(f"  - Samples of each layer cache: {self.max_buffer_size_per_ffn}")
        
        if self.mode in ["ffn_output", "both_io"]:
            print(f"  - FFN's analytical objectives: {len(self.targets)} modules")
        
        if self.mode in ["block_input", "both_io"]:
            print(f"  - Block input analysis target: {len(self.block_input_targets)} modules")
        
        if self.mode in ["block_output", "both_io"]:
            print(f"  - Block output analysis target: {len(self.block_output_targets)} modules")
        
        print(f"🔍 intrinsic dimension Indicator: PCA@{self.energy_threshold:.2f}")

    def _extract_tensor_from_output(self, output):
        activation_tensor = None
        if isinstance(output, torch.Tensor):
            activation_tensor = output
        elif isinstance(output, dict):
            activation_tensor = output.get('last_hidden_state', None)
        elif isinstance(output, tuple):
            if len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    activation_tensor = first
                elif isinstance(first, (tuple, list)) and len(first) > 0 and isinstance(first[0], torch.Tensor):
                    activation_tensor = first[0]
        return activation_tensor if isinstance(activation_tensor, torch.Tensor) else None

    def _sample_rows(self, activation_tensor: torch.Tensor, max_add: int = 200):
        if activation_tensor.dim() == 3:
            _, _, hidden_dim = activation_tensor.shape
            flat_activations = activation_tensor.reshape(-1, hidden_dim)
            max_add = min(int(max_add), int(flat_activations.size(0)))
            if max_add <= 0:
                return None
            sampled = flat_activations[:max_add]
            return sampled
        if activation_tensor.dim() == 2:
            return activation_tensor
        return None

    def _make_ffn_activation_hook(self, block_idx: int):
        def hook(module, inputs, output):
            activation_tensor = self._extract_tensor_from_output(output)
            if not isinstance(activation_tensor, torch.Tensor):
                return None

            sampled = self._sample_rows(activation_tensor, max_add=200)
            if sampled is None:
                return None

            selected_activations = sampled.detach().to(dtype=torch.float32).cpu().numpy()
            buf = self.activation_buffers.setdefault(int(block_idx), [])
            cap = int(self.max_buffer_size_per_ffn or 0)
            if cap <= 0:
                return None
            if len(buf) < cap:
                buf.extend(selected_activations[: max(0, cap - len(buf))])
            else:
                num_replace = len(selected_activations)
                if num_replace < len(buf):
                    indices_to_replace = np.random.choice(len(buf), num_replace, replace=False)
                    for i, ridx in enumerate(indices_to_replace):
                        buf[int(ridx)] = selected_activations[i]
                else:
                    self.activation_buffers[int(block_idx)] = selected_activations[:cap].tolist()
            return None

        return hook

    def _make_residual_activation_hook(self, block_idx: int):
        def hook(module, inputs, output):
            activation_tensor = self._extract_tensor_from_output(output)
            if not isinstance(activation_tensor, torch.Tensor):
                return None
            sampled = self._sample_rows(activation_tensor, max_add=200)
            if sampled is None:
                return None
            selected_activations = sampled.detach().to(dtype=torch.float32).cpu().numpy()
            buf = self.residual_buffers.setdefault(int(block_idx), [])
            cap = int(self.max_buffer_size_per_ffn or 0)
            if cap <= 0:
                return None
            if len(buf) < cap:
                buf.extend(selected_activations[: max(0, cap - len(buf))])
            else:
                num_replace = len(selected_activations)
                if num_replace < len(buf):
                    indices_to_replace = np.random.choice(len(buf), num_replace, replace=False)
                    for i, ridx in enumerate(indices_to_replace):
                        buf[int(ridx)] = selected_activations[i]
                else:
                    self.residual_buffers[int(block_idx)] = selected_activations[:cap].tolist()
            return None

        return hook
    def _make_block_input_hook(self, block_idx: int):
        def hook(module, inputs, output):
            # Analyse input for Block (inputs [0] usually hidden_states)
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                activation_tensor = inputs[0]
                sampled = self._sample_rows(activation_tensor, max_add=200)
                if sampled is None:
                    return None
                selected_activations = sampled.detach().to(dtype=torch.float32).cpu().numpy()
                buf = self.block_input_buffers.setdefault(int(block_idx), [])
                cap = int(self.max_buffer_size_per_ffn or 0)
                if cap <= 0:
                    return None
                if len(buf) < cap:
                    buf.extend(selected_activations[: max(0, cap - len(buf))])
                else:
                    num_replace = len(selected_activations)
                    if num_replace < len(buf):
                        indices_to_replace = np.random.choice(len(buf), num_replace, replace=False)
                        for i, ridx in enumerate(indices_to_replace):
                            buf[int(ridx)] = selected_activations[i]
                    else:
                        self.block_input_buffers[int(block_idx)] = selected_activations[:cap].tolist()
            return None
        return hook
    def _make_block_output_hook(self, block_idx: int):
        def hook(module, inputs, output):
            activation_tensor = self._extract_tensor_from_output(output)
            if not isinstance(activation_tensor, torch.Tensor):
                return None
            sampled = self._sample_rows(activation_tensor, max_add=200)
            if sampled is None:
                return None
            selected_activations = sampled.detach().to(dtype=torch.float32).cpu().numpy()
            buf = self.block_output_buffers.setdefault(int(block_idx), [])
            cap = int(self.max_buffer_size_per_ffn or 0)
            if cap <= 0:
                return None
            if len(buf) < cap:
                buf.extend(selected_activations[: max(0, cap - len(buf))])
            else:
                num_replace = len(selected_activations)
                if num_replace < len(buf):
                    indices_to_replace = np.random.choice(len(buf), num_replace, replace=False)
                    for i, ridx in enumerate(indices_to_replace):
                        buf[int(ridx)] = selected_activations[i]
                else:
                    self.block_output_buffers[int(block_idx)] = selected_activations[:cap].tolist()
            return None
        return hook
    
    def register_hook(self):
        if not self.targets and not self.block_input_targets and not self.block_output_targets:
            print("⚠️ Target module not found for ID analysis")
            return None
        ok_ffn = 0
        ok_block_input = 0
        ok_block_output = 0
        
        # Register FFN Output hooks
        if self.mode in ["ffn_output", "both_io"]:
            # Register FFN Output hooks
            for block_idx, name, module in self.targets:
                try:
                    h = module.register_forward_hook(self._make_ffn_activation_hook(block_idx))
                    self.hook_handles.append(h)
                    ok_ffn += 1
                except Exception as e:
                    print(f"⚠️ Could not register hooks to FFN level '{name}': {e}")
        
        if self.mode in ["block_input", "both_io"]:
            # Sign Block into hooks
            for block_idx, name, module in self.block_input_targets:
                try:
                    h = module.register_forward_hook(self._make_block_input_hook(block_idx))
                    self.hook_handles.append(h)
                    ok_block_input += 1
                except Exception as e:
                    print(f"⚠️ Could not register hooks to Block Input Layer '{name}': {e}")
        
        if self.mode in ["block_output", "both_io"]:
            # Register Block Output hooks
            for block_idx, name, module in self.block_output_targets:
                try:
                    h = module.register_forward_hook(self._make_block_output_hook(block_idx))
                    self.hook_handles.append(h)
                    ok_block_output += 1
                except Exception as e:
                    print(f"⚠️ Could not register hooks to Block Output Layer '{name}': {e}")
        # Print detailed registration results
        print(f"\nhooks registered:")
        # Print registration results
        if self.mode in ["ffn_output", "both_io"]:
            print(f"  - FFN output: {ok_ffn}/{len(self.targets)} modules")
        
        if self.mode in ["block_input", "both_io"]:
            print(f"  - Block input: {ok_block_input}/{len(self.block_input_targets)} modules")
        
        if self.mode in ["block_output", "both_io"]:
            print(f"  - Block output: {ok_block_output}/{len(self.block_output_targets)} modules")
        
        print(f"  - Total hooks: {len(self.hook_handles)}")
        
        return self.hook_handles

    def _spectrum_s2(self, X: np.ndarray):
        """Return squared singular values (proportional to covariance eigenvalues).

        Scaling by (n-1) cancels out for PR/entropy, so we can use S^2 directly.
        """
        if len(X) < 2 or X.shape[1] < 2:
            return None
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - X.mean(axis=0, keepdims=True)
        try:
            _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            s2 = (S ** 2)
            return s2
        except np.linalg.LinAlgError:
            return None

    def compute_intrinsic_dimension(self, X, *, energy_threshold: float = None):
        if len(X) < 2 or X.shape[1] < 2:
            return 0
        thr = float(self.energy_threshold if energy_threshold is None else energy_threshold)
        s2 = self._spectrum_s2(X)
        if s2 is None or len(s2) == 0:
            return max(1, X.shape[1] // 2)
        total_var = float(np.sum(s2))
        if total_var <= 0:
            return 1
        cum_var_ratio = np.cumsum(s2) / total_var
        for k, r in enumerate(cum_var_ratio):
            if float(r) >= thr:
                return k + 1
        return int(len(s2))

    def analyze_id(self, global_step):
        # Check for analytical data
        has_ffn_data = bool(self.activation_buffers)
        has_block_input_data = bool(self.block_input_buffers)
        has_block_output_data = bool(self.block_output_buffers)
        
        if not any([has_ffn_data, has_block_input_data, has_block_output_data]):
            print(f" No activation DATA collected, skipping ID analysis (step=){global_step})")
            return None

        step_results = {}
        # Analysis based on analytical models
        if self.mode == "ffn_output":
            results = self._analyze_ffn_outputs()
            if results:
                step_results = results
                self._print_analysis_results(global_step, results, "FFN")
        
        elif self.mode == "block_input":
            results = self._analyze_block_inputs()
            if results:
                step_results = results
                self._print_analysis_results(global_step, results, "Block input")
        
        elif self.mode == "block_output":
            results = self._analyze_block_outputs()
            if results:
                step_results = results
                self._print_analysis_results(global_step, results, "Block Output")
        
        elif self.mode == "both_io":
            # Analyze the intrinsic dimensions of inputs and outputs simultaneously
            step_results = self._analyze_both_io(global_step)

        if not step_results:
            print(f" activation samples is insufficient, skips ID analysis (step=){global_step})")
            return None

        self.id_history[int(global_step)] = step_results
        
        # Clear buffers after analysis so the next window is fresh.
        self.activation_buffers.clear()
        # Also clear other buffers for the same window.
        try:
            self.residual_buffers.clear()
            self.block_input_buffers.clear()
            self.block_output_buffers.clear()
        except Exception:
            pass

        print(f"\n{'='*60}")
        print(f"intrinsic dimension(ID) Analysis (step) {global_step})")
        print(f"{'='*60}")
        # Print a small summary for readability.
        # Print FFN Output Results
        if "ffn_output" in step_results:
            ffn_vals = list(step_results["ffn_output"].values())
            print(f"FFN output (premature): {len(ffn_vals)}blocks")
            print(f"  ID range: [{min(ffn_vals)}, {max(ffn_vals)}], average: {float(np.mean(ffn_vals)):.2f}")
            for k in sorted(step_results["ffn_output"].keys())[:8]:
                print(f"  block{k}: ID={step_results['ffn_output'][k]}")
            if len(ffn_vals) > 8:
                print("  ...")
        
        # Print Block Input and Output Results
        if "block_input" in step_results and "block_output" in step_results:
            input_vals = list(step_results["block_input"].values())
            output_vals = list(step_results["block_output"].values())
            
            print(f"\nBlock Input/Export Comparison: {len(input_vals)}blocks")
            print(f"  Enter ID range: [{min(input_vals)}, {max(input_vals)}], average: {float(np.mean(input_vals)):.2f}")
            print(f"  Output ID range: [{min(output_vals)}, {max(output_vals)}], average: {float(np.mean(output_vals)):.2f}")
            
            # Calculate and show rate of change
            changes = []
            for block_idx in step_results["block_input"].keys():
                if block_idx in step_results["block_output"]:
                    input_id = step_results["block_input"][block_idx]
                    output_id = step_results["block_output"][block_idx]
                    change = (output_id - input_id) / max(input_id, 1) * 100
                    changes.append(change)
                    if block_idx < 8:  # Show details only of the first eight Blocks
                        print(f"  block{block_idx}: Enter ID ={input_id}, output ID={output_id}, Change ={change:+.1f}%")
            
            if changes:
                print(f"  Average rate of change: {np.mean(changes):+.1f}%")
            
            if len(input_vals) > 8:
                print("  ...")
        print(f"{'='*60}\n")

        return step_results
    def clear_buffers(self):
        """Clear all buffers and release memory"""
        self.activation_buffers.clear()
        self.block_input_buffers.clear()
        self.block_output_buffers.clear()
        print("✅ All intrinsic dimension analysis buffers emptied")
 
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # Remove information from printout mode
        if self.mode == "ffn_output":
            target_type = "FFN"
        elif self.mode == "block_input":
            target_type = "Block input"
        elif self.mode == "block_output":
            target_type = "Block Output"
        else:
            target_type = "Mixed"
            
        print(f"✅ {target_type}intrinsic dimension Analysis hooks removed")
 
    def switch_mode(self, new_mode):
        """Switch analysis mode"""
        if new_mode not in ["ffn_output", "block_input", "block_output", "both_io"]:
            print(f"⚠️ Unsupported analytical models: {new_mode}")
            return False
            
        # Remove the current hooks
        self.remove_hooks()
        
        # Empty buffers
        self.clear_buffers()
        
        # Update Mode
        old_mode = self.mode
        self.mode = new_mode
        
        # Re-introduction target
        self.targets.clear()
        self.block_input_targets.clear()
        self.block_output_targets.clear()
        
        # Recheck model structures and set targets
        blocks = None
        blocks_prefix = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks') and isinstance(self.model.model.blocks, (list, torch.nn.ModuleList)):
            blocks = self.model.model.blocks
            blocks_prefix = "model.blocks"
        elif hasattr(self.model, 'blocks') and isinstance(self.model.blocks, (list, torch.nn.ModuleList)):
            blocks = self.model.blocks
            blocks_prefix = "blocks"
 
        if blocks is None or len(blocks) == 0:
            print("⚠️ Could not detect model.blocks, mode switch failed")
            return False
 
        # Set the target according to the new analytical model
        if self.mode in ["ffn_output", "both_io"]:
            for idx, block in enumerate(blocks):
                if hasattr(block, 'mlp') and block.mlp is not None:
                    name = f"{blocks_prefix}.{idx}.mlp"
                    self.targets.append((idx, name, block.mlp))
 
        if self.mode in ["block_input", "both_io"]:
            for idx, block in enumerate(blocks):
                self.block_input_targets.append((idx, f"{blocks_prefix}.{idx}.input", block))
 
        if self.mode in ["block_output", "both_io"]:
            for idx, block in enumerate(blocks):
                self.block_output_targets.append((idx, f"{blocks_prefix}.{idx}.output", block))
 
        # Re-registration hooks
        self.register_hook()
        
        print(f"✅ The analysis model has been passed on '{old_mode}'Change to '{new_mode}'")
        return True
 
    def get_analysis_summary(self):
        """Get analytical summary information"""
        summary = {
            'mode': self.mode,
            'energy_threshold': self.energy_threshold,
            'max_buffer_size': self.max_buffer_size_per_ffn,
            'total_hooks': len(self.hook_handles),
            'id_history_steps': len(self.id_history)
        }
        
        # Add BUFFER Status Information
        buffer_status = {}
        for buffer_name, buffer in [
            ('ffn', self.activation_buffers),
            ('block_input', self.block_input_buffers),
            ('block_output', self.block_output_buffers)
        ]:
            if buffer:
                buffer_status[buffer_name] = {
                    'active_blocks': len(buffer),
                    'total_samples': sum(len(buf) for buf in buffer.values())
                }
        
        summary['buffer_status'] = buffer_status
        return summary

    def _analyze_ffn_outputs(self):
        """Analyse FFN output"""
        results = {}
        for block_idx, buf in self.activation_buffers.items():
            if buf is None or len(buf) < 10:
                continue
                
            X = np.array(buf)
            if X.ndim != 2:
                continue
                
            id_value = self.compute_intrinsic_dimension(X)
            results[int(block_idx)] = int(id_value)
        
        return results
    def _analyze_block_inputs(self):
        """Analyse Block Input"""
        results = {}
        for block_idx, buf in self.block_input_buffers.items():
            if buf is None or len(buf) < 10:
                continue
                
            X = np.array(buf)
            if X.ndim != 2:
                continue
            if X is not None:
                id_value = self.compute_intrinsic_dimension(X)
                results[int(block_idx)] = int(id_value)
            else:
                print(f"⚠️ Could not analyse Block {block_idx}input because the data format is invalid")
        
        return results
    def _analyze_block_outputs(self):
        """Analyse Block Output"""
        results = {}
        for block_idx, buf in self.block_output_buffers.items():
            if buf is None or len(buf) < 10:
                continue
                
            X = np.array(buf)
            if X.ndim != 2:
                continue
            if X is not None:
                id_value = self.compute_intrinsic_dimension(X)
                results[int(block_idx)] = int(id_value)
            else:
                print(f"⚠️ Could not analyse Block {block_idx}output because the data format is invalid")
        
        return results
    def _analyze_both_io(self, global_step):
        """Analyze the intrinsic dimensions of inputs and outputs simultaneously"""
        results = {}
        
        # Analyse Inputs
        input_results = self._analyze_block_inputs()
        if input_results:
            results["input"] = input_results
        
        # Analyse Output
        output_results = self._analyze_block_outputs()
        if output_results:
            results["output"] = output_results
        
        # Calculate input output comparison
        if input_results and output_results:
            comparison_results = {}
            for block_idx in input_results.keys():
                if block_idx in output_results:
                    input_id = input_results[block_idx]
                    output_id = output_results[block_idx]
                    
                    # Calculate dimension change rate
                    if input_id > 0:
                        change_rate = output_id / input_id
                    else:
                        change_rate = float('inf')
                    
                    comparison_results[block_idx] = {
                        'input_id': input_id,
                        'output_id': output_id,
                        'change_rate': change_rate,
                        'absolute_change': output_id - input_id
                    }
            
            if comparison_results:
                results["comparison"] = comparison_results
        
        # Print Details
        self._print_both_io_results(global_step, results)
        
        return results
    def _print_analysis_results(self, global_step, results, analysis_type):
        """Print Analysis"""
        if not results:
            return
            
        avg_id = np.mean(list(results.values()))
        print(f"\n📊 {analysis_type}intrinsic dimension(ID) Analysis (All Layers) (step) {global_step}):")
        print(f"  Average ID: {avg_id:.2f}")
        
        # Print ID values for each Block
        for block_idx, id_value in results.items():
            print(f"  Block {block_idx}: ID={id_value}")
    def _print_both_io_results(self, global_step, results):
        """Print Output Analysis"""
        print(f"\nBlock Input and Output intrinsic dimension (ID) Analysis (all layers) (step) {global_step})")
        
        if "input" in results:
            input_avg = np.mean(list(results["input"].values()))
            print(f"  Enter average ID: {input_avg:.2f}")
        
        if "output" in results:
            output_avg = np.mean(list(results["output"].values()))
            print(f"  Output average ID: {output_avg:.2f}")
        
        if "comparison" in results:
            print(f"  Dimension change analysis:")
            for block_idx, comparison in results["comparison"].items():
                trend = "↑Increase" if comparison['change_rate'] > 0 else "↓Reduction"
                print(f"    Block {block_idx}: Enter ID ={comparison['input_id']} → Output ID ={comparison['output_id']} ({trend} {abs(comparison['change_rate']):.2%})")
         # Clear buffers after analysis so the next window is fresh.
        self.activation_buffers.clear()
        # Also clear other buffers for the same window.
        try:
            self.block_input_buffers.clear()
            self.block_output_buffers.clear()
        except Exception as e:
            print(f"⚠️ Error while clearing Block for output buffers: {e}")
    def save_effdim_history(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.effdim_history, f, indent=2, ensure_ascii=False)
        print(f"✅ Effective dimension history saved: {filepath}")

    def save_history(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.id_history, f, indent=2, ensure_ascii=False)
        print(f"✅ ID History saved: {filepath}")

    def remove_hooks(self):
        if not self.hook_handles:
            return
        try:
            for h in self.hook_handles:
                h.remove()
            self.hook_handles.clear()
            print("✅ intrinsic dimension Analysis hooks removed")
        except Exception as e:
            print(f"⚠️ Error while removing intrinsic-dimension analysis hooks: {e}")

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


def get_cosine_schedule_with_warmup_and_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following a cosine curve after linear warmup.
    The minimum learning rate is determined by min_lr_ratio * initial_lr.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # +1 Avoid step 0 when lr=0 and make sure that the last step is made
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 1.0  # No decay:warmup after maintaining constant lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)  # Cursor on [0,1] to prevent negative values or transboundary movements
        cosine_factor = (math.cos(math.pi * progress) + 1) / 2
        return max(min_lr_ratio, cosine_factor)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

print(f"TOP import pid={os.getpid()}, __name__=__name__", flush=True)


def get_cosine_schedule_with_warmup_and_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following a cosine curve after linear warmup.
    The minimum learning rate is determined by min_lr_ratio * initial_lr.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # +1 Avoid step 0 when lr=0 and make sure that the last step is made
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 1.0  # No decay:warmup after maintaining constant lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)  # Cursor on [0,1] to prevent negative values or transboundary movements
        cosine_factor = (math.cos(math.pi * progress) + 1) / 2
        return max(min_lr_ratio, cosine_factor)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

print(f"TOP import pid={os.getpid()}, __name__=__name__", flush=True)

# ===== 1. Add Resource Monitoring Functions =====
def check_memory():
    """Check the system and GPU memory"""
    print(f"System memory: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().available / (1024**3):.1f} GB Available)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {allocated:.1f}/{total:.1f} GB ({allocated/total*100:.1f}%)")

# ===== 2. Initial Checks =====
if __name__ == "__main__":
    print("=== Initial Resource Status ===")
    check_memory()


def _collate_streaming_blocks(batch):
    """Custom collate for StreamingTextDataset.

    Avoids the default_collate shared-memory preallocation path in DataLoader
    worker processes (which can raise 'Trying to resize storage that is not resizable'
    on Windows).
    """
    # 1. Safe empty value judgement
    if batch is None or (isinstance(batch, (list, tuple)) and len(batch) == 0):
        return torch.empty(0, dtype=torch.long)
    
    # [New] If the catch is already Tensor, then directly returns
    if isinstance(batch, torch.Tensor):
        return batch
    # Ensure each sample is a plain contiguous CPU tensor.
    tensors = [torch.as_tensor(x, dtype=torch.long).contiguous() for x in batch]
    return torch.stack(tensors, dim=0)


def __collate_streaming_blocks(batch):
    """
    The batch merge function under the prefill mode.
    Compatibility with local mode (list of single samples) and server mode (loads of packed sheets).
    """
    # 1. Safe empty value judgement
    if batch is None or (isinstance(batch, (list, tuple)) and len(batch) == 0):
        return torch.empty(0, dtype=torch.long)
    
    # [New] If the catch is already Tensor, then directly returns
    if isinstance(batch, torch.Tensor):
        return batch
        
    # Local mode: batch is a list of 1D sheets, execute stack pack
    if isinstance(batch, (list, tuple)) and isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
        
    # 4. Apologies
    raise TypeError(f"Unsupported batch type in collate_fn: {type(batch)}")

def _collate_first_item(batch):
    """Picklable collate function (Windows num_workers>0 safe).

    Used for ActivationCacheDataset where each yielded item is already a full sample dict.
    """
    return batch[0]


# ------------------------
# Helpers
# ------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ------------------------
# DDP Utilities
# ------------------------

def is_main_process() -> bool:
    """Return True if this is the main (rank 0) process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    """Return the total number of processes in the current distributed group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def ddp_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce the mean of a tensor across all processes."""
    if get_world_size() <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def setup_ddp(local_rank: int, world_size: int):
    """Initialize the NCCL process group for single-machine multi-GPU DDP.

    IMPORTANT: torch.cuda.set_device() MUST be called BEFORE dist.init_process_group()
    to prevent NCCL from binding all processes to GPU 0.
    """
    if world_size <= 1:
        print("[DDP] world_size <= 1, skipping distributed initialization")
        return

    # CRITICAL: bind this process to its GPU BEFORE initializing the process group.
    # Otherwise NCCL may create CUDA contexts on GPU 0 for all ranks, causing OOM.
    torch.cuda.set_device(local_rank)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
    )
    if is_main_process():
        print(f"[DDP] Initialized: rank={get_rank()}, world_size={get_world_size()}, "
              f"local_rank={local_rank}, device=cuda:{local_rank}")


def cleanup_ddp():
    """Clean up the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ------------------------
# Training
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU GPT-like pretraining (PyTorch + transformers)")

    # data
    p.add_argument("--train_folder", type=str,default="../MandelbrotV1/data/10GB", help="Folder containing .txt files (recursively)")
    p.add_argument("--max_train_lines", type=int, default=None, help="max number of lines to read (for debugging)")
    # model / tokenizer

    _default_tokenizer_dir = os.path.join(os.path.dirname(__file__), "trained_tokenizer_test3")
    p.add_argument("--tokenizer_name", type=str, default="../MandelbrotV1/tok_26layers4/26_tokenizers", help="Tokenizer to use (default: gpt2)")
    p.add_argument("--vocab_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=512, help="context length / block size")  # From 64 to 1024.
    p.add_argument("--pad_mode", type=bool, default=True, help="Whether to use fill mode") 
    # architecture (default aims ~1B params; verify with printed count)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--add_special_tokens", type=bool, default=True, help="Whether to add special tokens to the tokenizer")
    p.add_argument("--layer_idx", type=int, default=0, help="Index of the layer to use for tokenizer")

    # optimization (client-side)
    p.add_argument("--per_device_train_batch_size", type=int, default=50000)  # Increase batch_size to make full use of GPU memory
    p.add_argument("--prefetch_size", type=int, default=4, help="Number of batches to prefetch in the background")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_epochs", type=int, default=-1, help="Maximum number of training epochs")
    p.add_argument("--min_loss", type=float, default=0.01, help="Minimum loss threshold to stop training")

    p.add_argument("--gradient_accumulation_steps", type=int, default=64)  # Lower accumulation accelerates the rotation
    p.add_argument("--learning_rate", type=float, default=1e-4)  # Improved stability in lower learning rates
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
   
    
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of initial learning rate")

    # runtime
    p.add_argument("--output_dir", type=str, default="../MandelbrotV1/outputs_incremental_pretraining2", help="Output directory for checkpoints")
    
    p.add_argument("--logging_steps", type=int, default=100)  # More frequent log output for monitoring
    p.add_argument("--logging_epochs", type=int, default=100, help="Log metrics every N epochs (like logging_steps)")
    p.add_argument(
        "--debug_grad_stats",
        action="store_true",
        help="Print gradient coverage and Top-K grad norms at logging steps (captured before optimizer.step).",
    )
    p.add_argument(
        "--debug_grad_topk",
        type=int,
        default=8,
        help="Top-K trainable parameters to print by L2 grad norm when --debug_grad_stats is enabled.",
    )
    
    # neurons monitor parameters
    p.add_argument("--enable_neuron_check", type=bool, default=False, help="Enable dead-neuron detection")
    p.add_argument("--neuron_check_steps", type=int, default=50000, help="Frequency of dead-neuron detection (checked every step)")
    p.add_argument("--neuron_track_steps", type=int, default=50000, help="Frequency of neuron activation tracking (save and print every step)")
    p.add_argument("--rank_check_steps", type=int, default=50000, help="weight-matrix rank analysis frequency in steps (recommended: the same as neuron_check_steps)")
    
    # Performance monitor switch
    p.add_argument("--enable_gpu_monitor", type=bool, default=False, help="Enable GPU UTILZATION surveillance (pynvml required)")
    p.add_argument("--enable_rank_analysis",type=bool, default=False, help="Enable WEIGHT-matrix rank analysis")

    # Add ID to analyse the parameters
    p.add_argument("--id_check_steps", type=int, default=10000, help="intrinsic dimension Analysis Frequency (Analysis per Step)")
    p.add_argument("--enable_id_analysis",type=bool, default=False, help="Enable intrinsic dimension analysis")
    p.add_argument("--id_max_samples", type=int, default=1664, help="Maximum activation samples cached per FFN for ID analysis (larger is more accurate but slower)")
    p.add_argument("--id_energy_threshold", type=float, default=0.95, help="PCA-ID explain the threshold for the margin (default 0.95; higher sensitive)")
    p.add_argument("--id_mode", type=str, choices=["ffn_output", "block_input", "block_output", "both_io"], default="both_io", 
                help="Intrinsic-dimension analysis mode: ffn_output (FFN output), block_input (block input), block_output (block output), both_io (analyze both input and output)")


    # (CKA/GC indicator removed)
    
    p.add_argument("--seed", type=int, default=42)

    # DDP (Distributed Data Parallel) arguments
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for distributed training. Set by torchrun/launch utility automatically.")
    p.add_argument("--no_ddp_find_unused", type=bool, default=False,
                   help="Disable find_unused_parameters in DDP. "
                        "By default it is enabled because this MoE model has conditional expert routing.")
    # FSDP (ZeRO-3 / ZeRO-2) arguments
    p.add_argument("--use_fsdp", action="store_true", default=False,
                   help="Use Fully Sharded Data Parallelism (ZeRO-3) instead of DDP. "
                        "Shards model parameters, gradients, and optimizer states across GPUs.")
    p.add_argument("--fsdp_sharding_strategy", type=str, default="full",
                   choices=["full", "grad", "none"],
                   help="FSDP sharding: full=ZeRO-3 (params+grads+opt), "
                        "grad=ZeRO-2 (grads+opt), none=DDP-like (no sharding)")
    p.add_argument("--torch_compile", type=bool, default=False,
                   help="Enable torch.compile for model acceleration (experimental, may slow down MoE models).")
    
    #(server-side)
    p.add_argument("--no_gradient_checkpointing", type=bool, default=True,
                   help="Disable gradient checkpointing to improve training speed (at the cost of higher memory usage).")
    p.add_argument("--use_compute_comm_overlap", type=bool, default=True, help="Whether to calculate overlap of communications")
    p.add_argument("--use_triple", type=bool, default=True, help="Whether to use a triad")
    p.add_argument("--mem_threshold", type=float, default=0.9, help="System threshold (default 0.85)")
    p.add_argument("--vram_threshold", type=float, default=0.85, help="GPU memory threshold (default 0.85)")
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every N epochs when --save_strategy epoch")

    p.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Training precision mode: fp32 | fp16 | bf16",
    )
    p.add_argument(
        "--save_weight_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Dtype used to save model weights on disk (safetensors/bin).",
    )
    p.add_argument("--fp16", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--bf16", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from (overrides --model_name_or_path)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--by_eos", type=bool, default = None, help="Whether to cut by EOS")
    p.add_argument("--is_Distribution", type=bool, default=True, help="Distribute incremental pre-training")
    p.add_argument("--distribution_Type", type=str, default="middle_Server", choices=["server", "client","middle_Server","local","incremental_pretrain_full","incremental_pretrain"], help="Distributed training type (is_Distribution=True Effective): server, middle_Server, clit Local training type (is_Distribution=False Effective): local, incremental_pretain_full, incremental_pretrain")
    p.add_argument("--worker_id", type=str, default="server_1", help="Worker id for distributed training")
    p.add_argument("--model_name_or_path", type=str, default="../MandelbrotV1/configs/middle2", help="Path to pre-trained model or model identifier from huggingface.co/models")
    # Incremental init / expansion
    p.add_argument(
        "--init_from_dir",
        type=str,
        #default="../MandelbrotV1/outputs_france_reduction_test/final_test",
        default=None,
        help="Initialize weights from an existing saved model directory (supports sharded safetensors index / model.safetensors / pytorch_model.bin).",
    )

    # Optional: overlay/partial load weights (load a single block/FFN without loading the whole checkpoint)
    p.add_argument(
        "--overlay_from_dir",
        type=str,
        default=None,
        help=(
            "Overlay (partial) weights from another checkpoint directory after the model is created. "
            "Works best with HF sharded safetensors (model.safetensors.index.json + shards/*). "
            "Use --overlay_block_idxs / --overlay_ffn / --overlay_prefix to select what to load."
        ),
    )
    p.add_argument(
        "--overlay_prefix",
        type=str,
        action="append",
        default=None,
        help=(
            "State_dict key prefix to overlay. Can be repeated. Example: --overlay_prefix model.blocks.2. "
            "(This is the most general option.)"
        ),
    )
    p.add_argument(
        "--overlay_block_idxs",
        type=str,
        default=None,
        help="Comma-separated block indices to overlay, e.g. '2' or '1,2,3'. Expands to prefix 'model.blocks.<i>.'.",
    )
    p.add_argument(
        "--overlay_ffn",
        type=str,
        default=None,
        help=(
            "Comma-separated FFN spec(s) 'B:L' to overlay one plain-MLP FFN layer in block B at layer index L. "
            "Expands to prefixes 'model.blocks.B.mlp.layers.L.' and 'model.blocks.B.mlp.layernorm.L.'."
        ),
    )
    p.add_argument(
        "--overlay_include_model_norm",
        action="store_true",
        help="When using --overlay_ffn, also overlay 'model.norm.<L>.' (if present in checkpoint).",
    )
    p.add_argument(
        "--add_blocks",
        type=int,
        default=2,
        help="Increase config.num_hidden_blocks by N and randomly initialize the new blocks (then you can freeze old params and train only new blocks).",
    )
    p.add_argument(
        "--add_ffn_layers",
        type=int,
        default=0,
        help="Increase config.layer_num by N (and extend layer_dimdiff) to append new FFN layers to every block. New layers are randomly initialized.",
    )
    p.add_argument(
        "--train_only_new_blocks",
        action="store_true",
        help="When --add_blocks>0, freeze all params except newly added block(s) and their corresponding model.norm entries.",
    )
    p.add_argument(
        "--train_only_new_ffn",
        action="store_true",
        help="When --add_ffn_layers>0, freeze all params except the newly added FFN layer(s) (mlp.layers/L + mlp.layernorm/L) in a chosen block and its corresponding model.norm entries.",
    )
    p.add_argument(
        "--new_ffn_block_idx",
        type=int,
        default=-1,
        help="Which block to train the newly added FFN layer(s) in when using --train_only_new_ffn (default -1 means last block).",
    )
    # Incremental / block-wise training
    p.add_argument(
        "--train_only_block_idx",
        type=int,
        default=None,
        help="Freeze all parameters except model.blocks[block_idx] and its corresponding model.norm entries.",
    )
    p.add_argument(
        "--train_only_last_block",
        action="store_true",
        help="Shortcut: set --train_only_block_idx to the last block after loading the model.",
    )
    p.add_argument(
        "--also_train_lm_head",
        action="store_true",
        help="When using --train_only_*, also keep lm_head trainable (default: frozen).",
    )
    p.add_argument(
        "--also_train_embeddings",
        action="store_true",
        help="When using --train_only_*, also keep embeddings trainable (default: frozen).",
    )
    p.add_argument(
        "--freeze_embeddings",
        type=bool,
        default=True
        ,
        help="When using --train_only_*, also keep embeddings trainable (default: frozen).",
    )
    p.add_argument("--dataloader_drop_last", type=bool, default=False, help="Whether to drop the last incomplete batch")
    p.add_argument(
        "--train_only_embeddings_and_lm_head",
        action="store_true",
        help="Warmup mode for vocab growth: freeze everything except embeddings (+ embedding_manager units) and lm_head.",
    )

    p.add_argument(
        "--paired_train_folder",
        type=str,
        default=None,
        help="Optional: when generating activation cache, use paired corpora A(train_folder)->B(paired_train_folder) by zipping lines. Cache is computed from A, but labels/input_ids are taken from B.",
    )

    p.add_argument(
        "--cache_dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Activation cache dtype on disk.",
    )
    p.add_argument(
        "--cache_max_batches",
        type=int,
        default=None,
        help="Limit number of cached batches (debug).",
    )

    # Block-output cache (incremental block training)
    p.add_argument(
        "--generate_block_output_cache",
        type=bool,
        default=False,
        help="Generate block-output cache files (block_hidden_states after a chosen block) for incremental block training.",
    )
    p.add_argument(
        "--use_block_output_cache",
        type=bool,
        default=False,
        help="Train using block-output cache instead of token->forward. Requires cache files generated beforehand.",
    )
    p.add_argument(
        "--block_cache_dir",
        type=str,
        default="../MandelbrotV1/outputs_france_reduction_incremental_pretraining/block_cache",
        help="Directory to write/read block-output cache .pt files.",
    )
    p.add_argument(
        "--cache_after_block_idx",
        type=int,
        default=1,
        help="When generating block-output cache: capture hidden_states after this block idx.",
    )
    p.add_argument(
        "--cache_start_block_idx",
        type=int,
        default=22,
        help="When training from block-output cache: start executing from this block idx (usually cache_after_block_idx+1).",
    )
    p.add_argument(
        "--cache_end_block_idx",
        type=int,
        default=25,
        help="When training from block-output cache: stop executing at this block idx (usually cache_after_block_idx).",
    )
    p.add_argument(
        "--cache_start_expert_id",
        type=int,
        default=0,
        help="When training from block-output cache: start executing from this block idx (usually cache_after_block_idx+1).",
    )
    p.add_argument(
        "--cache_end_expert_id",
        type=int,
        default=0,
        help="When training from block-output cache: stop executing at this block idx (usually cache_after_block_idx).",
    )
    p.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    p.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading")
    p.add_argument("--save_strategy", type=str, default="steps", help="Save strategy: 'steps' or 'epoch'")
    p.add_argument("--save_total_limit", type=int, default=30, help="Maximum number of checkpoints to keep")
    p.add_argument("--report_to", type=str, default="tensorboard", help="The integration to report the results and logs to.")
    p.add_argument("--log_dir", type=str, default="../MandelbrotV1/outputs_france_reduction_incremental_pretraining/runs/logs_251111", help="Tensorboard log dir")
    p.add_argument("--save_images", action="store_true", help="Whether to save input images to TensorBoard")
    p.add_argument("--log_interval", type=int, default=3, help="Interval (in epochs) to log images to TensorBoard")

    p.add_argument(
        "--render_torchviz_graphs",
        type=bool,
        default=False,
        help="Render torchviz computation graphs (writes model_computation_graph.* and model_graph*.png). Disabled by default to avoid generating many files.",
    )

    p.add_argument("--early_patience", type=int, default=-1, help="Early stopping patience")
    p.add_argument("--early_min_delta", type=float, default=1e-4, help="Early stopping min delta")
    p.add_argument("--time_budget_hours", type=float, default=None, help="Time budget in hours")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    # Optional: Mandelbrot HF-compatible sharded safetensors (per-block / per-FFN)
    p.add_argument(
        "--save_shards",
        type=bool,
        default=True,
        help="Also save a HF-compatible sharded safetensors checkpoint (model.safetensors.index.json + shards/*).",
    )
    p.add_argument(
        "--shards_per_ffn",
        type=bool,
        default=True,
        help="Shard MLP as block+layer (model.blocks.B.mlp.layers.L.*) instead of one file per block.",
    )
    p.add_argument(
        "--shards_moe_per_expert",
        type=bool,
        default=True,
        help="Shard MoE layers per expert (blockXXXX_expertYYY) and split gate/shared_experts into their own shards.",
    )
    p.add_argument(
        "--shards_subdir",
        type=str,
        default="shards",
        help="Subdirectory name under each save_dir to place shard files.",
    )
    p.add_argument(
        "--also_save_block_shards",
        type=bool,
        default=False,
        help=(
            "When saving HF sharded safetensors with --shards_per_ffn, also write an additional per-block sharded "
            "checkpoint under <save_dir>/block_shards (with its own model.safetensors.index.json + shards/*)."
        ),
    )
    p.add_argument(
        "--block_shards_dirname",
        type=str,
        default="block_shards",
        help="Directory name under each save_dir to store the additional per-block sharded checkpoint.",
    )
    p.add_argument("--grad_norm_threshold", type=float, default=None, help="Gradient norm threshold for explosion detection")
    p.add_argument("--eval_every_steps", type=int, default=20, help="Evaluate every N steps")
    p.add_argument("--valid_loader", type=DataLoader, default=None, help="Path to validation data folder")
    
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Device to run training on: 'auto' (default), 'cuda', or 'cpu'")

    return p.parse_args()





def _to_cache_dtype(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "fp16":
        return x.to(dtype=torch.float16)
    return x.to(dtype=torch.float32)

def save_checkpoint(model, optimizer, epoch, global_step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": time.time()
    }, path)
    print(f"[CKPT] saved {path}")

def load_checkpoint(model, optimizer, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

class EarlyStopper:
    def __init__(self, mode='min', patience=5, min_delta=1e-4, restore_best=True):
        assert mode in ('min', 'max')
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best = restore_best
        self.best = None
        self.bad_epochs = 0
        self.best_ckpt = None

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)

    def step(self, current_metric, model=None, optimizer=None, epoch=None, global_step=None, ckpt_dir=None,save_checkpoint_func=None,load_checkpoint_func=None  ):
        """Return True indicates that training should be discontinued"""
        if self._is_better(current_metric, self.best):
            self.best = current_metric
            self.bad_epochs = 0
            if model is not None and ckpt_dir is not None:  # Update Output Directory to Save Best model
                path = os.path.join(ckpt_dir, f"best_epoch{epoch}_step{global_step}.pt")
                if save_checkpoint_func:
                    save_checkpoint_func(model, optimizer, epoch, global_step, path)
                # else:
                #     save_checkpoint(model, optimizer, epoch, global_step, path)
                self.best_ckpt = path
        else:
            self.bad_epochs += 1
        if self.bad_epochs > self.patience and self.patience>0 :
            # Trigger pre-stop.
            if self.restore_best and self.best_ckpt is not None and model is not None:
                print(f"[EarlyStopper] restore best ckpt: {self.best_ckpt}")
                if load_checkpoint_func:
                    load_checkpoint_func(model, optimizer, self.best_ckpt)
                # else:
                #     load_checkpoint(model, optimizer, self.best_ckpt)
            return True
        return False

def should_stop_hard(epoch, global_step, start_time,loss, min_loss=None, max_epochs=None, max_steps=None, max_seconds=None):
    if max_epochs is not None and max_epochs >= 0 and epoch >= max_epochs:
        return True, f"reach max_epochs ({epoch} >= {max_epochs})"
    if min_loss is not None and min_loss >= 0 and min_loss >= loss:
        return True, f"reach min_loss ({min_loss} >= {loss})"
    if max_steps is not None and max_steps >= 0 and global_step >= max_steps:
        return True, f"reach max_steps ({global_step} >= {max_steps})"
    if max_seconds is not None and (time.time() - start_time) >= max_seconds:
        return True, f"reach time_budget ({time.time() - start_time:.0f}s >= {max_seconds}s)"
    return False, ""


def _compute_shifted_ce_loss_from_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute causal-LM CE loss from logits/labels with standard shift."""
    if not torch.is_tensor(labels) or not torch.is_tensor(logits):
        raise TypeError("labels/logits must be tensors")

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    if labels.dim() != 2 or logits.dim() != 3:
        raise ValueError(f"Unexpected labels/logits shape: labels={tuple(labels.shape)}, logits={tuple(logits.shape)}")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = int(shift_logits.size(-1))
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    n = min(int(shift_logits.size(0)), int(shift_labels.size(0)))
    if n <= 0:
        raise ValueError("Empty shifted logits/labels after alignment")

    shift_logits = shift_logits[:n]
    shift_labels = shift_labels[:n].to(shift_logits.device)

    return torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)


def _normalize_or_recompute_loss(loss, outputs, labels) -> torch.Tensor:
    """Ensure loss is a tensor without changing the model's original loss semantics."""
    if torch.is_tensor(loss):
        return loss

    if isinstance(loss, (float, int)):
        if torch.is_tensor(labels):
            ref_device = labels.device
        else:
            ref_device = torch.device("cpu")
        return torch.tensor(float(loss), dtype=torch.float32, device=ref_device)

    raise RuntimeError(
        f"Cannot normalize loss. type(loss)={type(loss)}, "
        f"has_labels={torch.is_tensor(labels)}"
    )


def _make_grad_debug_snapshot(model, *, topk: int = 8, target_prefixes: Optional[List[str]] = None) -> dict:
    """Collect per-step gradient coverage stats for trainable parameters.

    Notes:
    - This should be called after scaler.unscale_(optimizer) to report unscaled gradient norms.
    - target_prefixes is optional and can narrow diagnostics to specific modules
      (e.g., newly added blocks).
    """

    model_ref = model.module if hasattr(model, "module") else model
    topk = max(int(topk), 0)
    prefixes = list(target_prefixes) if target_prefixes else None

    all_stats = {"trainable": 0, "grad_present": 0, "grad_nonzero": 0}
    target_stats = {"trainable": 0, "grad_present": 0, "grad_nonzero": 0} if prefixes else None

    all_rows = []
    target_rows = []

    for name, p in model_ref.named_parameters():
        if not p.requires_grad:
            continue

        grad = p.grad
        grad_norm = 0.0
        has_grad = grad is not None
        if has_grad:
            grad_norm = float(grad.detach().norm(2).item())

        all_stats["trainable"] += 1
        if has_grad:
            all_stats["grad_present"] += 1
        if has_grad and (not math.isfinite(grad_norm) or grad_norm > 0.0):
            all_stats["grad_nonzero"] += 1
        all_rows.append((grad_norm, name))

        if target_stats is not None and any(name.startswith(pref) for pref in prefixes):
            target_stats["trainable"] += 1
            if has_grad:
                target_stats["grad_present"] += 1
            if has_grad and (not math.isfinite(grad_norm) or grad_norm > 0.0):
                target_stats["grad_nonzero"] += 1
            target_rows.append((grad_norm, name))

    all_topk = sorted(all_rows, key=lambda x: x[0], reverse=True)[:topk] if topk > 0 else []
    target_topk = sorted(target_rows, key=lambda x: x[0], reverse=True)[:topk] if topk > 0 else []

    return {
        "all": all_stats,
        "target": target_stats,
        "all_topk": all_topk,
        "target_topk": target_topk,
    }

def collect_and_reset_losses(model, device):
    """
    loss to collect and reset MoE experts (only selected experts > 0)
    returns: (avg_non_zero_count_ratio, avg_indep_los_ratio)
    Ratio = cumulative value / total number of neurons
    """
    total_non_zero_count_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_indep_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_each_neurons = 0  # Total ecoch_neurons
    total_indep_neurons = 0  # Total indep_neurons
    
    if model.training:
        # Lightweight FFN-only mode does not have the full transformer structure.
        if not hasattr(model, "model") or not hasattr(model.model, "blocks"):
            return 0.0, 0.0

        # Collect non_zero_count_loss and the total neuron count from MoE experts
        layer_count = 0
        moe_block_count = 0
        for b_idx, block in enumerate(model.model.blocks):
            # Process MandelbrotV1MoE (with attributes)
            if hasattr(block.mlp, 'experts'):
                moe_block_count += 1
                for e_idx, expert in enumerate(block.mlp.experts):
                    if expert is not None and hasattr(expert, 'layers'):
                        for l_idx, layer in enumerate(expert.layers):
                            # Number of neurons (each_neurons)
                            
                            
                            if hasattr(layer, 'non_zero_count_loss'):
                                layer_loss = layer.non_zero_count_loss.detach().to(device)
                                # Collect > 0 layers (expert selected by MoE)
                                if layer_loss.item() > 0:
                                    if hasattr(layer, 'each_neurons'):
                                        neurons_in_layer = layer.each_neurons
                                        total_each_neurons += neurons_in_layer
                                    total_non_zero_count_loss = total_non_zero_count_loss + layer_loss
                                    layer_count += 1
                                    if global_vars["is_Loss_Log"] and False:
                                        print(f"[Collect] block={b_idx} expert={e_idx} layer={l_idx} non_zero_count={layer_loss.item():.0f}")
            else:
                # Process normal FFN (no experts properties)
                if hasattr(block.mlp, 'layers'):
                    for l_idx, layer in enumerate(block.mlp.layers):
                        if hasattr(layer, 'non_zero_count_loss'):
                            layer_loss = layer.non_zero_count_loss.detach().to(device)
                            # Collect > 0 layers (although normal FFN should not have non-zero loss, we also count)
                            if layer_loss.item() > 0:
                                if hasattr(layer, 'each_neurons'):
                                    neurons_in_layer = layer.each_neurons
                                    total_each_neurons += neurons_in_layer
                                total_non_zero_count_loss = total_non_zero_count_loss + layer_loss
                                layer_count += 1
                                if global_vars["is_Loss_Log"] and False:
                                    print(f"[Collect] block={b_idx} layer={l_idx} non_zero_count={layer_loss.item():.0f}")
        # Calculate ratio
        avg_non_zero_count_ratio = (total_non_zero_count_loss / layer_count).item() if layer_count > 0 else -1
        
        if global_vars["is_Loss_Log"] and layer_count > 0 and False:
            print(f"[Total] Collected {layer_count} MoE layers, total_each_neurons={total_each_neurons}, "
                  f"non_zero_count={total_non_zero_count_loss.item():.0f}, "
                  f"ratio={avg_non_zero_count_ratio:.4f}")
        
        # Indep_loss for collecting MoE extensions
        layer_count_indep = 0
        for b_idx, block in enumerate(model.model.blocks):
            # Handle MandelbrotV1MoE
            if hasattr(block.mlp, 'experts'):
                for e_idx, expert in enumerate(block.mlp.experts):
                    if expert is not None and hasattr(expert, 'layers'):
                        for l_idx, layer in enumerate(expert.layers):
                            # Number of neurons (indep_neurons)
    
                            if hasattr(layer, 'indep_loss'):
                                layer_loss = layer.indep_loss.detach().to(device)
                                # Collect only > 0 layers (expert selected by MoE)
                                if layer_loss.item() > 0:
                                    if hasattr(layer, 'indep_neurons'):
                                        neurons_in_layer = layer.indep_neurons
                                        total_indep_neurons += neurons_in_layer
                                    total_indep_loss = total_indep_loss + layer_loss
                                    layer_count_indep += 1
                                    if global_vars["is_Loss_Log"] and False:
                                        print(f"[Collect] block={b_idx} expert={e_idx} layer={l_idx} moe indep_loss={layer_loss.item():.0f}")
            else:
                # Process normal FFN (no experts properties)
                if hasattr(block.mlp, 'layers'):
                    for l_idx, layer in enumerate(block.mlp.layers):
                        if hasattr(layer, 'indep_loss'):
                            layer_loss = layer.indep_loss.detach().to(device)
                            # Only collect > 0 layers (although normal FFN should not have non-zero loss, we also count)
                            if layer_loss.item() > 0:
                                if hasattr(layer, 'indep_neurons'):
                                    neurons_in_layer = layer.indep_neurons
                                    total_indep_neurons += neurons_in_layer
                                total_indep_loss = total_indep_loss + layer_loss
                                layer_count_indep += 1
                                if global_vars["is_Loss_Log"] and False:
                                    print(f"[Collect] block={b_idx} layer={l_idx} indep_loss={layer_loss.item():.0f}")
        # Calculate ratio
        avg_indep_loss_ratio = (total_indep_loss / layer_count_indep).item() if layer_count_indep > 0 else -1
        
        if global_vars["is_Loss_Log"] and layer_count_indep > 0 and False:
            print(f"[Total] Collected {layer_count_indep} MoE layers, total_indep_neurons={total_indep_neurons}, "
                  f"indep_loss={total_indep_loss.item():.0f}, "
                  f"ratio={avg_indep_loss_ratio:.4f}")
        
        # Reset moE extensions to loss 0 (avoids accumulation in next round)
        reset_non_zero_count_loss_and_indep_loss(model)
        
    else:
        avg_non_zero_count_ratio = 0.0
        avg_indep_loss_ratio = 0.0
    
    return avg_non_zero_count_ratio, avg_indep_loss_ratio

def reset_non_zero_count_loss_and_indep_loss(model):
    # Reset moE extensions to loss 0 (avoids accumulation in next round)
    for b_idx, block in enumerate(model.model.blocks):
        if hasattr(block.mlp, 'experts'):
            for expert in block.mlp.experts:
                if expert is not None and hasattr(expert, 'layers'):
                    for layer in expert.layers:
                        if hasattr(layer, 'non_zero_count_loss'):
                            layer.non_zero_count_loss = torch.zeros(
                                1, dtype=layer.non_zero_count_loss.dtype,
                                device=layer.non_zero_count_loss.device
                            )
                        if hasattr(layer, 'indep_loss'):
                            layer.indep_loss  = torch.zeros(
                                1, dtype=layer.indep_loss.dtype,
                                device=layer.indep_loss.device
                            )
        else:
            for l_idx, layer in enumerate(block.mlp.layers):
                if hasattr(layer, 'non_zero_count_loss'):
                        layer.non_zero_count_loss = torch.zeros(
                            1, dtype=layer.non_zero_count_loss.dtype,
                            device=layer.non_zero_count_loss.device
                        )
                if hasattr(layer, 'indep_loss'):
                    layer.indep_loss  = torch.zeros(
                        1, dtype=layer.indep_loss.dtype,
                        device=layer.indep_loss.device
                    )
    avg_non_zero_count_ratio = 0.0
    avg_indep_loss_ratio = 0.0
    
    return avg_non_zero_count_ratio, avg_indep_loss_ratio

def get_inputs(global_ids)-> torch.Tensor:

    if isinstance(global_ids, list):
        # Extract all values of tansor
        processed_ids = []
        for item in global_ids:
            if isinstance(item, torch.Tensor):
                processed_ids.extend(item.tolist())
            else:
                processed_ids.append(item)
        input_ids = torch.tensor(processed_ids, dtype=torch.long).view(1, -1)
    else:
        input_ids = global_ids.view(1, -1)
    return input_ids

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    running_loss = 0.0
    raw_loss_window_total = 0
    loss_list = [] 

    # ===== DDP initialization (must happen before any CUDA usage) =====
    # Detect local_rank: torchrun sets LOCAL_RANK env var and passes --local_rank
    ddp_enabled = False
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        local_rank = int(args.local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    if world_size <= 0:
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    if world_size <= 0 and torch.cuda.device_count() > 1 and local_rank >= 0:
        world_size = torch.cuda.device_count()

    # CRITICAL: bind process to its GPU BEFORE any CUDA operation or NCCL init.
    # This prevents all ranks from defaulting to GPU 0.
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if local_rank >= 0 and world_size > 1:
        ddp_enabled = True
        if not args.is_Distribution:
            setup_ddp(local_rank, world_size)

    if is_main_process():
        print(f"[DDP] ddp_enabled={ddp_enabled}, local_rank={local_rank}, world_size={world_size}, "
              f"torch.cuda.device_count={torch.cuda.device_count()}")

    print("[ImportCheck] tokenization_MandelbrotV1_fast:", tokenizer_module.__file__)
    print("[ImportCheck] modeling_MandelbrotV1:", modeling_module.__file__)


    log_dir=os.path.join(args.output_dir, "logs_251104")
    os.makedirs(log_dir, exist_ok=True)

    # Only rank 0 writes to TensorBoard; others get a no-op writer
    class _NoopWriter:
        """No-op TensorBoard writer for non-rank-0 processes."""
        def add_scalar(self, *args, **kwargs): pass
        def add_scalars(self, *args, **kwargs): pass
        def add_histogram(self, *args, **kwargs): pass
        def add_hparams(self, *args, **kwargs): pass
        def add_figure(self, *args, **kwargs): pass
        def add_image(self, *args, **kwargs): pass
        def add_text(self, *args, **kwargs): pass
        def close(self): pass
        def flush(self): pass

    if is_main_process():
        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        tb_writer = _NoopWriter() 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _tb_add_histogram_safe(
        writer: SummaryWriter,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        *,
        max_elements: int = 200_000,
        max_bins: int = 64,
    ) -> None:
        """Write histogram to TensorBoard without exploding CPU RAM.

        Torch's histogram summary converts values to a float64 numpy array.
        For large tensors this can allocate hundreds of MB and crash.
        We downsample to a bounded number of elements before logging.
        """
        if writer is None or tensor is None:
            return
        try:
            t = tensor.detach()
            if t.is_sparse:
                t = t.to_dense()
            t = t.reshape(-1)
            n = int(t.numel())
            if n <= 0:
                return

            if n > max_elements:
                stride = int(math.ceil(n / max_elements))
                t = t[::stride]

            # Keep the tensor small before TensorBoard's internal float64 cast.
            t = t.float()
            if t.is_cuda:
                t = t.cpu()

            writer.add_histogram(tag, t, step, max_bins=max_bins)
        except (MemoryError, RuntimeError, ValueError) as e:
            # Don't crash training just because TensorBoard logging failed.
            if step == 0 or step % 1000 == 0:
                print(f"[TB] Skip histogram '{tag}' at step {step}: {type(e).__name__}: {e}")

    # device
    if ddp_enabled:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("Requested device 'cuda' but CUDA is not available; falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    requested_precision = str(args.precision).lower()

    if args.fp16 and args.bf16:
        raise ValueError("--fp16 and --bf16 are mutually exclusive; choose only one")
    if args.fp16:
        print("Warning: --fp16 is deprecated, use --precision fp16")
        requested_precision = "fp16"
    if args.bf16:
        print("Warning: --bf16 is deprecated, use --precision bf16")
        requested_precision = "bf16"

    if requested_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {requested_precision}")

    if requested_precision in {"fp16", "bf16"} and device.type != "cuda":
        print(f"Warning: --precision {requested_precision} requires CUDA; falling back to fp32")
        requested_precision = "fp32"

    if requested_precision == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
        print("Warning: current CUDA device does not support bf16; falling back to fp32")
        requested_precision = "fp32"

    use_fp16 = requested_precision == "fp16"
    use_bf16 = requested_precision == "bf16"
    amp_enabled = bool(use_fp16 or use_bf16)

    if use_fp16:
        amp_dtype = torch.float16
    elif use_bf16:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    precision_mode = requested_precision
    autocast_kwargs = {
        "device_type": "cuda",
        "enabled": amp_enabled,
        "dtype": amp_dtype,
    }

    save_weight_dtype_mode = str(args.save_weight_dtype).lower()
    if save_weight_dtype_mode not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported save_weight_dtype: {save_weight_dtype_mode}")
    save_weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[save_weight_dtype_mode]

    print(f"Precision mode: {precision_mode} (AMP {'enabled' if amp_enabled else 'disabled'})")
    print(f"Save weight dtype on disk: {save_weight_dtype}")

    # seed (add rank offset so each GPU sees a different sequence)
    base_seed = int(args.seed) + get_rank()
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    # Select one of the following options:
    #############################################
    # 1) Specifies model and multilayer root directory
    #############################################
    # Model_name point to the weight/ configure the directory
    

    final_dir = os.path.join(args.output_dir, "final_test")


    # Load multilayer manager (core)
    tokenizer_manager = MandelbrotV1TokenizerManager.from_pretrained(
        pretrained_path=args.tokenizer_name,
        layer_subfolder_prefix="layer",
        num_layers=None,
    )

    MandelbrotV1TokenizerManager.tokenizer_manager = tokenizer_manager
    print("[TokenizerManager] layer info:", tokenizer_manager.get_layer_info())

    # tokenizer

    # Option 2: If it is an absolute path
    # # Windows Path Example
    # # Linux Path Example

    # Option 3: If the HuggingFace cache path is used
    # args.model_name_or_path = "C:/Users/yourusername/.cache/huggingface/hub/models--deepseek-ai--MandelbrotV1"

    # ===== 3. Optimized tokenizer Loading =====
    print("\n=== Loading Tokenizer ===")
    try:
        layer_idx=args.layer_idx if args.layer_idx>=0 else tokenizer_manager.num_layers-1
        tokenizer=tokenizer_manager.tokenizers[layer_idx]   
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.model_name_or_path, 
        #     trust_remote_code=True,
        # #cache_dir=None, # Use default cache
        #     #resume_download=False,
        #     local_files_only=True,
        #     use_fast=True,
        # )
        # '''
        # tokenizer= MandelbrotV1TokenizerFast.from_pretrained(
        #                                                 args.model_name_or_path,
        #                                                 tokenizer_file=args.model_name_or_path+"/tokenizer.json",
        #                                                 trust_remote_code=True,
        # cache_dir=None, # Use default cache
        #                                                 resume_download=False,
        #                                                 local_files_only=True,
        # )
        # '''
        
        # messages = [
        #     {"role": "user", "content": "say hello"}
        # ]
        # input_ids = tokenizer.apply_chat_template(
        #     messages, 
        #     add_generation_prompt=True, 
        #     return_tensors=True
        # )
        if tokenizer.eos_token_id is None:
            # Ensure EOS exists
            tokenizer.add_special_tokens({"eos_token": ""})
        #eos_id = tokenizer.eos_token_id
        eos_id = None
        
        # print("input_ids:",input_ids)        
        # print("input_chat:",tokenizer.decode(input_ids[0]))
        
        # Make sure you enter the right device.
        print("✅ tokenizer loaded successfully")
        #exit(1)
    except Exception as e:
        print(f"❌ tokenizer loading failed: {e}")
        exit(1)

    # ===== 4. Create Quantization Configuration =====
    def create_quantization_config(bits=8):
        """Create Quantified Configuration"""
        if bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            return None

    # ===== 5. Optimized model Loading (Most Important) =====
    print("\n=== Loading model ===")
    model = None

    def use_block_output_cache(batch):
        # batch is a dict loaded from block cache file
        if not isinstance(batch, dict) or ("block_hidden_states" not in batch):
            keys = sorted(batch.keys()) if isinstance(batch, dict) else [type(batch).__name__]
            raise ValueError(
                "--use_block_output_cache expects block-cache files containing keys "
                "['input_ids', 'labels', 'block_hidden_states']. "
                f"Got keys={keys}. Check that --block_cache_dir points to the correct cache directory."
            )
        input_ids = batch["input_ids"].clone().to(device, non_blocking=True)
        labels = batch["labels"].clone().to(device, non_blocking=True)
        cache_block_hidden = batch["block_hidden_states"].clone().to(device, non_blocking=True)
        cache_block_hidden.requires_grad = True
        cache_after_block_idx = batch["cache_after_block_idx"]
        if not bool(getattr(config, "use_dimensionality_reduction", True)):
            labels, cache_selected_layer = _normalize_old_vocab_cache_labels(input_ids, labels,use_dimensionality_reduction)
        else:
            cache_selected_layer = int(batch.get("selected_layer", layer_idx))

        return input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer

    def use_block_cache_for_generation(input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer,capture=None,capture_after_block_idx=None):
  
        model_ref = model.module if hasattr(model, "module") else model

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        use_dimensionality_reduction=bool(getattr(config, "use_dimensionality_reduction", True))
        if not use_dimensionality_reduction:
            model_ref.last_layer_idx = int(cache_selected_layer)
            model_ref.bMax_frequency = False
        model_ref.model.load_block_out_cacth(cache_block_hidden,cache_after_block_idx)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=False,
            activation_capture=capture,
            capture_after_block_idx=capture_after_block_idx,
            layer_idx=int(cache_selected_layer),
        )
        
        return outputs,capture

    # Check whether there are trained weight files under the root directory (pytorch_model.bin / model. safetensors et al.)
    # Where to initialize weights/config from.
    init_dir = str(args.init_from_dir) if args.init_from_dir else final_dir

    possible_weight_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "tf_model.h5",
    ]
    existing_weight_file = None
    for fname in possible_weight_files:
        fpath = os.path.join(init_dir, fname)
        if os.path.isfile(fpath):
            existing_weight_file = fpath
            break

    if existing_weight_file is None:
        print("⚠️ No training weight files have been found, and random initialization of weights (from config) will be used.")
        print("   If you want to use it from_pretained, please train/save: model.save_pretained (`<dir>)")

    # Always load config from init_dir if available, otherwise from model_name_or_path
    config_source = init_dir if os.path.isdir(init_dir) else args.model_name_or_path
    config = MandelbrotV1Config.from_pretrained(config_source, trust_remote_code=True, local_files_only=True)
    config.vocab_size = tokenizer_manager.total_vocab_size

    # Track base sizes for incremental expansion.
    base_num_blocks = int(getattr(config, "num_hidden_blocks", 0) or 0)
    if base_num_blocks <= 0:
        base_num_blocks = int(getattr(config, "num_hidden_layers", 1) or 1)

    base_layer_num = int(getattr(config, "layer_num", 0) or 0)
    if base_layer_num <= 0:
        # Fallback: infer from layer_dimdiff if present.
        _ld = getattr(config, "layer_dimdiff", None)
        if isinstance(_ld, dict):
            base_layer_num = len(_ld)
        else:
            base_layer_num = 3

    # Try different loading strategies
    loading_strategies = [
        ("float32", None),
        ("8bit quantification", create_quantization_config(8)),
        ("4bit quantification", create_quantization_config(4))
    ]

    # Incremental expansion path: build model from (expanded) config then load base weights with strict=False.
    incremental_add_blocks = int(args.add_blocks or 0)
    incremental_add_ffn_layers = int(args.add_ffn_layers or 0)
    is_incremental_expand = (incremental_add_blocks > 0) or (incremental_add_ffn_layers > 0)
    # When performing incremental expansion or when using/generating block/activation caches,
    # we avoid writing full model files (model.safetensors/pytorch_model.bin) and trainer_state.pt.
    should_skip_full_checkpoint = bool(
        args.use_block_output_cache
        or args.generate_block_output_cache
        or getattr(config, "incremental_pretraining", False)
        or True
    )

    
# ==================================================================
# Step 1: [configuration level] Process incremental extension logic (Expand Config)
# ==================================================================
    if existing_weight_file is not None and args.use_block_output_cache and is_incremental_expand:
        if incremental_add_blocks > 0:
            config.num_hidden_blocks = base_num_blocks + incremental_add_blocks

        if incremental_add_ffn_layers > 0:
            new_layer_num = base_layer_num + incremental_add_ffn_layers
            ldiff = getattr(config, "layer_dimdiff", None)
            if not isinstance(ldiff, dict):
                ldiff = {str(i): 2 for i in range(base_layer_num)}
            else:
                ldiff = {str(k): int(v) for k, v in ldiff.items()}
                
            last_val = int(ldiff.get(str(base_layer_num - 1), 2)) if base_layer_num > 0 else 2
            for i in range(base_layer_num, new_layer_num):
                ldiff[str(i)] = last_val
                
            config.layer_num = new_layer_num
            config.layer_dimdiff = ldiff
        
        args.is_Distribution=False, 

        args.distribution_Type="incremental_pretrain"

        print(f"[Incremental] Expanding Model Structure | "
            f"blocks: {base_num_blocks} -> {config.num_hidden_blocks} | "
            f"layers: {base_layer_num} -> {config.layer_num}")


    # ==================================================================
    # Step 2: Harmonise Block Cache Verifying and Initialization Parameter Preparation
    # ==================================================================
    
    start_block_idx = int(args.cache_start_block_idx)
    end_block_idx = int(args.cache_end_block_idx)

    start_expert_id = int(args.cache_start_expert_id)
    end_expert_id = int(args.cache_end_expert_id)

    if args.generate_block_output_cache or args.use_block_output_cache:
        total_blocks = int(getattr(config, "num_hidden_blocks", 0) or 0)
        if not (0 <= start_block_idx < total_blocks):
            raise ValueError(f"train_block_from_cache start block out of range: {start_block_idx}")
        
        print("[LightweightBlockCache] model structure created")
        print(f"  Created blocks: {list(range(total_blocks))}")
        print(f"  Base checkpoint blocks present: {list(range(base_num_blocks))}")


    # ==================================================================
    # Step 3: Select loading according to condition (retain the original load policy cycle)
    # ==================================================================
    if existing_weight_file is not None:
        # Encapsulate a lazy load function and avoid rereading disks every time you try again
       
        def _load_base_sd(_init_dir):
            base_sd = None
            try:
                from modeling_MandelbrotV1 import MandelbrotV1ShardManager
                base_sd = MandelbrotV1ShardManager.load_state_dict_from_dir(_init_dir)
            except Exception:
                bin_path = os.path.join(_init_dir, "pytorch_model.bin")
                if os.path.isfile(bin_path):
                    base_sd = torch.load(bin_path, map_location="cpu")
                else:
                    raise FileNotFoundError("No valid weights found in init_dir.")
            return base_sd

        # Try different loading strategies / quantifying only when weight files exist
        for strategy_name, quant_config in loading_strategies:
            print(f"\nTry {strategy_name} loading...")
            try:
                # Clears the remnants of the last failed round
                if model is not None:
                    model = None
                    torch.cuda.empty_cache()
                    gc.collect()

                # ✅ Core changes: use the native tectonic function for the case of model directly
                # Customised business parameters safely enter _init_, never through any black box
                model = MandelbrotV1ForCausalLM(
                    config=config, 
                    start_block_idx=start_block_idx, 
                    end_block_idx=end_block_idx,
                    start_expert_id=start_expert_id,
                    end_expert_id=end_expert_id,
                    is_Distribution=args.is_Distribution, 
                    distribution_Type=args.distribution_Type
                )

                # ✅ Core changes: manually inject state_disc into model
                sd = _load_base_sd(init_dir)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                
                if len(unexpected) > 0:
                    print(f"[{strategy_name}] Unexpected keys: {len(unexpected)}")
                if len(missing) > 0:
                    print(f"[{strategy_name}] Missing keys (Expected if expanding): {len(missing)}")
                    
                # TODO: If your quantification or device_map needs to be effective in the original mode,
                # This is where you can use the accelerate library Dispach_model (model, device_map="auto")
                # Or simply.to.
                
                print(f"✅ {strategy_name} model loaded successfully (using existing weights: {os.path.basename(existing_weight_file)})")
                break  # Loading successful, jumping out of the policy cycle
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ {strategy_name} loading failed: insufficient GPU memory; trying the next strategy...")
                    continue
                else:
                    print(f"❌ {strategy_name} Loading failed: {e}")
                    continue
            
            except Exception as e:
                print(f"❌ {strategy_name} Loading failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if model is None:
            print("❌ All loading strategies failed.")
            exit(1)

    else:
        # No pre-training weights, direct random initialization
        model = MandelbrotV1ForCausalLM(
            config=config, 
            start_block_idx=start_block_idx,
            end_block_idx=end_block_idx,
            start_expert_id=start_expert_id,
            end_expert_id=end_expert_id,
            is_Distribution=args.is_Distribution,
            distribution_Type=args.distribution_Type,
        )
        print("✅ Use random initialization model (no pre-training weights)")


    # ==================================================================
    # Step 4: move to target device in a uniform manner
    # ==================================================================
    if torch.cuda.is_available():
        if args.use_fsdp:
            # FSDP manage devices on its own, model must stay in CPU before packaging, otherwise when the flatten parameter is available
            print(f"✅ FSDP will handle device placement (model kept on CPU before wrapping)")
        else:
            _target_device = torch.device(f"cuda:{local_rank}") if ddp_enabled else torch.device("cuda")
            model = model.to(_target_device)
            if is_main_process():
                print(f"✅ model moved to {_target_device}")
    else:
        print("⚠️ CUDA is not available, using CPU training")
        

        # for name, param in model.named_parameters():
        #     if 'weight' in name and param.dim() > 1:
        # If 'embeding' not in name: # Embedded layers usually use other initializations
        #             nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        #     elif 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'norm' in name:
        #             nn.init.constant_(param, 1)

        # Move model to the selected device
        try:
            model = model.to(device)
            print(f" Model moved to {device}")
        except Exception as e:
            print(f" Failed to move model to {device}: {e}. Falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)

    # Optional: overlay partial weights (single block / single FFN / arbitrary prefixes)
    if args.overlay_from_dir:
        from modeling_MandelbrotV1 import MandelbrotV1ShardManager

        # BitsAndBytes quantized models don't reliably support load_state_dict overlays.
        if bool(getattr(model, "is_loaded_in_8bit", False)) or bool(getattr(model, "is_loaded_in_4bit", False)):
            raise RuntimeError(
                "--overlay_from_dir is not supported with 8bit/4bit quantized loading. "
                "Please load the model without bitsandbytes quantization if you need overlay."
            )

        def _parse_int_csv(s: str):
            items = []
            for part in str(s).split(","):
                part = part.strip()
                if not part:
                    continue
                items.append(int(part))
            return items

        def _parse_ffn_specs(s: str):
            # 'B:L,B:L'
            specs = []
            for part in str(s).split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(f"Invalid --overlay_ffn item: {part!r}. Expected 'B:L'.")
                b_str, l_str = part.split(":", 1)
                specs.append((int(b_str.strip()), int(l_str.strip())))
            return specs

        prefixes = []
        if args.overlay_prefix:
            prefixes.extend([str(p) for p in args.overlay_prefix if str(p).strip()])

        if args.overlay_block_idxs:
            for block_idx in _parse_int_csv(args.overlay_block_idxs):
                prefixes.append(f"model.blocks.{block_idx}.")

        if args.overlay_ffn:
            for block_idx, layer_idx in _parse_ffn_specs(args.overlay_ffn):
                prefixes.append(f"model.blocks.{block_idx}.mlp.layers.{layer_idx}.")
                prefixes.append(f"model.blocks.{block_idx}.mlp.layernorm.{layer_idx}.")
                if args.overlay_include_model_norm:
                    prefixes.append(f"model.norm.{layer_idx}.")

        # Deduplicate while preserving order
        seen = set()
        prefixes = [p for p in prefixes if not (p in seen or seen.add(p))]

        if not prefixes:
            raise ValueError(
                "--overlay_from_dir was provided but no selection was given. "
                "Use one of: --overlay_block_idxs / --overlay_ffn / --overlay_prefix"
            )

        print(f"[Overlay] loading from: {args.overlay_from_dir}")
        print(f"[Overlay] prefixes: {prefixes}")
        overlay_sd = MandelbrotV1ShardManager.load_state_dict_from_dir(
            str(args.overlay_from_dir), prefixes=prefixes, map_location="cpu"
        )
        missing, unexpected = model.load_state_dict(overlay_sd, strict=False)
        print(f"✅ [Overlay] loaded tensors: {len(overlay_sd)} | missing={len(missing)} | unexpected={len(unexpected)}")

    # ===== torch.compile (before DDP wrapping, Linux only) =====
    if args.torch_compile:
        if is_main_process():
            print("🔧 Applying torch.compile (dynamic=True, mode=default, fullgraph=False)...")
        model = torch.compile(
            model,
            dynamic=True,          # MoE: expert selection is data-dependent → dynamic graph
            fullgraph=False,       # MoE has control flow that can't be a single graph
            mode="default",        # Default mode (no CUDA graphs) — CUDA graphs incompatible with forward hooks
        )
        if is_main_process():
            print("✅ torch.compile applied (first batch will compile, expect slower start)")

    
    def _apply_incremental_freeze(train_only_block_idx: int) -> None:
        """Freeze everything except a single block and its per-layer norms.

        Important: For memory reasons, this works best when training the *last* block.
        If you train a non-last block, later frozen blocks will still be in the autograd
        graph (because their inputs require grad), which increases memory.
        """

        layer_num = int(getattr(config, "layer_num", 0) or 0)
        if layer_num <= 0:
            raise ValueError(f"Invalid config.layer_num={layer_num}; cannot compute norm index mapping")

        num_blocks = int(getattr(config, "num_hidden_blocks", 0) or 0)
        if num_blocks <= 0 and hasattr(model, "model") and hasattr(model.model, "blocks"):
            num_blocks = len(model.model.blocks)
        if num_blocks <= 0:
            raise ValueError("Cannot determine num_hidden_blocks for incremental freeze")

        if not (0 <= train_only_block_idx < num_blocks):
            raise ValueError(f"train_only_block_idx out of range: {train_only_block_idx} (num_blocks={num_blocks})")

        if train_only_block_idx != (num_blocks - 1):
            print(
                f"⚠️ Incremental training warning: training block {train_only_block_idx} but model has {num_blocks} blocks. "
                "For best memory savings, train the LAST block."
            )

        # Freeze all first
        for _, p in model.named_parameters():
            p.requires_grad = False

        # Unfreeze the target block
        block_prefix = f"model.blocks.{train_only_block_idx}."

        # Unfreeze per-layer norms for this block: global_idx = block_idx*layer_num + layer_idx
        norm_start = train_only_block_idx * layer_num
        norm_prefixes = [f"model.norm.{i}." for i in range(norm_start, norm_start + layer_num)]

        extra_prefixes = []
        if args.also_train_lm_head:
            extra_prefixes.append("lm_head.")
        if args.also_train_embeddings:
            # Depending on tokenizer_manager presence, model may use embedding_manager or embed_tokens.
            extra_prefixes.extend(["model.embedding_manager.", "model.embed_tokens."])

        train_prefixes = [block_prefix, *norm_prefixes, *extra_prefixes]

        trainable_names = []
        for n, p in model.named_parameters():
            if any(n.startswith(pref) for pref in train_prefixes):
                p.requires_grad = True
                trainable_names.append(n)

        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(
            f"✅ Incremental freeze enabled: train_only_block_idx={train_only_block_idx} | "
            f"trainable={trainable_count:,} / total={total_count:,} params"
        )
        if len(trainable_names) == 0:
            raise RuntimeError("No trainable parameters matched; check naming prefixes")

    def _apply_incremental_freeze_blocks(train_block_indices) -> None:
        """Freeze everything except selected blocks and their per-layer model.norm entries."""

        layer_num = int(getattr(config, "layer_num", 0) or 0)
        if layer_num <= 0:
            raise ValueError(f"Invalid config.layer_num={layer_num}; cannot compute norm index mapping")

        num_blocks = int(getattr(config, "num_hidden_blocks", 0) or 0)
        if num_blocks <= 0 and hasattr(model, "model") and hasattr(model.model, "blocks"):
            num_blocks = len(model.model.blocks)

        # Freeze all first
        for _, p in model.named_parameters():
            p.requires_grad = False

        train_prefixes = []
        for b in train_block_indices:
            if not (0 <= int(b) < int(num_blocks)):
                raise ValueError(f"train_block_idx out of range: {b} (num_blocks={num_blocks})")
            train_prefixes.append(f"model.blocks.{int(b)}.")
            norm_start = int(b) * layer_num
            train_prefixes.extend([f"model.norm.{i}." for i in range(norm_start, norm_start + layer_num)])

        if args.also_train_lm_head:
            train_prefixes.append("lm_head.")
        if args.also_train_embeddings:
            train_prefixes.extend(["model.embedding_manager.", "model.embed_tokens."])

        trainable = 0
        total = 0
        for n, p in model.named_parameters():
            total += p.numel()
            if any(n.startswith(pref) for pref in train_prefixes):
                p.requires_grad = True
                trainable += p.numel()

        print(f"✅ Incremental freeze enabled: train_blocks={list(train_block_indices)} | trainable={trainable:,} / total={total:,} params")
        if trainable == 0:
            raise RuntimeError("No trainable parameters matched for train_blocks")

    def _apply_incremental_freeze_new_ffn(block_idx: int, new_layer_indices) -> None:
        """Freeze everything except newly added FFN layers in one block.

        Matches both dense MLP and MoE experts by name:
        - model.blocks.B.mlp.layers.L.* / model.blocks.B.mlp.layernorm.L.*
        - model.blocks.B.mlp.experts.E.layers.L.* / model.blocks.B.mlp.experts.E.layernorm.L.* (if present)
        """

        import re

        layer_num = int(getattr(config, "layer_num", 0) or 0)
        if layer_num <= 0:
            raise ValueError(f"Invalid config.layer_num={layer_num}; cannot compute norm index mapping")

        num_blocks = int(getattr(config, "num_hidden_blocks", 0) or 0)
        if num_blocks <= 0 and hasattr(model, "model") and hasattr(model.model, "blocks"):
            num_blocks = len(model.model.blocks)

        if block_idx < 0:
            block_idx = num_blocks - 1
        if not (0 <= int(block_idx) < int(num_blocks)):
            raise ValueError(f"new_ffn_block_idx out of range: {block_idx} (num_blocks={num_blocks})")

        # Freeze all
        for _, p in model.named_parameters():
            p.requires_grad = False

        # Build regexes for trainable param names
        layer_alt = "|".join(str(int(i)) for i in new_layer_indices)
        dense_re = re.compile(rf"^model/.blocks/.{int(block_idx)}/.mlp/.(layers|layernorm)/.({layer_alt})/.")
        moe_re = re.compile(rf"^model/.blocks/.{int(block_idx)}/.mlp/.experts/./d+/.(layers|layernorm)/.({layer_alt})/.")
        # Gate often needs to adapt when new layers are added, but we keep it frozen by default unless you enable also_train_*.

        norm_prefixes = []
        for layer_idx in new_layer_indices:
            global_idx = int(block_idx) * layer_num + int(layer_idx)
            norm_prefixes.append(f"model.norm.{global_idx}.")

        if args.also_train_lm_head:
            extra_prefixes = ["lm_head."]
        else:
            extra_prefixes = []
        if args.also_train_embeddings:
            extra_prefixes.extend(["model.embedding_manager.", "model.embed_tokens."])

        trainable = 0
        total = 0
        for n, p in model.named_parameters():
            total += p.numel()
            if dense_re.match(n) or moe_re.match(n) or any(n.startswith(pref) for pref in norm_prefixes + extra_prefixes):
                p.requires_grad = True
                trainable += p.numel()

        print(
            f"✅ Incremental freeze enabled: train_only_new_ffn block={block_idx} layers={list(new_layer_indices)} | "
            f"trainable={trainable:,} / total={total:,} params"
        )
        if trainable == 0:
            raise RuntimeError("No trainable parameters matched for new FFN layers")

    def _apply_incremental_freeze_embeddings_and_lm_head() -> None:
        """Freeze everything except embeddings and lm_head.

        This is intended for the "vocab must grow" scenario when you add tokenizer layer(s).
        """

        for _, p in model.named_parameters():
            p.requires_grad = False


        train_prefixes = [
            # embeddings
            "model.embedding_manager.",
            "model.embed_tokens.",
            # output
            "lm_head.",
        ]

        trainable = 0
        total = 0
        matched = 0
        for n, p in model.named_parameters():
            total += p.numel()
            if any(n.startswith(pref) for pref in train_prefixes):
                p.requires_grad = True
                trainable += p.numel()
                matched += 1
            

        print(f"✅ Warmup freeze enabled: train_only_embeddings_and_lm_head | trainable={trainable:,} / total={total:,} params")
        if matched == 0 or trainable == 0:
            raise RuntimeError("No trainable parameters matched for embeddings+lm_head warmup")

    def initialize_model_weights(model,layer_idx, hidden_size, init_dir):
        # Unwrap DDP if needed to access model internals
        model_ref = model.module if hasattr(model, "module") else model

        for _, p in model.named_parameters():
            p.requires_grad = True

        train_prefixes = [
            # embeddings
            #"model.embedding_manager.",
            #"model.embed_tokens.",
            # output
            #"lm_head.",
            model_ref.model.embedding_manager._old_key(layer_idx, hidden_size)
        ]

    
        trainable = 0
        total = 0
        matched = 0

        """Initialize new weights for Blocks"""
        index_file = os.path.join(init_dir, "model.safetensors.index.json")
        whole_file = os.path.join(init_dir, "model.safetensors")
        
        if not os.path.exists(index_file) and not os.path.exists(whole_file):
            print(f"Warning: Neither {index_file} nor {whole_file} found")
            print("Will initialize all new weights with default initialization")
            loaded_weights=set()
        elif os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            loaded_weights = set(index_data.get('weight_map', {}).keys())
            print(f"Loaded weights from checkpoint: {len(loaded_weights)}")
        else:
            print(f"Warning: Only found {whole_file}, no index file")
            return
        
        print(f"Initializing weights not in checkpoint...")
        
        for name, param in model.named_parameters():
            total += param.numel()
            if any(pref in name for pref in train_prefixes):
                param.requires_grad = not args.freeze_embeddings if args.freeze_embeddings is not None else True
                trainable += param.numel()
                matched += 1
         
            if name not in loaded_weights:
                # Core fix: noorm weight no longer hard 1 and switch to small random numbers
                if 'weight' in name and param.dim() > 1:
                    if 'embedding' not in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'norm' in name.lower():
                    nn.init.constant_(param, 1)
                
                #print(f"  Initialized: {name}, shape={param.shape}")

        print(f"✅ Warmup freeze enabled: freeze_only_embeddings | trainable={trainable:,} / total={total:,} params")
        if matched == 0 or trainable == 0:
            raise RuntimeError("No trainable parameters matched for embeddings+lm_head warmup")

    

    

    # ===== Incremental / block-wise training (freeze params before optimizer is built) =====
    debug_grad_target_prefixes = None
    debug_grad_target_desc = None

    if (
        args.use_block_output_cache
        and args.train_only_new_blocks
        and int(args.add_blocks or 0) > 0
        and not args.also_train_lm_head
    ):
        args.also_train_lm_head = True
        print("[Auto] Enabled --also_train_lm_head for block-output-cache new-block training.")

    if args.train_only_new_blocks and int(args.add_blocks or 0) > 0 and not args.debug_grad_stats:
        args.debug_grad_stats = True
        print("[Auto] Enabled --debug_grad_stats for train_only_new_blocks.")

    if args.use_block_output_cache :
        # Lightweight mode already set requires_grad correctly when constructing the model.
        print("✅ Skipping incremental freeze helpers (lightweight cache-only mode)")
    else:
        if args.train_only_embeddings_and_lm_head:
            # Keep this mode exclusive to avoid surprising interactions.
            if args.train_only_new_blocks or args.train_only_new_ffn or args.train_only_last_block or (args.train_only_block_idx is not None):
                raise ValueError(
                    "--train_only_embeddings_and_lm_head is mutually exclusive with --train_only_new_blocks/--train_only_new_ffn/--train_only_block_idx/--train_only_last_block"
                )
            _apply_incremental_freeze_embeddings_and_lm_head()

        if args.train_only_new_blocks and int(args.add_blocks or 0) > 0:
            # Train only blocks that were newly appended.
            total_blocks = int(getattr(config, "num_hidden_blocks", base_num_blocks) or base_num_blocks)
            new_blocks = list(range(base_num_blocks, total_blocks))
            _apply_incremental_freeze_blocks(new_blocks)
            if args.debug_grad_stats:
                layer_num_dbg = int(getattr(config, "layer_num", 0) or 0)
                debug_grad_target_prefixes = [f"model.blocks.{int(b)}." for b in new_blocks]
                if layer_num_dbg > 0:
                    for b in new_blocks:
                        norm_start = int(b) * layer_num_dbg
                        debug_grad_target_prefixes.extend(
                            [f"model.norm.{i}." for i in range(norm_start, norm_start + layer_num_dbg)]
                        )
                debug_grad_target_desc = f"new_blocks={new_blocks}"

        if args.train_only_new_ffn and int(args.add_ffn_layers or 0) > 0:
            # Train only the newly appended FFN layer(s) in a chosen block.
            new_layers = list(range(base_layer_num, int(getattr(config, "layer_num", base_layer_num))))
            _apply_incremental_freeze_new_ffn(int(args.new_ffn_block_idx), new_layers)

        if args.train_only_last_block:
            if hasattr(model, "model") and hasattr(model.model, "blocks"):
                args.train_only_block_idx = len(model.model.blocks) - 1
            else:
                args.train_only_block_idx = int(getattr(config, "num_hidden_blocks", 1)) - 1

        if args.train_only_block_idx is not None:
            _apply_incremental_freeze(int(args.train_only_block_idx))
            if args.debug_grad_stats:
                b = int(args.train_only_block_idx)
                layer_num_dbg = int(getattr(config, "layer_num", 0) or 0)
                debug_grad_target_prefixes = [f"model.blocks.{b}."]
                if layer_num_dbg > 0:
                    norm_start = b * layer_num_dbg
                    debug_grad_target_prefixes.extend(
                        [f"model.norm.{i}." for i in range(norm_start, norm_start + layer_num_dbg)]
                    )
                debug_grad_target_desc = f"block_idx={b}"

    if args.debug_grad_stats:
        if debug_grad_target_prefixes:
            print(f"[GradDebug] target={debug_grad_target_desc} | prefixes={len(debug_grad_target_prefixes)}")
        else:
            print("[GradDebug] target=all trainable parameters")

    def _build_state_dict_for_save(model_for_save: nn.Module, *, trainable_only: bool = False,layer_idx: int = -1,hidden_size: int = 128):
        # FSDP: Requires collection of full volume state_dict (default is split)
        if _fsdp_enabled and isinstance(model_for_save, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=(not ddp_enabled or is_main_process()))
            with FSDP.state_dict_type(model_for_save, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model_for_save.state_dict()
        else:
            state_dict = model_for_save.state_dict()

        # Get unwrapped model references (also accessible under FSDP/DDP wrapping)
        _model_inner = model_for_save.module if hasattr(model_for_save, "module") else model_for_save
        old_key = _model_inner.model.embedding_manager._old_key(layer_idx, hidden_size) if layer_idx>=0 else ""

        """Build statedict to save, excluding blocks, noorm, mm_head and embedding"""
    
        # Define all prefixes that need to be excluded
        include_prefixes = []
        
        # 1. Exclude Blocks (index < start_block_idx)
        for block_idx in range(len(_model_inner.model.blocks)):

            if _model_inner.model.blocks[block_idx].need_normal:

                include_prefixes.append(f'model.blocks.{block_idx}.')
            
                # 2. Exclusion of the noorm layer
                include_prefixes.append(f'model.norm.{block_idx}.')
                
                # 3. Exclusions
                include_prefixes.append(f'lm_head.{block_idx}.')
                
                # 4. Exclusion of the embeding layer
                include_prefixes.append(f'model.embedding_manager.units.L{block_idx}_')

                include_prefixes.append(f'model.embedding_manager.units.O_L{block_idx}_')

                
        if layer_idx>=0:
            include_prefixes.append(f'model.embedding_manager.units.{ old_key}')

        def should_include(name: str) -> bool:
            return any(name.startswith(prefix) for prefix in include_prefixes)
        
        trainable_names = []
        
        for name, param in _model_inner.named_parameters():
            # Only trainingable parameters are saved
            if trainable_only and not param.requires_grad:
                continue
            
            # Check whether or not to exclude
            if should_include(name) and param.device.type != 'meta':
                trainable_names.append(name)
            
        state_dict = {name: tensor for name, tensor in state_dict.items() if name in trainable_names}
  
        if save_weight_dtype == torch.float32:
            return dict(state_dict)

        converted_state_dict = {}
        for name, tensor in state_dict.items():
            value = tensor.detach()
            if torch.is_floating_point(value):
                value = value.to(dtype=save_weight_dtype)
            converted_state_dict[name] = value.cpu()
        return converted_state_dict

    def _set_config_torch_dtype(model_for_save: nn.Module):
        try:
            model_for_save.config.torch_dtype = str(save_weight_dtype).replace("torch.", "")
        except Exception:
            pass

    def _force_saved_config_dtype(save_dir: str):
        try:
            config_path = os.path.join(save_dir, "config.json")
            if not os.path.isfile(config_path):
                return
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["torch_dtype"] = str(save_weight_dtype).replace("torch.", "")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception as e:
            print(f"⚠️ Warning: failed to update config torch_dtype in {save_dir}: {e}")

    use_dimensionality_reduction= getattr(model, "use_dimensionality_reduction", True) 

    layer_idx=args.layer_idx if args.layer_idx>=0 and use_dimensionality_reduction else tokenizer_manager.num_layers-1

    model.last_layer_idx=None if use_dimensionality_reduction else layer_idx
    model.bMax_frequency=use_dimensionality_reduction


    def _save_hf_sharded_safetensors_compat(
        shard_manager,
        model_or_state_dict,
        out_dir: str,
        *,
        per_ffn: bool,
        moe_per_expert: bool,
        shards_subdir: str,
        dtype: torch.dtype,
        overwrite: bool,
    ):
        save_fn = shard_manager.save_hf_sharded_safetensors
        save_kwargs = {
            "per_ffn": per_ffn,
            "moe_per_expert": moe_per_expert,
            "shards_subdir": shards_subdir,
            "overwrite": overwrite,
        }
        try:
            if "dtype" in inspect.signature(save_fn).parameters:
                save_kwargs["dtype"] = dtype
        except (TypeError, ValueError):
            pass
        return save_fn(model_or_state_dict, out_dir, **save_kwargs)

    # dataset & dataloader (streaming)

 
    # Build a rank-aware worker_init_fn so DataLoader workers get unique seeds per rank
    def _ddp_worker_init_fn(worker_id: int):
        """Set unique seed for each DataLoader worker across ranks."""
        rank = get_rank()
        ws = get_world_size()
        # Each rank's workers get a distinct seed offset
        worker_seed = int(args.seed) + rank * 1000 + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def combined_worker_init_fn(worker_id):
        # DDP seed
        if ddp_enabled:
            _ddp_worker_init_fn(worker_id)
        # Client mode tokenizer/stub init
        if args.is_Distribution and args.distribution_Type == "client":
            worker_init_fn(worker_id)

    if (args.is_Distribution and args.distribution_Type != "client"):
        args.dataloader_num_workers = 0

    _need_gc = not (args.use_block_output_cache or args.no_gradient_checkpointing)

    if args.use_block_output_cache or (args.generate_block_output_cache and not args.train_folder):
        if not args.block_cache_dir:
            raise ValueError("--use_block_output_cache requires --block_cache_dir")

        if ddp_enabled:
            # Shard cache files across ranks so each rank processes different samples
            all_cache_files = sorted(
                [os.path.join(args.block_cache_dir, f) for f in os.listdir(args.block_cache_dir)
                 if f.lower().endswith(".pt")]
            )
            # Partition files by rank
            per_rank_files = all_cache_files[get_rank()::get_world_size()]
            if is_main_process():
                print(f"[DDP] ActivationCacheDataset: total files={len(all_cache_files)}, "
                      f"rank {get_rank()} gets {len(per_rank_files)} files")
            dataset = ActivationCacheDataset(args.block_cache_dir, file_list=per_rank_files)
        else:
            dataset = ActivationCacheDataset(args.block_cache_dir)


        dataloader = AsyncPrefetchDataLoader(
            dataset,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
            drop_last=False,
            collate_fn=_collate_first_item,
            prefetch_factor=3 if args.dataloader_num_workers > 0 else None,
            pin_memory=not nccl_mgr.use_nccl,
            persistent_workers=bool(args.dataloader_num_workers > 0),
            worker_init_fn=global_worker_init_fn if args.dataloader_num_workers > 0 else None,
            prefetch_size=args.prefetch_size
        )
        if is_main_process():
            print(f"✅ Using BlockOutputCacheDataset from {args.block_cache_dir}")
    else:
        collate_fn = _collate_streaming_blocks
        batch_size = args.per_device_train_batch_size
        server_url = getattr(config, "incremental_url", None)
        server_port = getattr(config, "incremental_port", None)

        if ( args.generate_block_output_cache) and args.paired_train_folder:
            dataset = PairedLineTextDataset(
                config,
                folder_a=args.train_folder,
                folder_b=args.paired_train_folder,
                tokenizer=None,
                pad_mode=args.pad_mode,
                tokenizer_name=args.tokenizer_name,
                tokenizer_layer_idx=layer_idx,
                block_size=args.block_size,
                eos_token_id=eos_id,
                max_lines=args.max_train_lines,
                
            )
            # Keep raw tuple, avoid default_collate batching.
            collate_fn = _collate_first_item
            batch_size = 1

        else:
            # -DDP Data Disc: Collect all files and distribute them evenly according to rank -
            _all_files = []
            _folder = Path(args.train_folder)
            if _folder.is_dir():
                _all_files = sorted(
                    [str(p) for p in _folder.rglob("*.txt")] +
                    [str(p) for p in _folder.rglob("*.jsonl")]
                )
            elif _folder.is_file():
                _all_files = [str(_folder)]

            if ddp_enabled and len(_all_files) > 0:
                rank = get_rank()
                world_size = get_world_size()
                # Cross rank stagger to ensure that every GPU-distributed file is as even as possible
                _rank_files = _all_files[rank::world_size]
                if is_main_process():
                    _sizes = [
                        len(_all_files[r::world_size])
                        for r in range(world_size)
                    ]
                    print(f"[DDP] File sharding: {len(_all_files)} total files, "
                          f"per-rank distribution: {_sizes}")
            else:
                _rank_files = _all_files  # No splits for single cards or files when empty

            # When DDP is enabled, pass the underlying (unwrapped) model to the dataset
            _model_for_dataset = model.module if (ddp_enabled and hasattr(model, "module")) else model
            dataset = StreamingTextDataset(
                config,
                folder=args.train_folder,
                tokenizer=None,
                pad_mode=args.pad_mode,
                tokenizer_name=args.tokenizer_name,
                tokenizer_layer_idx=layer_idx,
                block_size=args.block_size,
                batch_size=batch_size,
                eos_token_id=eos_id,
                max_lines=args.max_train_lines,
                by_eos=args.by_eos,
                add_special_tokens=args.add_special_tokens,
                use_dimensionality_reduction=use_dimensionality_reduction,
                is_Distribution=args.is_Distribution,
                distribution_Type=args.distribution_Type,
                worker_id= args.worker_id,
                model=_model_for_dataset,
                ddp_enabled=ddp_enabled,
                seed=args.seed,
                use_gradient_checkpointing=_need_gc,
                use_triple=args.use_triple,
                use_compute_comm_overlap=args.use_compute_comm_overlap,
                mem_threshold=args.mem_threshold,
                vram_threshold=args.vram_threshold,
                save_epochs=args.save_epochs,
                save_steps=args.save_steps,
            )
            dataset.tm=MandelbrotV1TokenizerManager.tokenizer_manager
            dataset.tokenizer = MandelbrotV1TokenizerManager.tokenizer_manager.tokenizers[layer_idx]
            
            # -DDP Data Sub-discreet: Created and replaced the file list of dataset (retrain script only, not BaseTextDataset)
            if ddp_enabled and len(_rank_files) > 0:
                dataset.files_txt = [Path(f) for f in _rank_files if f.lower().endswith(".txt")]
                dataset.files_jsonl = [Path(f) for f in _rank_files if f.lower().endswith(".jsonl")]
                if is_main_process():
                    print(f"[DDP] rank={get_rank()}: assigned {len(dataset.files_txt)} txt + "
                          f"{len(dataset.files_jsonl)} jsonl files")

            collate_fn = __collate_streaming_blocks if args.pad_mode else _collate_streaming_blocks

    
        
        dataloader = AsyncPrefetchDataLoader(
            dataset,
            batch_size=None if args.is_Distribution and args.distribution_Type!="client" else batch_size,
            num_workers=args.dataloader_num_workers,
            drop_last=bool(args.dataloader_drop_last),
            collate_fn=collate_fn,
            prefetch_factor=3 if args.dataloader_num_workers > 0 else None,
            pin_memory=not nccl_mgr.use_nccl,
            persistent_workers=bool(args.dataloader_num_workers > 0),
            worker_init_fn=global_worker_init_fn if args.dataloader_num_workers > 0 else None,
            prefetch_size=args.prefetch_size
        )

    # Initialize communication monitor (gRPC stub/ DataLoader must be created before it can be intercepted
    mon = GPUMonitor()
    mon.start()
    print("✅ Communication monitor activated (torch.distributed + grPC communications)")

    global_worker_init_fn(-1,dataset)

    _need_gc=dataset.use_gradient_checkpointing
    args.save_epochs=dataset.save_epochs
    args.save_steps=dataset.save_steps

   
    # memory saving options
    # NOTE: For block-output-cache replay, gradient checkpointing can zero out grads
    # on newly added blocks (start_block_idx path). Keep it disabled in that mode.
    try:
        # FSDP must enable gradient checkpointing, otherwise activations OOM
        
        if args.use_fsdp:
            _need_gc = True  # FSDP Force Start
            if is_main_process():
                print("[FSDP] Forcing gradient_checkpointing ON (activations are NOT sharded)")
        if _need_gc:
            model.gradient_checkpointing_enable()
            print("Enabled model.gradient_checkpointing")
        else:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
                print("Disabled model.gradient_checkpointing")
            else:
                print("gradient_checkpointing_disable not available; keeping current model setting")
    except Exception:
        print("gradient_checkpointing not available for this model type")

    if is_main_process():
        print("dataloader = DataLoader = ", dataloader)
    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters,lr=args.learning_rate,betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

    # DDP: optionally scale learning rate linearly with world_size
    _ws = get_world_size()
    if _ws > 1:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * _ws
        if is_main_process():
            print(f"[DDP] Learning rate scaled by world_size: {args.learning_rate:.3e} -> "
                  f"{optimizer.param_groups[0]['lr']:.3e}")

    total_steps = args.max_steps
    if total_steps <= 0:
        # Epoch mode (max_steps=-1): total step estimates based on data sets
        _ws = get_world_size()
        _bs = args.per_device_train_batch_size * max(_ws, 1)
        _epochs = args.max_epochs if args.max_epochs > 0 else 1
        # Estimate: number of statistical data set documents and average line count
        import glob as _glob
        _files = _glob.glob(os.path.join(args.train_folder, "*.jsonl")) or _glob.glob(os.path.join(args.train_folder, "*.txt"))
        if _files:
            with open(_files[0], "r", encoding="utf-8") as _f:
                _first_lines = sum(1 for _ in _f)
            _total_lines_est = _first_lines * len(_files)
            # ~1.5 Blocks/line (based on Block_size=512 + Average English tokenization)
            _blocks_per_line = 1.5
            _total_blocks = int(_total_lines_est * _blocks_per_line)
            _steps_per_epoch = max(_total_blocks // _bs, 1)
            total_steps = _steps_per_epoch * _epochs
        else:
            total_steps = 100000  # fallback
        if is_main_process():
            print(f"[Scheduler] Epoch mode: estimated total_steps={total_steps} "
                  f"(files={len(_files) if _files else 0}, steps/epoch~={total_steps // _epochs})")

    scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    scaler = torch.amp.GradScaler(enabled=use_fp16)

    device = next(model.parameters()).device
    # logging
    #tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    # Initialize global_step
    global_step = 0
    resume_epoch = 0
    
    if args.resume_from_checkpoint:
        print("Resuming from:", args.resume_from_checkpoint)
        
        # Load model directly to initialised model
        ckpt_path = Path(args.resume_from_checkpoint)
        
        # Try loading safenesors or pytoch_model.bin
        safetensors_path = ckpt_path / "model.safetensors"
        pytorch_bin_path = ckpt_path / "pytorch_model.bin"
        
        if not should_skip_full_checkpoint:
            if safetensors_path.exists():
                print(f"Loading model weights from {safetensors_path}")
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict, strict=False)
                print("✅ Model weights loaded successfully from safetensors")
            elif pytorch_bin_path.exists():
                print(f"Loading model weights from {pytorch_bin_path}")
                state_dict = torch.load(pytorch_bin_path, map_location=device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print("✅ Model weights loaded successfully from pytorch_model.bin")
            else:
                print(f"⚠️ Warning: No model weights found in {args.resume_from_checkpoint}")
        else:
            sd = _load_base_sd(args.resume_from_checkpoint)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"✅ [Overlay] loaded tensors: {len(sd)} | missing={len(missing)} | unexpected={len(unexpected)}")
            
        
        # Loading training status
        opt_path = ckpt_path / "trainer_state.pt"
        if opt_path.exists():
            st = torch.load(str(opt_path), map_location=device, weights_only=False)
            
            # Save learning rates in checkpoint for comparison
            old_checkpoint_lr = None
            if "scheduler_state_dict" in st and "_last_lr" in st["scheduler_state_dict"]:
                old_checkpoint_lr = st["scheduler_state_dict"]["_last_lr"][0]
            
            optimizer.load_state_dict(st.get("optimizer_state_dict", optimizer.state_dict()))
            scheduler.load_state_dict(st.get("scheduler_state_dict", scheduler.state_dict()))
            if use_fp16 and "scaler_state_dict" in st:
                scaler.load_state_dict(st["scaler_state_dict"])
            global_step = st.get("global_step", 0)
            resume_epoch = st.get("epoch", 0)
            
            # ✅ Check whether learning rates need to be covered
            current_scheduler_lr = scheduler.get_last_lr()[0]
            if old_checkpoint_lr is not None and abs(args.learning_rate - old_checkpoint_lr) > 1e-9:
                # User specifies a new learning rate, which covers Scheduler
                print(f"🔧 Detected learning rate change:")
                print(f"   Checkpoint LR: {old_checkpoint_lr:.3e}")
                print(f"   New LR requested: {args.learning_rate:.3e}")
                
                # Set the learning rate for optimizer manually
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate
                
                # Recreate Scheduler to use new learning rates
                scheduler = get_cosine_schedule_with_warmup_and_min_lr(
                    optimizer, 
                    num_warmup_steps=args.warmup_steps, 
                    num_training_steps=total_steps,
                    min_lr_ratio=args.min_lr_ratio
                )
                # Step in Scheduler to the current step
                for _ in range(global_step):
                    scheduler.step()
                
                print(f"✅ Learning rate successfully updated to: {scheduler.get_last_lr()[0]:.3e}")
            else:
                print(f"✅ Using checkpoint learning rate: {current_scheduler_lr:.3e}")
            
            print(f"✅ Loaded checkpoint: global_step={global_step}, epoch={resume_epoch}")
            print(f"✅ Training will continue from step {global_step + 1}")
        else:
            print(f"⚠️ Warning: trainer_state.pt not found, training state not restored")  
   
            


    model.train()

    initialize_model_weights(model, layer_idx=layer_idx, hidden_size=config.hidden_size, init_dir=init_dir)

    # ===== Wrap model with DDP or FSDP (after all weight loading/overlay is complete) =====
    if ddp_enabled and not args.is_Distribution:
        
        if args.use_fsdp:
            # ── FSDP (ZeRO-3 / ZeRO-2): Parameters/ gradient/ Optimizer fraction -
            _fsdp_strategy = {
                "full": ShardingStrategy.FULL_SHARD,       # ZeRO-3
                "grad": ShardingStrategy.SHARD_GRAD_OP,    # ZeRO-2
                "none": ShardingStrategy.NO_SHARD,         # DDP-like
            }[args.fsdp_sharding_strategy]

            _fsdp_mp = None
            if amp_enabled:
                _fsdp_mp = FSDPMixedPrecision(
                    param_dtype=amp_dtype,
                    reduce_dtype=amp_dtype,
                    buffer_dtype=amp_dtype,
                )

            # [Core repair] Use the most robust custom packaging strategy
            from modeling_MandelbrotV1 import MandelbrotV1DecoderBlock
            import functools

            def _custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
                # 1. When you meet DecoderBlock, return to True, the FSDP will return to its inner submodule and package it.
                if isinstance(module, MandelbrotV1DecoderBlock):
                    return True
                # Ebedding module cannot be packaged by FSDP alone - FSDP will be nn.Embedding.weight
                # flatten to 1-D FlatParameter, causing F. Embedding to "weight must be 2-D"
                # Ebedding is managed centrally by top FSDP (use_orig_params=True to ensure that the parameter shape remains the same)
                # 3. All other modules, return to False, not separately packaged
                return False

            _auto_wrap_policy = functools.partial(_custom_auto_wrap_policy)

            # Extra Protection: A visual message to FSDP not to flatten parameters of the n. Ebedingding submodule
            from torch.nn import Embedding as _TorchEmbedding
            _ignored_modules = [
                m for m in model.modules() if isinstance(m, _TorchEmbedding)
            ]

            model = FSDP(
                model,
                sharding_strategy=_fsdp_strategy,
                mixed_precision=_fsdp_mp,
                auto_wrap_policy=_auto_wrap_policy,
                device_id=local_rank,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                use_orig_params=True,   # Keep original parameter name compatible with optimizer
                ignored_modules=_ignored_modules if _ignored_modules else None,
            )

            if is_main_process():
                print(f"✅ Model wrapped with FSDP (strategy={args.fsdp_sharding_strategy}, "
                      f"auto_wrap=MandelbrotV1DecoderBlock, ignored_modules={len(_ignored_modules)} nn.Embedding)")

            # Tag FSDP enabled for subsequent state_dict collection
            _fsdp_enabled = True
        else:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=not args.no_ddp_find_unused,
            )
            if is_main_process():
                print(f"✅ Model wrapped with DDP (find_unused_parameters={not args.no_ddp_find_unused})")
            _fsdp_enabled = False
        
    else:
        _fsdp_enabled = False


    # ===== 5.1 Write getlogits Output to TensorBoard =====
    # Note: The loss of the getlogits calculates and prints within the model; here the tb is recorded with the Monkey-patch without changing the model file.
    try:
        # _orig_getlogits = model.getlogits

        # def _tb_getlogits(self, hidden_states, logits, labels,_labels_dict, layer_idx, _loss,_experts):
        #     logits_out, loss_out = _orig_getlogits(hidden_states, logits, labels,_labels_dict, layer_idx, _loss,_experts)
        #     step = global_vars.get("global_step", None)
        #     if step is not None:
        #         try:
        #             tb_writer.add_scalar(f"getlogits/layer_{layer_idx}_loss", float(loss_out.detach().item()), int(step))
        #         except Exception:
        #             pass
        # # Optional: decode the labels as text records (only when should_log enabled)
        #         if global_vars.get("is_Loss_Log", False):
        #             try:
        #                 lab = labels.detach().view(-1).cpu().tolist()
        #                 decoded = tokenizer_manager.tokenizers[layer_idx].decode(lab, skip_special_tokens=False)
        #                 tb_writer.add_text(f"getlogits/layer_{layer_idx}_labels", decoded, int(step))
        #             except Exception:
        #                 pass
        #     return logits_out, loss_out

        # model.getlogits = MethodType(_tb_getlogits, model)
        print("✅ TensorBoard logging enabled for getlogits (per-layer auxiliary loss)")
    except Exception as e:
        print(f"⚠️ Failed to hook getlogits into TensorBoard: {e}")

    print("Starting training loop...")
    start_time = time.time()
    epoch = resume_epoch
    num_train_epochs = int(args.max_epochs) if args.max_epochs is not None and  args.max_epochs > 0 else sys.maxsize
    if num_train_epochs <= 0:
        raise ValueError(f"--num_train_epochs must be > 0, got {num_train_epochs}")
    print(f"Training schedule: num_train_epochs={num_train_epochs}, max_steps={args.max_steps}")
    running_loss = 0.0
    raw_loss_window_total = 0
    raw_loss_window_non_tensor = 0
    raw_loss_window_int_one = 0
    last_grad_debug_snapshot = None
    last_grad_debug_global_step = None
    # If the training is repeated, skipping the first logging display anomaly
    steps_since_last_log = global_step % args.logging_steps if args.resume_from_checkpoint else 0
    early = EarlyStopper(mode='min', patience=args.early_patience, min_delta=args.early_min_delta, restore_best=False)
    max_seconds = args.time_budget_hours * 3600 if args.time_budget_hours else None
    
    # Generate computational PDF (only once at the start of training)
    if args.render_torchviz_graphs and (not args.resume_from_checkpoint or global_step == 0):
        try:
            print("🎨 Generating computation graph PDF...")
            model.train()  # Maintain training mode to show complete calculation
            model_ref = model.module if hasattr(model, "module") else model

            if args.use_block_output_cache :
                if not args.block_cache_dir:
                    raise ValueError("--render_torchviz_graphs with lightweight block-cache mode requires --block_cache_dir")

                cache_files = sorted(
                    [os.path.join(args.block_cache_dir, f) for f in os.listdir(args.block_cache_dir) if f.lower().endswith(".pt")]
                )
                if not cache_files:
                    raise RuntimeError(f"No .pt cache files found in {args.block_cache_dir}")

                example_batch = torch.load(cache_files[0], map_location="cpu", weights_only=False)
                if not isinstance(example_batch, dict) or ("block_hidden_states" not in example_batch):
                    keys = sorted(example_batch.keys()) if isinstance(example_batch, dict) else [type(example_batch).__name__]
                    raise ValueError(
                        "--render_torchviz_graphs expected a block-cache sample containing 'block_hidden_states'. "
                        f"Got keys={keys}"
                    )

                input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer = use_block_output_cache(example_batch)
                capture = {}
                outputs,capture=use_block_cache_for_generation(input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer,capture=capture,capture_after_block_idx=args.cache_after_block_idx)
            else:
                if hasattr(tokenizer_manager, 'offsets') and len(tokenizer_manager.offsets) >= 2:
                    layer0_start = tokenizer_manager.offsets[0]
                    layer0_end = tokenizer_manager.offsets[1]
                else:
                    layer0_start = 0
                    layer0_end = getattr(tokenizer_manager, 'total_vocab_size', None) or config.vocab_size
                if layer0_end <= layer0_start:
                    layer0_end = layer0_start + 1

                example_input = torch.randint(layer0_start, layer0_end, (1, min(32, args.block_size)), device=device)
                example_labels = example_input.clone()
                example_attention_mask = torch.ones_like(example_input, dtype=torch.long, device=example_input.device)
                if not isinstance(model, MandelbrotV1ForCausalLM):
                    model.module.last_layer_idx=None if use_dimensionality_reduction else layer_idx
                    model.module.bMax_frequency=use_dimensionality_reduction
                outputs = model(
                    input_ids=example_input,
                    attention_mask=example_attention_mask,
                    labels=example_labels,
                    layer_idx=0,
                )
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
            else:
                loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("Example forward did not return a loss; cannot render torchviz graph")
            loss = _normalize_or_recompute_loss(loss, outputs, example_labels)
            
            # Generate calculation diagram using torchviz
            dot = make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=False)
            
            # Set Graphic Properties
            dot.attr(rankdir='TB')  # Top to Bottom Layout
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Save as.dot file first (no graphviz executable required)
            graph_path = os.path.join(args.output_dir, "model_computation_graph")
            dot_file = graph_path + ".dot"
            with open(dot_file, 'w', encoding='utf-8') as f:
                f.write(dot.source)
            print(f"✅ Computation graph saved to: {dot_file}")
            print(f"   To convert to PDF/PNG, install Graphviz from: https://graphviz.org/download/")
            print(f"   Then run: dot -Tpdf {dot_file} -o {graph_path}.pdf")
            print(f"   Or run: dot -Tpng {dot_file} -o {graph_path}.png")
            
            # Try rendering PDF (if graphviz is installed)
            try:
                dot.render(graph_path, format='pdf', cleanup=True)
                print(f"✅ PDF generated: {graph_path}.pdf")
                # Try to generate PNG
                try:
                    dot.render(graph_path, format='png', cleanup=True)
                    print(f"✅ PNG generated: {graph_path}.png")
                except:
                    pass
            except Exception as render_err:
                print(f"⚠️ PDF rendering skipped (Graphviz not found): {render_err}")
                print(f"   Install with: winget install graphviz")
                print(f"   Or download: https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/10.0.1/windows_10_cmake_Release_Graphviz-10.0.1-win64.exe")
            
        except Exception as e:
            print(f"⚠️ Failed to generate computation graph: {e}")
    
    # For recording final indicators
    final_loss = 0.0
    final_ppl = 0.0
    final_id = None
    
    # Initializing dead-neuron detection
    print("\nInitializing dead-neuron detection...")
    if args.enable_neuron_check:
        neuron_checker = DeadNeuronChecker(model, threshold_dead=0.0)
        
        # Initialize neuron activation tracking (optional, for detailed analysis)
        print("\nInitialize neuron activation tracking...")
        activation_tracker = NeuronActivationTracker(
            model,
            tokenizer=None,  
            max_records=50,  # Keep up to 50 step records
            save_interval=args.neuron_track_steps  # Save detailed activation records every neuron_track_steps
        )
        print(f"   activation tracking Frequency: per {args.neuron_track_steps} steps")
        
        # Diagnostic information: check model structure
        print("\nmodel Structural diagnosis:")
        layer_count = 0
        moe_expert_count = 0
        for name, module in model.named_modules():
            if type(module).__name__ == 'MandelbrotV1Layer':
                layer_count += 1
                if 'experts' in name or 'mlp.layers' in name:
                    moe_expert_count += 1
                    if moe_expert_count <= 50:  # Show only the top 50
                        print(f"   Layer of discovery: {name}")
        
        print(f"   Total Mandelbrot V1Layer: {layer_count}")
        print(f"   MoE expert layer: {moe_expert_count}")
        
        if moe_expert_count == 0:
            print("   ⚠️ Warning: no MoE expert layer found; neuron detection may be ineffective")
        else:
            print(f"   ✅ Death detector: Registered {len(neuron_checker.hooks)//2} layers")
            print(f"   ✅ activation tracking: Registered {len(activation_tracker.hooks)} layers")
        
        print("✅ neurons detector activated")
        print(f"   Frequency of death detection: per {args.neuron_check_steps} steps")
        print(f"   activation tracking Frequency: per {args.neuron_track_steps} steps")
    
    # Initialization performance analyser
    print("\nInitialization of training performance analyser...")
    perf_profiler = TrainingPerformanceProfiler(model, config, device=device, enable_gpu_monitor=args.enable_gpu_monitor,grad_accum_steps=args.gradient_accumulation_steps)
    print("✅ Performance analyser activated.")
    perf_records = []  # (tokens, elapsed_seconds) per micro-batch
    if args.enable_gpu_monitor:
        print(f"   To be monitored: FLOPs, throughput, GPU memory, GPU utilization, power, temperature")
    else:
        print(f"   Will monitor: FLOPs, throughput, GPU memory")
        print(f"   reminder: Enable GPS detailed monitoring using --enable_gpu_monitor")
    
    # Initialize WEIGHT-matrix rank analyser (by switch)
    rank_analyzer = None
    if args.enable_rank_analysis:
        print("\nInitialize WEIGHT-matrix rank analyser...")
        rank_analyzer = WeightRankAnalyzer(model, device=device)
        print("✅ rank analyser activated")
        print(f"   Will monitor weight-matrix rank, condition numbers, and singular-value distributions")
        print(f"   rank Analysis Frequency: per {args.rank_check_steps} steps")
    else:
        print("\nWEIGHT-matrix rank analysis disabled")
        print(f"   reminder: Enable rank analysis using --enable_rank_analysis")

    # Initialize intrinsic dimension (ID) analyser (by switch)
    id_analyzer = None
    if args.enable_id_analysis:
        print("\nInitialize intrinsic dimension (ID) analyser...")
        id_analyzer = IntrinsicDimensionAnalyzer(
            model,
            device=device,
            max_buffer_size=args.id_max_samples,
            energy_threshold=float(getattr(args, 'id_energy_threshold', 0.95)),
            mode=args.id_mode
        )
        # Check for analytical targets
        has_targets = (id_analyzer.targets or id_analyzer.block_input_targets or 
                    id_analyzer.block_output_targets)
        
        if has_targets:
            id_analyzer.register_hook()
            print("✅ ID analyser activated")
            print(f"   Analysis mode: {id_analyzer.mode}")
            
        # Show surveillance target according to mode
            if id_analyzer.mode in ["ffn_output", "both_io"] and id_analyzer.targets:
                print(f"   Will monitor: {len(id_analyzer.targets)} FFN's intrinsic dimension")
            if id_analyzer.mode in ["block_input", "both_io"] and id_analyzer.block_input_targets:
                print(f"   Will monitor: {len(id_analyzer.block_input_targets)} intrinsic dimension for Block")
            if id_analyzer.mode in ["block_output", "both_io"] and id_analyzer.block_output_targets:
                print(f"   Will monitor: {len(id_analyzer.block_output_targets)} intrinsic dimension for Block")
                
            print(f"   Analyse frequency: per {args.id_check_steps} steps")
        else:
            print("⚠️ ID Analyzer initialization failed: no analytical target module found")
            id_analyzer = None
    else:
        print("\nintrinsic dimension analysis disabled")
        print(f"   reminder: Enable the intrinsic dimension analysis using --enable_id_analysis")
    
    # Recording hyperparameters at the start of training (only hparam, updated by metric later)
    if not args.resume_from_checkpoint or global_step == 0:
        try:
            hparam_dict = {
                'learning_rate': args.learning_rate,
                'batch_size': args.per_device_train_batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'block_size': args.block_size,
                'n_layer': args.n_layer,
                'n_head': args.n_head,
                'n_embd': args.n_embd,
                'warmup_steps': args.warmup_steps,
                'weight_decay': args.weight_decay,
                'max_grad_norm': args.max_grad_norm,
            }
            # Create hparams with initial values first, and update them later
            metric_dict = {
                'final_loss': 0.0,
                'final_ppl': 0.0,
                'final_id': 0.0,
                'total_steps': 0,
            }
            tb_writer.add_hparams(hparam_dict, metric_dict)
            print("✅ Hyperparameters added to TensorBoard (will be updated at end)")
        except Exception as e:
            print(f"⚠️ Failed to add hyperparameters: {e}")

    def _prepare_cache_generation_batch(local_batch: torch.Tensor, tensor_name: str, _use_dimensionality_reduction: bool, _layer_idx: int):
        if local_batch.dim() == 1:
            local_batch = local_batch.unsqueeze(0)

        if _use_dimensionality_reduction:
            layer_vocab_size = int(tokenizer_manager.vocab_sizes[_layer_idx])
            in_range = (local_batch >= 0) & (local_batch < layer_vocab_size)
            if not torch.all(in_range):
                bad_local = local_batch[~in_range].detach().view(-1).cpu()
                raise ValueError(
                    f"{tensor_name} are not valid local ids for the selected tokenizer layer: "
                    f"layer_idx={layer_idx}, layer_vocab_size={layer_vocab_size}, "
                    f"local_min={int(local_batch.min().item())}, local_max={int(local_batch.max().item())}, "
                    f"bad_count={int(bad_local.numel())}, bad_preview={bad_local[:8].tolist()}"
                )
            source_global_ids = tokenizer_manager._local_to_global_ids(_layer_idx, local_batch.view(-1))
            input_ids = get_inputs(source_global_ids).to(device, non_blocking=True)
            return {
                "source_global_ids": input_ids,
                "model_input_ids": input_ids,
                "labels": input_ids.clone(),
                "forward_layer_idx": layer_idx,
                "selected_layer": layer_idx,
            }

        source_global_ids: List[int] = local_batch.tolist()[0]
        
        selected_layer=tokenizer_manager.get_max_layer_idx_by_max_frequency(source_global_ids,True,bMax_frequency=_use_dimensionality_reduction)


        return {
            "source_global_ids": local_batch,
            "model_input_ids": local_batch,
            "labels": local_batch,
            "forward_layer_idx": selected_layer,
            "selected_layer": selected_layer,
        }

    def _infer_old_vocab_layer_from_global_ids(global_ids: torch.Tensor,bMax_frequency: bool = True) -> int:
        if global_ids.dim() > 1:
            flat_ids = global_ids.view(-1)
        else:
            flat_ids = global_ids
        global_min = int(flat_ids.min().item())
        global_max = int(flat_ids.max().item())
        if bMax_frequency:
            for layer, (start, end) in enumerate(tokenizer_manager.old_layer_ranges):
                if start <= global_min and global_max < end:
                    return int(layer)
        else:
            return tokenizer_manager.get_max_layer_idx_by_max_frequency(flat_ids,True,bMax_frequency)
        raise ValueError(
            "Cannot infer a unique old-vocab layer from cache ids: "
            f"global_min={global_min}, global_max={global_max}"
        )

    def _normalize_old_vocab_cache_labels(input_ids_tensor: torch.Tensor, labels_tensor: torch.Tensor,bMax_frequency: bool = True):
        selected_layer = _infer_old_vocab_layer_from_global_ids(input_ids_tensor,bMax_frequency)
        old_offset = int(tokenizer_manager.old_offsets[selected_layer])
        old_vocab_size = int(tokenizer_manager.old_vocab_sizes[selected_layer])

        labels_min = int(labels_tensor.min().item())
        labels_max = int(labels_tensor.max().item())
        if labels_min >= old_offset and labels_max < (old_offset + old_vocab_size):
            labels_tensor = labels_tensor - old_offset

        if labels_tensor.dim() > 1:
            flat_labels = labels_tensor.view(-1)
        else:
            flat_labels = labels_tensor
        valid_mask = flat_labels != -100
        if torch.any(valid_mask):
            valid_labels = flat_labels[valid_mask]
            valid_min = int(valid_labels.min().item())
            valid_max = int(valid_labels.max().item())
            if valid_min < 0 or valid_max >= old_vocab_size:
                raise ValueError(
                    "Block-cache labels are incompatible with the inferred old-vocab output head: "
                    f"selected_layer={selected_layer}, old_vocab_size={old_vocab_size}, "
                    f"label_min={valid_min}, label_max={valid_max}"
                )

        return labels_tensor, selected_layer
    
    

    

    # ===== Optional: generate block-output cache and exit =====
    if args.generate_block_output_cache:
        if not args.block_cache_dir:
            raise ValueError("--generate_block_output_cache requires --block_cache_dir")
        os.makedirs(args.block_cache_dir, exist_ok=True)
        #model.eval()
        cached = 0
        skipped_short = 0
        with torch.no_grad():
            for batch in dataloader:
                if args.cache_max_batches is not None and cached >= int(args.cache_max_batches):
                    break

                if isinstance(dataloader.dataset, ActivationCacheDataset):
                    input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer = use_block_output_cache(batch)
                    capture = {}
                    outputs,capture=use_block_cache_for_generation(input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer,capture=capture,capture_after_block_idx=args.cache_after_block_idx)
                else:

                    # Support paired mode: batch can be (cache_input_local, label_input_local)
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        cache_input_local = batch[0].to(device, non_blocking=True)
                        label_input_local = batch[1].to(device, non_blocking=True)
                    else:
                        cache_input_local = batch.to(device, non_blocking=True)
                        label_input_local = cache_input_local

                    if cache_input_local.dim() == 1:
                        cache_input_local = cache_input_local.unsqueeze(0)
                    if label_input_local.dim() == 1:
                        label_input_local = label_input_local.unsqueeze(0)

                    cache_prepared = _prepare_cache_generation_batch(cache_input_local, "Block cache input_ids",use_dimensionality_reduction,layer_idx)
                    label_prepared = _prepare_cache_generation_batch(label_input_local, "Block cache labels",use_dimensionality_reduction,layer_idx)    
                    if int(cache_prepared["selected_layer"]) != int(label_prepared["selected_layer"]):
                        raise ValueError(
                            "Block cache generation requires cache input and label text to resolve to the same old-vocab layer. "
                            f"cache_selected_layer={cache_prepared['selected_layer']}, label_selected_layer={label_prepared['selected_layer']}"
                        )

                    cache_model_input_ids = cache_prepared["model_input_ids"]
                    label_model_input_ids = label_prepared["model_input_ids"]
                    input_ids = label_model_input_ids
                    labels = label_prepared["labels"]
                    forward_layer_idx = int(cache_prepared["forward_layer_idx"])

                    cache_seq_len = int(cache_model_input_ids.size(1))
                    label_seq_len = int(label_model_input_ids.size(1))
                    if cache_seq_len < 2 or label_seq_len < 2:
                        skipped_short += 1
                        if skipped_short <= 5:
                            print(
                                f"⚠️ [BlockCache] skip short sample (cache_seq_len={cache_seq_len}, label_seq_len={label_seq_len}). "
                                "This often comes from empty/too-short lines in train_folder/paired_train_folder."
                            )
                        continue

                    capture = {}
                    attention_mask = torch.ones_like(cache_model_input_ids, dtype=torch.long, device=cache_model_input_ids.device)
                    if not isinstance(model, MandelbrotV1ForCausalLM):
                        model.module.last_layer_idx=None if use_dimensionality_reduction else layer_idx
                        model.module.bMax_frequency=use_dimensionality_reduction
                    _ = model(
                        input_ids=cache_model_input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        activation_capture=capture,
                        capture_after_block_idx=int(args.cache_after_block_idx),
                        layer_idx=forward_layer_idx,
                    )

                if "block_hidden_states" not in capture:
                    raise RuntimeError(
                        "Block-output capture failed; expected key block_hidden_states. "
                        "Check cache_after_block_idx and model support."
                    )
                
                block_idx=capture.get("cache_after_block_idx")
                block_hidden = capture["block_hidden_states"].detach().cpu()
                item = {
                    "input_ids": input_ids.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "cache_after_block_idx": block_idx,
                    "block_hidden_states": _to_cache_dtype(block_hidden, args.cache_dtype),
                }

                out_path = os.path.join(args.block_cache_dir, f"block_cache_block_idx_{block_idx}_{cached:06d}.pt")
                torch.save(item, out_path)
                cached += 1
                if cached % 50 == 0:
                    print(f"[BlockCache] wrote {cached} files -> {args.block_cache_dir}")

        if cached == 0:
            raise RuntimeError(
                "Block-output cache generation produced 0 usable samples. "
                f"skipped_short={skipped_short}. "
                "Check for empty/too-short lines in your dataset(s), or increase text length/block_size."
            )
        print(f"✅ Block-output cache generation done: {cached} files in {args.block_cache_dir} (skipped_short={skipped_short})")
        return

    def save_checkpoint_common_logic(model, args, global_step=None, epoch=None,layer_idx: int = -1,hidden_size: int = 128):
        """
        Save the common logic of the checkpoint
        
        Args:
            model: model to save
            Args: Training parameters
            Global_step: Global steps (for step-based saving)
            epoch: Current wheel (for epoch-based saving)
        """
        # DDP: synchronize all ranks before saving (barrier is collective — must be called by all ranks)
        if ddp_enabled and False:
            dist.barrier()

        # Only rank 0 saves checkpoints
        if not is_main_process() and False:
            return

        # Determines the Save Directory
        if global_step is not None:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        elif epoch is not None:
            save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        else:
            raise ValueError("Either global_step or epoch must be provided")
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving checkpoint to {save_dir}")
        
        # Cleanup GPU memory to avoid insufficient memory while saving 
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache() 
        
        model_to_save = model.module if hasattr(model, "module") else model 
        if args.use_block_output_cache: 
            delta = _build_lightweight_delta_state_dict(model_to_save) 
            torch.save(delta, os.path.join(save_dir, "ffn_delta.pt")) 
            with open(os.path.join(save_dir, "ffn_delta_meta.json"), "w", encoding="utf-8") as f: 
                json.dump( 
                    { 
                        "target_layer_idx": int(model_to_save.last_tracked_layer_idx) if hasattr(model_to_save, "last_tracked_layer_idx") else None, 
                        "use_layer_embeddings": bool(getattr(model_to_save, "use_layer_embeddings", False)), 
                        "also_train_lm_head": bool(args.also_train_lm_head), 
                    }, 
                    f, 
                    ensure_ascii=False, 
                    indent=2, 
                ) 
            print("✅ Saved lightweight FFN delta (ffn_delta.pt)") 
            _set_config_torch_dtype(model_to_save) 
            try: 
                model_to_save.config.save_pretrained(save_dir) 
            except Exception as e: 
                print(f"⚠️ Failed to save config for lightweight FFN checkpoint: {e}") 
            _force_saved_config_dtype(save_dir) 
            save_state_dict = _build_state_dict_for_save( 
                model_to_save, 
                trainable_only=True, 
                layer_idx=layer_idx,
                hidden_size=hidden_size,
            ) 
            shard_save_source = save_state_dict if save_state_dict is not None else model_to_save 
        else: 
            _set_config_torch_dtype(model_to_save) 
            save_state_dict = _build_state_dict_for_save( 
                model_to_save, 
                trainable_only=bool( 
                    args.use_block_output_cache 
                ), 
                layer_idx=layer_idx,
                hidden_size=hidden_size,
            ) 
            shard_save_source = save_state_dict if save_state_dict is not None else model_to_save 

            if not (should_skip_full_checkpoint): 
                try: 
                    # Try safetensors format 
                    if save_state_dict is None: 
                        model_to_save.save_pretrained(save_dir, safe_serialization=True) 
                    else: 
                        model_to_save.save_pretrained(save_dir, safe_serialization=True, state_dict=save_state_dict) 
                    print(f"✅ Saved using safetensors format ({save_weight_dtype_mode})") 
                except Exception as e: 
                    print(f"⚠️ Safetensors save failed: {e}") 
                    print("   Falling back to pytorch_model.bin format...") 
                    # Alternative: PyTorch Original Format 
                    try: 
                        if save_state_dict is None: 
                            model_to_save.save_pretrained(save_dir, safe_serialization=False) 
                        else: 
                            model_to_save.save_pretrained(save_dir, safe_serialization=False, state_dict=save_state_dict) 
                        print(f"✅ Saved using pytorch_model.bin format ({save_weight_dtype_mode})") 
                    except Exception as e2: 
                        print(f"⚠️ save_pretrained also failed: {e2}") 
                        # Final backup: direct save state_disc 
                        torch.save( 
                            model_to_save.state_dict() if save_state_dict is None else save_state_dict, 
                            os.path.join(save_dir, "pytorch_model.bin") 
                        ) 
                        try: 
                            model_to_save.config.save_pretrained(save_dir) 
                        except Exception: 
                            pass 
                        print(f"✅ Saved using direct state_dict method ({save_weight_dtype_mode})") 

                        _force_saved_config_dtype(save_dir) 
            else: 
                # Incremental training: do not save a complete model.safetensors file; save only Hugging Face sharded weights (index and shards) 
                try:
                    from modeling_MandelbrotV1 import MandelbrotV1ShardManager 

                    _save_hf_sharded_safetensors_compat( 
                        MandelbrotV1ShardManager, 
                        shard_save_source, 
                        save_dir, 
                        per_ffn=bool(args.shards_per_ffn), 
                        moe_per_expert=bool(args.shards_moe_per_expert), 
                        shards_subdir=str(args.shards_subdir), 
                        dtype=save_weight_dtype, 
                        overwrite=True, 
                    ) 
                     
                    model_to_save.config.save_pretrained(save_dir) 
                except Exception: 
                    pass 
                _force_saved_config_dtype(save_dir) 
                print(f"✅ Incremental: saved HF sharded safetensors (index+shards)")
        
        if global_step is not None:
            print(f"✅ Step checkpoint saved successfully at step {global_step}")
        elif epoch is not None:
            print(f"✅ Epoch checkpoint saved successfully at epoch {epoch}")

        # DDP: ensure all ranks wait for rank 0 to finish saving before continuing training
        if ddp_enabled and False:
            dist.barrier()

    did_print_first_batch = False
    did_warn_short_seq = False

    stop_training = False
    steps_limit_enabled = int(args.max_steps) >= 0
    epochs_limit_enabled = int(args.max_epochs) >= 0
    min_loss_limit_enabled = args.min_loss is not None and float(args.min_loss) >= 0

    count_steps = 0
    effective_num_train_epochs = num_train_epochs
    if is_main_process():
        print(f"[TRAIN] Coming into the training cycle{args.gradient_accumulation_steps} "
              f"| logging_steps={args.logging_steps} "
              f"The first lost output is expected to be in {args.gradient_accumulation_steps * args.logging_steps} After a micro-batch",
              flush=True)
        
    # Replace the previous layer_idx=None with loss_list=None
    def _log_training_step(global_step, epoch, synced_running_loss, steps_since_last_log, current_non_zero_count_ratio, current_indep_loss_ratio, scheduler, start_time, tb_writer, args, avg_perf, dataset=None, loss_list=None):
        """
        The log printing of the training steps and the TensorBoard log logic were sealed.
        """
        avg_loss = synced_running_loss / steps_since_last_log
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        layer_idx=-1
            # --- Add: printing Loss ---
        if loss_list:
            print(f"\n--- Step {global_step} Layer-wise Loss Details ---")
            for item in loss_list:
                layer_idx = item.get("layer_idx")
                # If the dictionary contains tensor, use.item() to convert to numbers; if.item() is passed, use directly
                layer_loss_val = item.get("loss")
                if hasattr(layer_loss_val, 'item'):
                    layer_loss_val = layer_loss_val.item()

                layer_weight = item.get("weight")
                    
                # Formatting output: Distinguishing the final output layer from the middle layer
                layer_name = "Final_Output" if layer_idx == -1 else f"Layer_{layer_idx}"
                print(f"  [{layer_name}] Loss: {layer_loss_val:.4f} Weight: {layer_weight:.4f}")
            print("-" * 40)

        # --- Changed: remove the single layer_idx from the combined output ---
        print(f"step={global_step} epoch={epoch} total_loss={avg_loss:.4f} ppl={ppl:.2f} "
            f"non_zero_ratio={current_non_zero_count_ratio:.4f} indep_ratio={current_indep_loss_ratio:.4f} "
            f"lr={lr:.3e} elapsed={elapsed/60:.1f}m")
        
        if raw_loss_window_total > 0:
            print(f" raw_loss stats: non_tensor={raw_loss_window_non_tensor}/{raw_loss_window_total}, "
                  f"int_one={raw_loss_window_int_one}/{raw_loss_window_total}")

        if args.debug_grad_stats:
            if last_grad_debug_snapshot is None:
                print(" grad_debug: no optimizer-step snapshot yet")
            else:
                def _pct(n, d): return (100.0 * float(n) / float(d)) if d else 0.0
                if last_grad_debug_global_step is not None:
                    print(f" grad_debug snapshot_step={last_grad_debug_global_step}")
                all_s = last_grad_debug_snapshot["all"]
                print(f" grad_debug(all): trainable={all_s['trainable']} "
                      f"grad_present={all_s['grad_present']} ({_pct(all_s['grad_present'], all_s['trainable']):.1f}%) "
                      f"grad_nonzero={all_s['grad_nonzero']} ({_pct(all_s['grad_nonzero'], all_s['trainable']):.1f}%)")
                tgt_s = last_grad_debug_snapshot.get("target")
                if tgt_s is not None:
                    label = debug_grad_target_desc or "target"
                    print(f" grad_debug({label}): trainable={tgt_s['trainable']} "
                          f"grad_present={tgt_s['grad_present']} ({_pct(tgt_s['grad_present'], tgt_s['trainable']):.1f}%) "
                          f"grad_nonzero={tgt_s['grad_nonzero']} ({_pct(tgt_s['grad_nonzero'], tgt_s['trainable']):.1f}%)")
                top_rows = last_grad_debug_snapshot.get("target_topk") or last_grad_debug_snapshot.get("all_topk")
                if top_rows:
                    print(" grad_debug topk ||g||2:")
                    for grad_norm, name in top_rows:
                        print(f" {name}: {grad_norm:.3e}")
        
        if avg_perf:
            print(f" Throughput: {avg_perf.get('avg_tokens_per_sec', 0):.1f} tok/s | "
                  f"Compute: {avg_perf.get('avg_tflops_per_sec', 0):.2f} TFLOP/s | "
                  f"GPU: {avg_perf.get('avg_gpu_util', 0):.0f}%")

        # Communications expenditure statistics
        s = mon.stats()
        if s["calls"] > 0:
            if s["mb"] >= 1.0:
                comm_str = f" | Comm:{s['calls']}calls {s['mb']:.1f}MB {s['comm_ms']:.0f}ms"
            else:
                comm_str = f" | Comm:{s['calls']}calls {s['mb']*1024:.0f}KB {s['comm_ms']:.0f}ms"
        else:
            comm_str = ""
        if comm_str:
            print(f"{comm_str}")

        # Writing TensorBoard
        tb_writer.add_scalar("train/loss", avg_loss, global_step)
        tb_writer.add_scalar("train/ppl", ppl, global_step)
        tb_writer.add_scalar("train/lr", lr, global_step)
        tb_writer.add_scalar("train/layer_idx", layer_idx, global_step) # <-- Added: record layer_idx
        tb_writer.add_scalar("train/non_zero_count_ratio", current_non_zero_count_ratio, global_step)
        tb_writer.add_scalar("train/indep_loss_ratio", current_indep_loss_ratio, global_step)
        
        if avg_perf:
            tb_writer.add_scalar("performance/tokens_per_sec", avg_perf.get('avg_tokens_per_sec', 0), global_step)
            tb_writer.add_scalar("performance/tflops_per_sec", avg_perf.get('avg_tflops_per_sec', 0), global_step)
            tb_writer.add_scalar("performance/ms_per_token", avg_perf.get('avg_ms_per_token', 0), global_step)
            if 'avg_gpu_util' in avg_perf:
                tb_writer.add_scalar("performance/gpu_util", avg_perf.get('avg_gpu_util', 0), global_step)
                tb_writer.add_scalar("performance/memory_used_gb", avg_perf.get('avg_memory_used_gb', 0), global_step)
            if 'avg_power_watts' in avg_perf:
                tb_writer.add_scalar("performance/power_watts", avg_perf.get('avg_power_watts', 0), global_step)
                tb_writer.add_scalar("performance/temperature_c", avg_perf.get('avg_temperature_c', 0), global_step)

        if dataset is not None:
            dataset.model=model
            dataset._wait_for_memory_available()

    for epoch in range(resume_epoch, effective_num_train_epochs):
        for batch in dataloader:
            # Let getlogitswrapper get the current step

            count_steps += 1
            #print(f"\n=== Epoch {epoch+1}/{effective_num_train_epochs}, Step {count_steps} === batch.size={batch.size()}")
            global_vars["global_step"] = global_step


            # Point("start a battling in dataloade", batt)
            if args.use_block_output_cache:
                input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer = use_block_output_cache(batch)
                
            else:
                if isinstance(batch, dict) and ("input_ids" in batch) and ("labels" in batch):
                    input_local = batch["input_ids"].to(device, non_blocking=True)
                    labels_local = batch["labels"].to(device, non_blocking=True)

                    if use_dimensionality_reduction:
                        # Convert local->global for input_ids (vectorized, no GPU→CPU sync)
                        input_global = tokenizer_manager._local_to_global_ids(layer_idx, input_local.view(-1))
                        input_ids = get_inputs(input_global).to(device, non_blocking=True)

                    # Convert labels local->global but preserve -100 mask
                    labels = labels_local.clone()
                    mask = labels != -100
                    if torch.any(mask):
                        tmp = labels.clone()
                        tmp[~mask] = 0
                        tmp_global = tokenizer_manager._local_to_global_ids(layer_idx, tmp.view(-1))
                        labels = tmp_global.view_as(labels)
                        labels[~mask] = -100
                else:
                    input_ids = batch.to(device, non_blocking=True)
                    if use_dimensionality_reduction:
                        # Convert Layer 0 local IDs to global IDs (vectorized, with no GPU-to-CPU synchronization)
                        global_ids = tokenizer_manager._local_to_global_ids(layer_idx, input_ids.view(-1))
                        input_ids = get_inputs(global_ids).to(device, non_blocking=True)
                    labels = input_ids.clone()
            # Add to Data Loading Part
            if args.save_images and epoch % args.log_interval == 0:
                img_grid = torchvision.utils.make_grid(input_ids[:4])
                tb_writer.add_image('train_samples', img_grid, epoch)            

            # Acquire in advance whether logs are needed (before collecting loss)
            should_log = (global_step % args.logging_steps == 0 and global_step > 0)
            if should_log:
                global_vars["is_Loss_Log"] = False
            else:
                global_vars["is_Loss_Log"] = False
            
            # Verify input_ids legitimacy and show layer distribution
            if global_step == 0 and not did_print_first_batch :
                min_id = input_ids.min().item()
                max_id = input_ids.max().item()
                total_vocab = tokenizer_manager.total_vocab_size
                if min_id < 0 or max_id >= total_vocab:
                    print(f"⚠️ Warning: Illegal token ID detected. min={min_id}, max={max_id}, vocab_size={total_vocab}")
                
                # Show first batch layer distribution
                print(f"\nAutoselect the optimal layer using the highest_with_content policy")
                layer_dist = {}
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                for token_id in input_ids[0].tolist():
                    if use_dimensionality_reduction:
                        layer_id = tokenizer_manager.get_layer_from_token_id(token_id)
                    else:
                        layer_id,_ = tokenizer_manager.get_old_layer_from_token_id(layer_idx,token_id,True)
                    layer_dist[layer_id] = layer_dist.get(layer_id, 0) + 1
                print(f"   The first catch is token-level distribution: {dict(sorted(layer_dist.items()))}")
                print(f"   Total {sum(layer_dist.values())} Token, distributed over {len(layer_dist)} layers")
                
                # Show the top 10 token examples
                print(f"   The first 10 examples of Token:")
                for i, token_id in enumerate(input_ids[0][:10].tolist()):
                    if use_dimensionality_reduction:
                        layer_id = tokenizer_manager.get_layer_from_token_id(token_id)
                        local_id = tokenizer_manager._global_to_local_id(layer_id,token_id)        
                    else:
                        layer_id,local_id = tokenizer_manager.get_old_layer_from_token_id(layer_idx,token_id,True)
                   
                    if use_dimensionality_reduction:
                        #text = tokenizer_manager.tokenizers[layer_id].decode([local_id])
                        text = tokenizer_manager.tokenizers[layer_id].id_to_token(local_id)
                    else:
                        text = tokenizer_manager.tokenizers[layer_id].old_id_to_token(token_id,True)
          
                    print(f"     Token-{i}: [L{layer_id}] {repr(text)}")
                did_print_first_batch = True
            
            # Guard: seq_len too short can lead to empty loss (e.g. shift makes 0 targets) and produce NaN.
            if input_ids.dim() != 2:
                input_ids = input_ids.view(input_ids.size(0), -1)
            if labels is not None and labels.dim() != 2:
                labels = labels.view(labels.size(0), -1)

            seq_len = int(input_ids.size(1))
            if seq_len < 2:
                if not did_warn_short_seq:
                    print(
                        f"⚠️ Batch seq_len={seq_len} is too short; loss may be NaN. "
                        "This usually means empty/too-short lines or cache files with only BOS."
                    )
                    did_warn_short_seq = True
                if args.is_Distribution and args.distribution_Type !="client" and input_ids[0]==dataloader.dataset._get_pad_token_id():
                    stop_training=True
                    break
                else:
                    continue

            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

            # Only multistep decoding token for neuron_track_steps (10K+ token per step decoding is extremely expensive)
            _do_track = (global_step % args.neuron_track_steps == 0)
            if _do_track and args.enable_neuron_check:
                activation_tracker.start_new_step(global_step, input_ids=input_ids, 
                                            tokenizer_manager=tokenizer_manager, 
                                            layer_idx=layer_idx,use_dimensionality_reduction=use_dimensionality_reduction)

            # Recording the start of the step (for performance analysis)
            step_start_time = time.time()
            
            with torch.amp.autocast(**autocast_kwargs):
                if args.use_block_output_cache:
                    outputs,_= use_block_cache_for_generation(input_ids, labels, cache_block_hidden, cache_after_block_idx, cache_selected_layer)   
        
                else:
                    use_dimensionality_reduction=bool(getattr(config, "use_dimensionality_reduction", True)) 
                    forward_layer_idx = layer_idx if not use_dimensionality_reduction else None
                    if not isinstance(model, MandelbrotV1ForCausalLM):
                        model.module.last_layer_idx=None if use_dimensionality_reduction else layer_idx
                        model.module.bMax_frequency=use_dimensionality_reduction
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        layer_idx=forward_layer_idx,
                        use_cache=False,
                    )

                if isinstance(outputs, dict):
                    loss = outputs.get("loss", None)
                    loss_list=outputs.get("loss_list", None)
                else:
                    loss = getattr(outputs, "loss", None)
                    loss_list=getattr(outputs, "loss_list", None)

                raw_loss_window_total += 1
                if not torch.is_tensor(loss):
                    raw_loss_window_non_tensor += 1
                    if isinstance(loss, int) and loss == 1:
                        raw_loss_window_int_one += 1

                if loss is None or loss_list is None:
                    raise RuntimeError(
                        f"Model output does not contain loss. type={type(outputs)} keys={list(outputs.keys()) if isinstance(outputs, dict) else 'n/a'}"
                    )

                # In block-output-cache mode, always optimize standard shifted CE on logits.
                # This avoids model-side incremental branches returning constant/non-tensor loss (e.g. 1).
              
                #loss = _normalize_or_recompute_loss(loss, outputs, labels)
                
                # Collect and reset loss in training code (only log steps, avoid going through all moE layers at every step)
                if should_log:
                    non_zero_count_loss, final_indep_loss = collect_and_reset_losses(model, device)
                else:
                    non_zero_count_loss, final_indep_loss = reset_non_zero_count_loss_and_indep_loss(model)
            
            # Complete activation tracking for the current step (only when start_new_step was called)
            if _do_track and args.enable_neuron_check:
                activation_tracker.finalize_step()
            
            # Show multilayered token information for internal recoding of model (for comparison)
            if global_step % args.neuron_track_steps == 0 and global_step > 0:
                if hasattr(model, 'mor_Layer_ids') and model.mor_Layer_ids:
                    print(f"\n{'='*80}")
                    print(f"model Internal multilayer token information (recoded)")
                    print(f"{'='*80}")
                    for layer_key, word_indices in model.mor_Layer_ids.items():
                        layer_id = layer_key  # Assume layer_key is the layer number
                        token_count = len(word_indices)
                        print(f"Layer {layer_id}: {token_count} tokens")
                        # Show Previous Tokens
                        if token_count > 0:
                            sample_tokens = []
                            for pos_idx, token_id in list(word_indices.items())[:5]:
                                try:
                                    text = tokenizer_manager.tokenizers[layer_id].decode([token_id])
                                    sample_tokens.append(f"pos{pos_idx}={repr(text)}")
                                except:
                                    sample_tokens.append(f"pos{pos_idx}=<id:{token_id}>")
                            print(f"  Example: {', '.join(sample_tokens)}")
                    print(f"{'='*80}\n")
            
            # Diagnostics: check the distribution of the input sequences
            if global_step % args.neuron_track_steps == 0 and global_step > 0:
                try:
                    layer_distribution = {}
                    for token_id in input_ids[0][:10].tolist():  # Ten tokens before check
                        layer_id = tokenizer_manager.get_layer_from_token_id(token_id)
                        layer_distribution[layer_id] = layer_distribution.get(layer_id, 0) + 1
                    print(f"\nEnter the sequence layer distribution (first 10 token): {layer_distribution}")
                    
                    # Show complete token sequences (first 20)
                    token_texts_sample = []
                    for token_id in input_ids[0][:20].tolist():
                        layer_id = tokenizer_manager.get_layer_from_token_id(token_id)
                        local_id = token_id - tokenizer_manager.offsets[layer_id]
                        text = tokenizer_manager.tokenizers[layer_id].decode([local_id])
                        token_texts_sample.append(f"[L{layer_id}]{repr(text)}")
                    print(f"   Full sequence examples: {' '.join(token_texts_sample[:20])}")
                except Exception as e:
                    print(f"⚠️ Dial Distribution Failed: {e}")
            
            # Check if hooks is working after the first step.
            if global_step == 0 and args.enable_neuron_check:
                try:
                    # Check token decoding.
                    if 0 in activation_tracker.token_texts:
                        token_sample = activation_tracker.token_texts[0][:5]
                        print(f"\nToken decoding check: {token_sample}")
                    
                    print(f"\nFirst step after check: collected {len(neuron_checker.activation_stats)} activation data")
                    if len(neuron_checker.activation_stats) == 0:
                        print("⚠️ Warning: No activation DATA collected after the first step")
                        print("   Possible causes:")
                        print("   1. MoE door control not selected for expert")
                        print("   2. The model structure does not match expectations")
                        print("   hooks registration failure")
                    else:
                        first_layer = list(neuron_checker.activation_stats.keys())[0]
                        first_stats = neuron_checker.activation_stats[first_layer]
                        print(f"   Example: {first_layer}")
                        print(f"   - Number of samples collected: {first_stats.get('total_samples', 0)}")
                        nonzero_per_neuron = first_stats.get('nonzero_count_per_neuron')
                        if nonzero_per_neuron is not None:
                            print(f"   - neurons: {nonzero_per_neuron.shape[0]}")
                            print(f"   - Nonzero activations: {nonzero_per_neuron.sum().item():.0f}")
                except Exception as e:
                    print(f"⚠️ First step of the check-up: {e}")

            
            

            loss = loss / args.gradient_accumulation_steps

            loss_for_log = loss.detach()

            if not torch.isfinite(loss):
                save_checkpoint_common_logic(model, args, global_step=global_step, epoch=epoch,layer_idx=layer_idx,hidden_size=config.hidden_size)
                #save_checkpoint(model, optimizer, epoch, global_step, os.path.join(args.ckpt_dir, f"nan_step{global_step}.pt"))
                print("[ERROR] loss not limited value, exit training")
            else:

                avg_perf = perf_profiler.get_average_metrics(last_n_steps=args.logging_steps)
                
                # _log_training_step(global_step, epoch, loss, loss_for_log,
                #                    1, 1, scheduler,
                #                    start_time, tb_writer, args, avg_perf, dataset=dataloader.dataset, loss_list=loss_list)

                if args.render_torchviz_graphs and global_vars["is_Loss_Log"]:
                    dot = make_dot(loss, params=dict(model.named_parameters()))
                    dot.render(args.output_dir+"/model_graph"+str(global_step), format="png")
                #with torch.autograd.detect_anomaly():
                
                if loss.grad_fn is not None:
                    scaler.scale(loss).backward()

            
 
            
            if ((global_step + 1) % args.gradient_accumulation_steps) == 0:
                            # Optional gradient Parameter Test
                if args.grad_norm_threshold is not None:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += (p.grad.data.norm(2).item() ** 2)
                    total_norm = total_norm ** 0.5
                    if total_norm > args.grad_norm_threshold:
                        save_checkpoint_common_logic(model, args, global_step=global_step, epoch=epoch,layer_idx=layer_idx,hidden_size=config.hidden_size)
                        #save_checkpoint(model, optimizer, epoch, global_step, os.path.join(args.ckpt_dir, f"grad_explode_{global_step}.pt"))
                        print(f"[ERROR] gradient EXPLAMINATION: {total_norm:.2f} > {args.grad_norm_threshold}; exiting training")
                        break
                
                # gradient clipping
                scaler.unscale_(optimizer)
                if args.debug_grad_stats:
                    last_grad_debug_snapshot = _make_grad_debug_snapshot(
                        model,
                        topk=int(args.debug_grad_topk),
                        target_prefixes=debug_grad_target_prefixes,
                    )
                    last_grad_debug_global_step = int(global_step + 1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Scheduler updates every step without relying on gradient_accumulation
            scheduler.step()

            running_loss += loss_for_log * args.gradient_accumulation_steps
            current_non_zero_count_ratio = non_zero_count_loss if isinstance(non_zero_count_loss, float) else 0.0
            current_indep_loss_ratio = final_indep_loss if isinstance(final_indep_loss, float) else 0.0
            steps_since_last_log += 1

            # DDP: sync running_loss across ranks only at logging steps (avoid per-step all_reduce)
            synced_running_loss = running_loss
            if ddp_enabled and should_log:
                # NCCL backend requires that tansor must be on CUDA devices; using current_device to ensure correct
                _sync_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                rl_tensor = torch.tensor(running_loss, dtype=torch.float32, device=_sync_device)
                if not args.is_Distribution and args.distribution_Type =="local":
                    dist.all_reduce(rl_tensor, op=dist.ReduceOp.SUM)
                synced_running_loss = rl_tensor.item() / get_world_size()
            
            # Document performance indicators
            step_elapsed_time = time.time() - step_start_time
            batch_size, seq_len = input_ids.shape
            # Recording micro-batch properties (tok/s for global calculations)
            perf_records.append((batch_size * seq_len, step_elapsed_time))
            perf_metrics = perf_profiler.record_step(
                batch_size=batch_size,
                seq_len=seq_len,
                elapsed_time=step_elapsed_time,
                loss=loss.item(),
                global_step=global_step,
                world_size=get_world_size()
            )
            early_stop= False
            if  should_log:
                # val_loss, val_metric = evaluate(model, dataloader, device, tokenizer_manager)
                avg_loss = synced_running_loss / args.logging_steps
                val_loss = avg_loss  # Here, training losses are used simply as placeholders to verify losses.
                tb_writer.add_scalar('valid/loss', val_loss, global_step)
                # print(f"[EVAL] step={global_step} val_loss={val_loss:.4f}")

                def _save_checkpoint(model, optimizer, epoch, global_step, path):
                    save_checkpoint_common_logic(model, args, global_step=global_step, epoch=epoch,layer_idx=layer_idx,hidden_size=config.hidden_size)

                path=os.path.join(args.output_dir, args.ckpt_dir)
                if early.step(val_loss, model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, ckpt_dir=path):
                    print(f"[TRAIN] Early stop triggered at step {global_step}, epoch {epoch}")
                    tb_writer.close()
                    early_stop= True
 
                avg_loss = synced_running_loss / steps_since_last_log
                ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                
                # Access to average performance indicators
                avg_perf = perf_profiler.get_average_metrics(last_n_steps=args.logging_steps)
                
                _log_training_step(global_step, epoch, synced_running_loss, steps_since_last_log,
                                   current_non_zero_count_ratio, current_indep_loss_ratio, scheduler,
                                   start_time, tb_writer, args, avg_perf, dataset=dataloader.dataset, loss_list=loss_list)
                
                # Recording the latest los and ppl for the last hparams
                final_loss = avg_loss
                final_ppl = ppl
                
                running_loss = 0.0
                raw_loss_window_total = 0
                raw_loss_window_non_tensor = 0
                raw_loss_window_int_one = 0
                steps_since_last_log = 0
                # Reset Communication Monitor
                mon.comm_calls = 0; mon.comm_bytes = 0; mon.comm_cpu_us = 0.0
                perf_records.clear()
            # Record the parameter distribution once per epoch
            # for name, param in model.named_parameters():
            #     tb_writer.add_histogram(name, param, epoch) 

            stop_step = global_step%args.logging_steps
            stop, reason = should_stop_hard(
                epoch,
                global_step,
                start_time,
                synced_running_loss / (stop_step if stop_step != 0 else 1),
                min_loss=args.min_loss if min_loss_limit_enabled else None,
                max_epochs=args.max_epochs if epochs_limit_enabled else None,
                max_steps=args.max_steps if steps_limit_enabled else None,
                max_seconds=max_seconds,
            )

            if args.is_Distribution and args.distribution_Type != "client":
                if dataloader.dataset.training_finished_event.is_set():
                    stop = True
                    early_stop=True
                    reason = "Training finished by server."
                else:
                    stop = False
                    early_stop = False

            if stop or early_stop :
                print(f"[TRAIN] Hard stop: {reason}")
                # Print final neurons health report before the end of the training
                print("\n" + "="*80)
                print("Final Neuron Health Report")
                print("="*80)
                if args.enable_neuron_check:
                    neuron_checker.print_summary()
                # Do not remove hooks here. Undoing at the end.
                tb_writer.close()
                stop_training = True
                break            
            
            # checkpoint
            if global_step > 0 and global_step % args.save_steps == 0 and args.save_steps > 0:
                save_checkpoint_common_logic(model, args, global_step=global_step,layer_idx=layer_idx,hidden_size=config.hidden_size)
        
                print(f"✅ Checkpoint saved successfully")
            global_step += 1

            # [TRAIN] Client {MandelbrotV1ForCausalLM.client_Worker_id} Completed a battling training
        # Epoch-level checkpointing: reuse the same save logic as step-based checkpoints
        # Do not create a new helper; enable epoch-save behavior when --max_steps == -1.
        if int(getattr(args, "max_steps", 0)) == -1:
            save_epochs = int(getattr(args, "save_epochs", 1) or 1)
            # relative epoch count since resume (1-based)
            rel_epoch = (epoch - resume_epoch + 1)
            if save_epochs > 0 and (rel_epoch % save_epochs == 0):
                save_checkpoint_common_logic(model, args, epoch=epoch,layer_idx=layer_idx,hidden_size=config.hidden_size)

                print(f"✅ Epoch checkpoint saved successfully")

        dataloader.dataset._wait_for_memory_available()

        if stop_training:
            avg_perf = perf_profiler.get_average_metrics(last_n_steps=args.logging_steps)
            _log_training_step(global_step, epoch, synced_running_loss, (stop_step if stop_step != 0 else 1),
                                   current_non_zero_count_ratio, current_indep_loss_ratio, scheduler,
                                   start_time, tb_writer, args, avg_perf, dataset=dataloader.dataset, loss_list=loss_list)
            break

    
    if args.is_Distribution:
        if args.distribution_Type == "client":
            try:
                req = mandelbrot_service_pb2.SignalRequest(
                    worker_id=dataloader.dataset.client_worker_id,
                    signal_name="TRAINING_FINISH",
                    message=f"worker:{dataloader.dataset.client_worker_id}, final_step:{global_step}",
                    timestamp=int(time.time())
                )
                sub=_worker_context['grpc_stub']
                # Send it as a guardian line to prevent blocking the main process from exiting
                if sub:
                    threading.Thread(target=lambda: sub.SendSignal(req, timeout=5), daemon=True).start()
                print(f"[Client {dataloader.dataset.client_worker_id}] Training finished, TRAINING_FINISH signal reported.")
            except Exception as e:
                print(f"Sending TRANING_FINISH signal failed: {e}")
        else:
            dataloader.dataset.close()
    

    # final save    
    os.makedirs(final_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model_to_save = model.module if hasattr(model, "module") else model
    
    _set_config_torch_dtype(model_to_save)
    save_state_dict = _build_state_dict_for_save(
        model_to_save,
        trainable_only=bool(
            args.use_block_output_cache 
        ),
        layer_idx=layer_idx,
        hidden_size=config.hidden_size,
    )
    shard_save_source = save_state_dict if save_state_dict is not None else model_to_save

    if not (should_skip_full_checkpoint):
        try:
            # Try safetensors format
            if save_state_dict is None:
                model_to_save.save_pretrained(final_dir, safe_serialization=True)
            else:
                model_to_save.save_pretrained(final_dir, safe_serialization=True, state_dict=save_state_dict)
            print(f"✅ Final model saved using safetensors format ({save_weight_dtype_mode})")
        except Exception as e:
            print(f"⚠️ Safetensors save failed: {e}")
            print("   Falling back to pytorch_model.bin format...")
            try:
                if save_state_dict is None:
                    model_to_save.save_pretrained(final_dir, safe_serialization=False)
                else:
                    model_to_save.save_pretrained(final_dir, safe_serialization=False, state_dict=save_state_dict)
                print(f"✅ Final model saved using pytorch_model.bin format ({save_weight_dtype_mode})")
            except Exception as e2:
                print(f"⚠️ save_pretrained also failed: {e2}")
                torch.save(
                    model_to_save.state_dict() if save_state_dict is None else save_state_dict,
                    os.path.join(final_dir, "pytorch_model.bin")
                )
                try:
                    model_to_save.config.save_pretrained(final_dir)
                except Exception:
                    pass
                print(f"✅ Final model saved using direct state_dict method ({save_weight_dtype_mode})")

        _force_saved_config_dtype(final_dir)
    else:
        # incremental or cache mode: save HF reviewed safetyensors (index + shards) only, not generating full mdel. safenensors
        try:
            from modeling_MandelbrotV1 import MandelbrotV1ShardManager

            _save_hf_sharded_safetensors_compat(
                MandelbrotV1ShardManager,
                shard_save_source,
                final_dir,
                per_ffn=bool(args.shards_per_ffn),
                moe_per_expert=bool(args.shards_moe_per_expert),
                shards_subdir=str(args.shards_subdir),
                dtype=save_weight_dtype,
                overwrite=True,
            )
            try:
                model_to_save.config.save_pretrained(final_dir)
            except Exception:
                pass
            _force_saved_config_dtype(final_dir)
            print(f"✅ Incremental: saved HF sharded safetensors (index+shards) to: {final_dir}")
        except Exception as e:
            print(f"⚠️ Incremental final sharded save failed: {e}")
            print("   Falling back to direct state_dict save...")
            try:
                torch.save(
                    model_to_save.state_dict() if save_state_dict is None else save_state_dict,
                    os.path.join(final_dir, "pytorch_model.bin")
                )
                try:
                    model_to_save.config.save_pretrained(final_dir)
                except Exception:
                    pass
                print(f"✅ Final saved using direct state_dict method ({save_weight_dtype_mode})")
            except Exception as e2:
                print(f"⚠️ Incremental final fallback save also failed: {e2}")

    # Optional final sharded save (rule-based shards + HF index)
    if args.save_shards:
        try:
            from modeling_MandelbrotV1 import MandelbrotV1ShardManager

            _save_hf_sharded_safetensors_compat(
                MandelbrotV1ShardManager,
                shard_save_source,
                final_dir,
                per_ffn=bool(args.shards_per_ffn),
                moe_per_expert=bool(args.shards_moe_per_expert),
                shards_subdir=str(args.shards_subdir),
                dtype=save_weight_dtype,
                overwrite=True,
            )
            print("✅ Final HF sharded safetensors saved (index + shards)")

            if bool(args.also_save_block_shards) and bool(args.shards_per_ffn):
                block_dir = os.path.join(final_dir, str(args.block_shards_dirname))
                os.makedirs(block_dir, exist_ok=True)
                _save_hf_sharded_safetensors_compat(
                    MandelbrotV1ShardManager,
                    shard_save_source,
                    block_dir,
                    per_ffn=False,
                    moe_per_expert=bool(args.shards_moe_per_expert),
                    shards_subdir=str(args.shards_subdir),
                    dtype=save_weight_dtype,
                    overwrite=True,
                )
                try:
                    model_to_save.config.save_pretrained(block_dir)
                except Exception:
                    pass
                print(f"✅ Final per-block sharded safetensors saved to: {block_dir}")
        except Exception as e:
            print(f"⚠️ Final HF sharded save failed: {e}")
    
    tokenizer.save_pretrained(final_dir)
    # Final trainer state save: skip for incremental expansion runs.
    if not (is_incremental_expand):
        torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "precision_mode": precision_mode,
            "use_fp16": int(use_fp16),
            "use_bf16": int(use_bf16),
            "global_step": global_step,
            "epoch": epoch,
        }, os.path.join(final_dir, "trainer_state.pt"))
    else:
        print(f"ℹ️ Skipping final trainer_state.pt save for incremental training (dir={final_dir})")

    # Update the end indicator for hyperparameters
    try:
        hparam_dict = {
            'learning_rate': args.learning_rate,
            'batch_size': args.per_device_train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'block_size': args.block_size,
            'precision_mode': precision_mode,
            'use_fp16': int(use_fp16),
            'use_bf16': int(use_bf16),
            'save_weight_dtype': save_weight_dtype_mode,
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'max_grad_norm': args.max_grad_norm,
        }
        metric_dict = {
            'hparams/final_loss': final_loss,
            'hparams/final_ppl': final_ppl,
            'hparams/final_id': float(final_id) if final_id is not None else 0.0,
            'hparams/total_steps': global_step,
        }
        # Write again to update the final indicator
        tb_writer.add_hparams(hparam_dict, metric_dict)
        print(f"Final hyperparameters updated (loss={final_loss:.4f}, ppl={final_ppl:.2f}, steps={global_step})")
    except Exception as e:
        print(f"Failed to update final hyperparameters: {e}")

    # Save ID/ Valid Dimensions after training and remove hooks
    if id_analyzer is not None:
        try:
            id_analyzer.save_history(os.path.join(final_dir, "intrinsic_dimension_history.json"))
        except Exception as e:
            print(f"⚠️ Failed to save ID history: {e}")
        try:
            id_analyzer.save_effdim_history(os.path.join(final_dir, "effective_dimension_history.json"))
        except Exception as e:
            print(f"⚠️ Failed to save a valid dimension history: {e}")
        id_analyzer.remove_hooks()

    if args.enable_neuron_check:
        # Print final neurons health report at the end of the training
        print("\n" + "="*80)
        print("Training completed - Final neurons Health Report")
        print("="*80)
        neuron_checker.print_summary()
    
    # Print final performance statistics
    print("\n" + "="*80)
    print("Training Complete - Performance Statistics Report")
    print("="*80)
    mon.stop()
    print("✅ Communication monitor stopped.")
    perf_profiler.print_summary(last_n_steps=500)
    
    # Save weight-matrix rank history (only when enabled)
    if args.enable_rank_analysis and rank_analyzer is not None and len(rank_analyzer.rank_history) > 0:
        rank_history_path = os.path.join(final_dir, "weight_rank_history.json")
        rank_analyzer.save_rank_history(rank_history_path)
        
        # Print the rank trend analysis
        print("\n" + "="*80)
        print("weight-matrix rank Trend Analysis")
        print("="*80)
        steps = sorted(rank_analyzer.rank_history.keys())
        if len(steps) >= 2:
            first_step = steps[0]
            last_step = steps[-1]
            
            first_stats = rank_analyzer._compute_rank_statistics(rank_analyzer.rank_history[first_step])
            last_stats = rank_analyzer._compute_rank_statistics(rank_analyzer.rank_history[last_step])
            
            print(f"\nInitial training (step) {first_step}):")
            print(f"   Average rank RATIO: {first_stats['avg_rank_ratio']:.4f}")
            print(f"   Full-rank matrices: {first_stats['full_rank_count']}")
            print(f"   Low-rank matrices: {first_stats['low_rank_count']}")
            
            print(f"\nPost-training (step) {last_step}):")
            print(f"   Average rank RATIO: {last_stats['avg_rank_ratio']:.4f}")
            print(f"   Full-rank matrices: {last_stats['full_rank_count']}")
            print(f"   Low-rank matrices: {last_stats['low_rank_count']}")
            
            rank_ratio_change = last_stats['avg_rank_ratio'] - first_stats['avg_rank_ratio']
            print(f"\nTrends in change:")
            if rank_ratio_change < -0.05:
                print(f"    rank Ratio drops on average {abs(rank_ratio_change):.4f} - Could be model degradation.")
            elif rank_ratio_change > 0.05:
                print(f"    Average rank Ratio up {rank_ratio_change:.4f} - model expressiveness enhanced")
            else:
                print(f"    Average rank RATIO (basic stability){rank_ratio_change:+.4f})")
            
            print(f"={'='*80}\n")

    if args.enable_neuron_check:
        # Save detailed report
        try:
            final_report = neuron_checker.get_detailed_report()
            report_path = os.path.join(final_dir, "final_neuron_health_report.json")
            
            # Clear non-sequencable objects in the report
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            serializable_report = make_serializable(final_report)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(serializable_report, f, indent=2, ensure_ascii=False)
            print(f"✅ neurons Health Report saved: {report_path}")
        except Exception as e:
            print(f"⚠️ Failed to save neurons health report: {e}")
            import traceback
            traceback.print_exc()
    
        # Clear hooks
        neuron_checker.remove_hooks()
        activation_tracker.remove_hooks()
    
    # Save the final activation tracking records (only the last step)
    if args.enable_neuron_check and len(activation_tracker.activation_records) > 0:
        final_activation_file = os.path.join(final_dir, "final_neuron_activations.json")
        # Only last step records.
        recent_steps = sorted(activation_tracker.activation_records.keys())
        if recent_steps:
            latest_step = recent_steps[-1]
            activation_tracker.save_to_file(final_activation_file, step=latest_step)
            print(f"✅ Saved final neuron activation records (step {latest_step}) to {final_activation_file}")
        
        # Print activation-frequency statistics
        print(f"\n{'='*80}")
        print(" Final neurons activation FREQUENCY Statistics")
        print(f"{'='*80}")
        freq_stats = activation_tracker.get_neuron_activation_frequency()
        for layer_name, neuron_freq in sorted(freq_stats.items()):
            top_neurons = sorted(neuron_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n{layer_name}:")
            print(f"  Total activated neurons: {len(neuron_freq)}")
            print(f"  Top 10 most active neurons:")
            for neuron_idx, freq in top_neurons:
                print(f"     neurons  {neuron_idx}: activation  {freq} times")
    
    tb_writer.close()
    print("Training finished. Saved to", final_dir)

    # Clean up DDP
    if ddp_enabled:
        cleanup_ddp()

def evaluate(model, valid_loader, device='cuda', tokenizer_manager=None, layer_idx=0):
    total = 0.0
    n_seen = 0
    for batch in valid_loader:
        input_ids = batch.to(device, non_blocking=True)
        global_ids = tokenizer_manager._local_to_global_ids(layer_idx, input_ids.view(-1))
        input_ids = global_ids.view(1, -1)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels, layer_idx=layer_idx)
        loss = outputs.loss
        bs = input_ids.size(0)
        total += loss.item() * bs
        n_seen += bs
    val_loss = total / max(1, n_seen)
    print(" val_loss=", val_loss)
    return val_loss, None

if __name__ == "__main__":
    # print(f"MAIN start pid={os.getpid()}", flush=True)
    main()