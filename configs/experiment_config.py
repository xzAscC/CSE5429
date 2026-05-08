from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentConfig:
    name: str  # Unique identifier, e.g., "c0_baseline"
    display_name: str  # Human-readable name, e.g., "C0: Baseline"
    description: str  # What this config tests
    llm_kwargs: dict[str, Any]  # kwargs passed to vllm.LLM constructor
    batch_mode: bool = False  # True for C5 (batch processing config)
    num_runs: int = 1  # >1 for C9 (determinism check)
    optional: bool = False  # True if config may fail (e.g., FP8)
    paper_theme: str = ""  # Which paper question this addresses


def get_all_configs(
    model: str = "Qwen/Qwen2-1.5B-Instruct",
    max_model_len: int = 2048,
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="c0_baseline",
            display_name="C0: Baseline (No Optimizations)",
            description="Standard vLLM inference with all optimizations disabled. Reference point for comparison.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="Reference baseline",
        ),
        ExperimentConfig(
            name="c1_chunked_prefill_small",
            display_name="C1: Chunked Prefill (512 tokens)",
            description="Enables chunked prefill with max_num_batched_tokens=512. Tests execution chunking effect.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 512,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="Execution chunking",
        ),
        ExperimentConfig(
            name="c2_chunked_prefill_large",
            display_name="C2: Chunked Prefill (2048 tokens)",
            description="Enables chunked prefill with max_num_batched_tokens=2048. Tests larger chunk size effect.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 2048,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="Execution chunking",
        ),
        ExperimentConfig(
            name="c3_cuda_graphs",
            display_name="C3: CUDA Graphs Enabled",
            description="Enables CUDA graph capture/replay. Tests kernel execution path differences.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": False,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="Kernel execution path",
        ),
        ExperimentConfig(
            name="c4_prefix_caching",
            display_name="C4: Prefix Caching Enabled",
            description="Enables KV cache reuse for shared prefixes. Tests KV cache reuse effect.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": True,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="KV cache reuse",
        ),
        ExperimentConfig(
            name="c5_batch_processing",
            display_name="C5: Batch Processing",
            description="Runs prompts in a batch rather than individually. Tests parallel scheduling effects.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            batch_mode=True,
            paper_theme="Parallel scheduling",
        ),
        ExperimentConfig(
            name="c6_low_memory",
            display_name="C6: Low GPU Memory (40%)",
            description="Reduces GPU memory utilization to 40%. Tests memory pressure and scheduling effects.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.4,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            paper_theme="Memory pressure",
        ),
        ExperimentConfig(
            name="c7_fp8_kv_cache",
            display_name="C7: FP8 KV Cache",
            description="Uses FP8 precision for KV cache. Tests KV cache quantization effect. May not be supported on all GPUs.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "fp8_e5m2",
                "swap_space": 0,
                "seed": 42,
            },
            optional=True,
            paper_theme="KV cache precision",
        ),
        ExperimentConfig(
            name="c8_cpu_swap",
            display_name="C8: CPU Swap Space",
            description="Enables CPU swap space for KV cache offloading. Tests KV cache location (GPU vs CPU) effect.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 4,
                "seed": 42,
            },
            paper_theme="KV cache location",
        ),
        ExperimentConfig(
            name="c9_determinism",
            display_name="C9: Determinism Check (3 runs)",
            description="Runs baseline config 3 times to verify deterministic output. Establishes repeatability baseline.",
            llm_kwargs={
                "model": model,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "swap_space": 0,
                "seed": 42,
            },
            num_runs=3,
            paper_theme="Repeatability baseline",
        ),
    ]


def get_config_by_name(name: str) -> ExperimentConfig | None:
    for config in get_all_configs():
        if config.name == name:
            return config
    return None
