from __future__ import annotations

import pytest

from configs.experiment_config import (
    ExperimentConfig,
    get_all_configs,
    get_config_by_name,
)


@pytest.fixture
def all_configs() -> list[ExperimentConfig]:
    return get_all_configs()


def _find(all_configs: list[ExperimentConfig], name: str) -> ExperimentConfig:
    for c in all_configs:
        if c.name == name:
            return c
    raise KeyError(name)


def test_get_all_configs_returns_10(all_configs: list[ExperimentConfig]) -> None:
    assert len(all_configs) == 10


def test_config_names_unique(all_configs: list[ExperimentConfig]) -> None:
    names = [c.name for c in all_configs]
    assert len(names) == len(set(names))


def test_baseline_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c0_baseline")
    assert c.llm_kwargs["enforce_eager"] is True
    assert c.llm_kwargs["enable_chunked_prefill"] is False
    assert c.llm_kwargs["enable_prefix_caching"] is False


def test_chunked_prefill_configs(all_configs: list[ExperimentConfig]) -> None:
    c1 = _find(all_configs, "c1_chunked_prefill_small")
    c2 = _find(all_configs, "c2_chunked_prefill_large")
    assert c1.llm_kwargs["max_num_batched_tokens"] == 512
    assert c2.llm_kwargs["max_num_batched_tokens"] == 2048


def test_cuda_graph_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c3_cuda_graphs")
    assert c.llm_kwargs["enforce_eager"] is False


def test_prefix_caching_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c4_prefix_caching")
    assert c.llm_kwargs["enable_prefix_caching"] is True


def test_batch_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c5_batch_processing")
    assert c.batch_mode is True


def test_low_memory_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c6_low_memory")
    assert c.llm_kwargs["gpu_memory_utilization"] == 0.4


def test_fp8_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c7_fp8_kv_cache")
    assert c.llm_kwargs["kv_cache_dtype"] == "fp8_e5m2"
    assert c.optional is True


def test_cpu_swap_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c8_cpu_swap")
    assert c.llm_kwargs["swap_space"] == 4


def test_determinism_config(all_configs: list[ExperimentConfig]) -> None:
    c = _find(all_configs, "c9_determinism")
    assert c.num_runs == 3


def test_config_has_name_and_description(all_configs: list[ExperimentConfig]) -> None:
    for c in all_configs:
        assert c.name, f"Config {c.display_name!r} has empty name"
        assert c.description, f"Config {c.name!r} has empty description"


def test_config_to_llm_kwargs(all_configs: list[ExperimentConfig]) -> None:
    for c in all_configs:
        assert "model" in c.llm_kwargs, (
            f"Config {c.name!r} missing 'model' in llm_kwargs"
        )
        assert "seed" in c.llm_kwargs, f"Config {c.name!r} missing 'seed' in llm_kwargs"


def test_get_config_by_name_found() -> None:
    config = get_config_by_name("c0_baseline")
    assert config is not None
    assert config.name == "c0_baseline"


def test_get_config_by_name_not_found() -> None:
    config = get_config_by_name("nonexistent_config")
    assert config is None


def test_model_parameter_override() -> None:
    custom_model = "custom/model-name"
    configs = get_all_configs(model=custom_model)
    for c in configs:
        assert c.llm_kwargs["model"] == custom_model, (
            f"Config {c.name!r} has model {c.llm_kwargs['model']!r}, expected {custom_model!r}"
        )
