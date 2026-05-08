from __future__ import annotations

from dataclasses import dataclass, field

from configs.experiment_config import ExperimentConfig
from src.prompts import Prompt
from src.utils import cleanup_gpu, setup_logging

logger = setup_logging()


@dataclass
class InferenceOutput:
    prompt_id: str
    prompt_text: str
    output_text: str
    token_ids: list[int]
    num_generated_tokens: int
    config_name: str
    run_index: int


@dataclass
class ConfigResult:
    config_name: str
    display_name: str
    outputs: list[InferenceOutput] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _build_output(
    prompt: Prompt,
    completion,
    config_name: str,
    run_index: int,
) -> InferenceOutput:
    text = completion.outputs[0].text
    token_ids = list(completion.outputs[0].token_ids)
    return InferenceOutput(
        prompt_id=prompt.id,
        prompt_text=prompt.text,
        output_text=text,
        token_ids=token_ids,
        num_generated_tokens=len(token_ids),
        config_name=config_name,
        run_index=run_index,
    )


def run_config(
    config: ExperimentConfig,
    prompts: list[Prompt],
    max_tokens: int = 256,
) -> ConfigResult:
    try:
        import vllm
    except ImportError:
        raise ImportError("vLLM is not installed. Install it with: pip install vllm")

    result = ConfigResult(
        config_name=config.name,
        display_name=config.display_name,
    )
    logger.info("Running config %s...", config.display_name)

    try:
        sampling_params = vllm.SamplingParams(
            temperature=0, max_tokens=max_tokens, seed=42
        )

        for run_idx in range(config.num_runs):
            llm = vllm.LLM(**config.llm_kwargs)
            try:
                if config.batch_mode:
                    completions = llm.generate(
                        [p.text for p in prompts], sampling_params
                    )
                    for prompt, comp in zip(prompts, completions):
                        result.outputs.append(
                            _build_output(prompt, comp, config.name, run_idx)
                        )
                else:
                    for prompt in prompts:
                        completions = llm.generate([prompt.text], sampling_params)
                        result.outputs.append(
                            _build_output(prompt, completions[0], config.name, run_idx)
                        )
            finally:
                del llm
                cleanup_gpu()

    except Exception as exc:
        result.error = str(exc)
        logger.error("Error in config %s: %s", config.display_name, exc)
        return result

    logger.info(
        "Config %s complete. %d outputs.",
        config.display_name,
        len(result.outputs),
    )
    return result


def run_all_configs(
    configs: list[ExperimentConfig],
    prompts: list[Prompt],
    max_tokens: int = 256,
    skip_unsupported: bool = True,
) -> dict[str, ConfigResult]:
    results: dict[str, ConfigResult] = {}

    for config in configs:
        result = run_config(config, prompts, max_tokens)

        if result.error is not None and config.optional and skip_unsupported:
            result.skipped = True
            result.skip_reason = result.error
            result.error = None
            logger.info(
                "Skipping optional config %s (unsupported): %s",
                config.display_name,
                result.skip_reason,
            )

        results[config.name] = result

    return results
