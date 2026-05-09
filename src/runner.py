from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

from configs.experiment_config import ExperimentConfig
from src.prompts import Prompt
from src.utils import setup_logging

logger = setup_logging()

_SUBPROCESS_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "_run_subprocess.py",
)


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


def run_config(
    config: ExperimentConfig,
    prompts: list[Prompt],
    max_tokens: int = 256,
) -> ConfigResult:
    logger.info("Running config %s in isolated subprocess...", config.display_name)

    prompts_data = [{"id": p.id, "text": p.text} for p in prompts]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        result_path = tmp.name

    python = sys.executable
    cmd = [
        python,
        _SUBPROCESS_SCRIPT,
        config.name,
        config.display_name,
        json.dumps(config.llm_kwargs),
        json.dumps(prompts_data),
        str(config.batch_mode),
        str(config.num_runs),
        str(max_tokens),
        result_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            timeout=900,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return ConfigResult(
            config_name=config.name,
            display_name=config.display_name,
            error="Config timed out after 900 seconds",
        )

    if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        return ConfigResult(
            config_name=config.name,
            display_name=config.display_name,
            error="No results written (process may have crashed)",
        )

    with open(result_path) as f:
        data = json.load(f)

    result = ConfigResult(
        config_name=data["config_name"],
        display_name=data["display_name"],
        error=data.get("error"),
        skipped=data.get("skipped", False),
        skip_reason=data.get("skip_reason"),
    )

    for o in data.get("outputs", []):
        result.outputs.append(
            InferenceOutput(
                prompt_id=o["prompt_id"],
                prompt_text=o["prompt_text"],
                output_text=o["output_text"],
                token_ids=o["token_ids"],
                num_generated_tokens=o["num_generated_tokens"],
                config_name=o["config_name"],
                run_index=o["run_index"],
            )
        )

    if result.error:
        logger.error("Error in config %s: %s", config.display_name, result.error)
    else:
        logger.info(
            "Config %s complete. %d outputs.", config.display_name, len(result.outputs)
        )

    try:
        os.unlink(result_path)
    except OSError:
        pass

    return result


def run_all_configs(
    configs: list[ExperimentConfig],
    prompts: list[Prompt],
    max_tokens: int = 256,
    skip_unsupported: bool = True,
) -> dict[str, ConfigResult]:
    results: dict[str, ConfigResult] = {}

    for i, config in enumerate(configs):
        logger.info(
            "Starting config %d/%d: %s",
            i + 1,
            len(configs),
            config.display_name,
        )
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
