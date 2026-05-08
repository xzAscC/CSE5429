from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("vllm_experiment")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def save_results(data: dict[str, Any], filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results(filepath: str) -> dict[str, Any]:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def cleanup_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_results_path(config_name: str, output_dir: str = "results/raw") -> str:
    return os.path.join(output_dir, f"{config_name}.json")
