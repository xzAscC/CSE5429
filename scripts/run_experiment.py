#!/usr/bin/env python3
"""Main experiment orchestrator for LLM inference configuration comparison."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

# Allow running from any directory by adding project root to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import get_all_configs, get_config_by_name
from evaluation.comparison import compare_all_configs
from evaluation.report import generate_all_reports
from src.prompts import get_all_prompts
from src.runner import ConfigResult, run_all_configs
from src.utils import (
    ensure_dir,
    get_results_path,
    load_results,
    save_results,
    setup_logging,
)

logger = logging.getLogger("vllm_experiment")


def _serialize_config_result(result: ConfigResult) -> dict:
    """Convert a ConfigResult to a JSON-serializable dict."""
    return {
        "config_name": result.config_name,
        "display_name": result.display_name,
        "outputs": [
            {
                "prompt_id": o.prompt_id,
                "prompt_text": o.prompt_text,
                "output_text": o.output_text,
                "token_ids": o.token_ids,
                "num_generated_tokens": o.num_generated_tokens,
                "config_name": o.config_name,
                "run_index": o.run_index,
            }
            for o in result.outputs
        ],
        "error": result.error,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
    }


def _extract_comparison_outputs(
    serialized: dict[str, dict],
) -> dict[str, list[dict]]:
    """Build the dict format expected by compare_all_configs.

    Filters out configs that had errors or were skipped.
    """
    comparison_data: dict[str, list[dict]] = {}
    for config_name, data in serialized.items():
        if data.get("error") or data.get("skipped"):
            continue
        comparison_data[config_name] = [
            {"prompt_id": o["prompt_id"], "output_text": o["output_text"]}
            for o in data["outputs"]
        ]
    return comparison_data


def _load_existing_results(output_dir: str) -> dict[str, dict]:
    """Discover and load all result JSON files from output_dir."""
    results: dict[str, dict] = {}
    pattern = os.path.join(output_dir, "*.json")
    for filepath in sorted(glob.glob(pattern)):
        config_name = os.path.splitext(os.path.basename(filepath))[0]
        results[config_name] = load_results(filepath)
    return results


def _build_config_results_for_report(
    serialized: dict[str, dict],
) -> dict:
    """Build a lightweight namespace map compatible with generate_all_reports.

    The report module reads .display_name, .error, .skipped via getattr.
    """
    from types import SimpleNamespace

    out = {}
    for name, data in serialized.items():
        out[name] = SimpleNamespace(
            display_name=data.get("display_name", name),
            error=data.get("error"),
            skipped=data.get("skipped", False),
        )
    return out


def run_report_only(args: argparse.Namespace) -> None:
    """Generate reports from existing result files."""
    logger.info("Loading existing results from %s", args.output_dir)
    serialized = _load_existing_results(args.output_dir)

    if not serialized:
        logger.error("No result files found in %s", args.output_dir)
        return

    logger.info("Found results for %d configs", len(serialized))

    comparison_data = _extract_comparison_outputs(serialized)
    if not comparison_data:
        logger.error("No valid (non-skipped, non-error) configs to compare")
        return

    logger.info("Comparing %d configs...", len(comparison_data))
    comparisons = compare_all_configs(comparison_data, args.include_semantic)

    config_results = _build_config_results_for_report(serialized)
    report_paths = generate_all_reports(comparisons, config_results, args.report_dir)

    print(f"\nReports generated ({len(report_paths)} files) in {args.report_dir}:")
    for name in sorted(report_paths):
        print(f"  - {name}")
    print(f"\nCompared {len(comparison_data)} configurations.")


def run_experiment(args: argparse.Namespace) -> None:
    """Run configs, save results, generate reports."""
    if args.config:
        config = get_config_by_name(args.config)
        if config is None:
            logger.error("Config '%s' not found", args.config)
            sys.exit(1)
        configs = [config]
    else:
        configs = get_all_configs(args.model, args.max_model_len)

    prompts = get_all_prompts()
    logger.info(
        "Running %d config(s) with %d prompts (max_tokens=%d)",
        len(configs),
        len(prompts),
        args.max_tokens,
    )

    results = run_all_configs(
        configs,
        prompts,
        args.max_tokens,
        args.skip_unsupported,
    )

    # Save individual results and collect serialized data.
    ensure_dir(args.output_dir)
    serialized: dict[str, dict] = {}
    for config_name, result in results.items():
        serialized[config_name] = _serialize_config_result(result)
        path = get_results_path(config_name, args.output_dir)
        save_results(serialized[config_name], path)
        logger.info("Saved results to %s", path)

    # Build comparison data and run comparisons.
    comparison_data = _extract_comparison_outputs(serialized)
    comparisons = {}
    if len(comparison_data) >= 2:
        logger.info("Running pairwise comparisons...")
        comparisons = compare_all_configs(comparison_data, args.include_semantic)
    else:
        logger.warning(
            "Need at least 2 valid configs for comparison, got %d",
            len(comparison_data),
        )

    # Generate reports.
    config_results = _build_config_results_for_report(serialized)
    report_paths = generate_all_reports(comparisons, config_results, args.report_dir)

    # Summary.
    num_ok = sum(
        1 for d in serialized.values() if not d.get("error") and not d.get("skipped")
    )
    num_errors = sum(1 for d in serialized.values() if d.get("error"))
    num_skipped = sum(1 for d in serialized.values() if d.get("skipped"))

    print("\n=== Experiment Summary ===")
    print(
        f"Configs run:  {num_ok} succeeded, {num_errors} errors, {num_skipped} skipped"
    )
    print(f"Results:      {args.output_dir}/")
    if report_paths:
        print(f"Reports:      {args.report_dir}/ ({len(report_paths)} files)")
    else:
        print("Reports:      not generated (insufficient valid configs)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM inference configuration comparison experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run a single config by name (e.g., 'c0_baseline')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all configurations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Model to use (default: Qwen/Qwen2-1.5B-Instruct)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model context length (default: 2048)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw",
        help="Directory for raw output JSON files (default: results/raw)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/reports",
        help="Directory for generated reports (default: results/reports)",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        default=True,
        help="Skip optional configs that fail (default: True)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results without running inference",
    )
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        default=False,
        help="Include semantic similarity in comparisons (slow)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.report_only:
        run_report_only(args)
    elif args.config or args.run_all:
        run_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
