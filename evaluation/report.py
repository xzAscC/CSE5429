"""Report generation for LLM output comparison analysis."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.comparison import ConfigPairComparison, comparison_matrix_to_dict

logger = logging.getLogger(__name__)

_DISTANCE_METRICS = {
    "edit_distance",
    "edit_distance_normalized",
    "mean_first_divergence_token",
}


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def generate_csv(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    output_path: str,
) -> None:
    """Write a summary CSV with one row per config pair."""
    _ensure_parent(output_path)
    headers = [
        "config_a",
        "config_b",
        "num_prompts",
        "exact_matches",
        "exact_match_ratio",
        "token_match_ratio",
        "bleu",
        "rouge_l",
        "edit_distance",
        "edit_distance_normalized",
        "semantic_similarity",
        "first_divergence_token",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for (ca, cb), comp in sorted(comparisons.items()):
            ratio = (
                comp.num_exact_matches / comp.num_prompts_compared
                if comp.num_prompts_compared
                else 0.0
            )
            writer.writerow(
                [
                    ca,
                    cb,
                    comp.num_prompts_compared,
                    comp.num_exact_matches,
                    ratio,
                    comp.mean_token_match_ratio,
                    comp.mean_bleu,
                    comp.mean_rouge_l_f,
                    comp.mean_edit_distance,
                    comp.mean_edit_distance_normalized,
                    comp.mean_semantic_similarity,
                    comp.mean_first_divergence_token,
                ]
            )
    logger.info("Wrote metrics CSV to %s", output_path)


def generate_heatmap(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    metric: str,
    output_path: str,
) -> None:
    """Render and save a heatmap for a single metric."""
    _ensure_parent(output_path)
    matrix = comparison_matrix_to_dict(comparisons, metric=metric)
    configs = sorted(matrix.keys())

    # Build 2D numpy array; use NaN for missing entries.
    n = len(configs)
    data = np.full((n, n), np.nan)
    for i, ca in enumerate(configs):
        for j, cb in enumerate(configs):
            val = matrix.get(ca, {}).get(cb)
            if val is not None:
                data[i, j] = val
            else:
                # Try reverse direction and mirror.
                rev = matrix.get(cb, {}).get(ca)
                if rev is not None:
                    data[i, j] = rev

    cmap = "RdYlGn_r" if metric in _DISTANCE_METRICS else "RdYlGn"

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel("Config B")
    ax.set_ylabel("Config A")
    ax.set_title(f"Pairwise {metric}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, format="png")
    plt.close(fig)
    logger.info("Wrote heatmap (%s) to %s", metric, output_path)


def generate_per_prompt_csv(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    output_path: str,
) -> None:
    """Write a CSV with one row per (config_pair, prompt_id)."""
    _ensure_parent(output_path)
    headers = [
        "config_a",
        "config_b",
        "prompt_id",
        "exact_match",
        "token_match_ratio",
        "bleu",
        "rouge_l",
        "edit_distance",
        "first_divergence_token",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for (ca, cb), comp in sorted(comparisons.items()):
            for pc in comp.prompt_comparisons:
                r = pc.result
                writer.writerow(
                    [
                        ca,
                        cb,
                        pc.prompt_id,
                        r.exact_match,
                        r.token_match_ratio,
                        r.bleu,
                        r.rouge_l_f,
                        r.edit_distance,
                        r.first_divergence_token,
                    ]
                )
    logger.info("Wrote per-prompt CSV to %s", output_path)


def generate_markdown_report(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    config_results: dict[str, Any],
    output_path: str,
) -> str:
    """Generate a comprehensive markdown report.

    Returns the markdown content as a string.
    """
    _ensure_parent(output_path)
    lines: list[str] = []

    # -- helpers --
    def _status(cr: Any) -> str:
        if getattr(cr, "error", None):
            return f"error ({cr.error})"
        if getattr(cr, "skipped", False):
            return "skipped"
        return "ran"

    def _fmt(v: float, width: int = 4) -> str:
        return f"{v:.{width}f}"

    # Collect unique config names.
    config_names: set[str] = set()
    for ca, cb in comparisons:
        config_names.add(ca)
        config_names.add(cb)

    # Identify baseline (first config alphabetically, typically c0_baseline).
    baseline_name = sorted(config_names)[0] if config_names else None

    # -- Executive Summary --
    lines.append("# LLM Inference Configuration Comparison Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    all_identical = (
        all(
            comp.mean_exact_match == 1.0
            for (ca, _), comp in comparisons.items()
            if ca != _
        )
        if comparisons
        else True
    )

    if all_identical:
        lines.append(
            "All compared inference configurations produce identical outputs. "
            "Optimization parallel inference configurations do **not** change LLM outputs "
            "under the tested conditions."
        )
    else:
        changed = [
            cb
            for (ca, cb), comp in comparisons.items()
            if ca == baseline_name
            and cb != baseline_name
            and comp.mean_exact_match < 1.0
        ]
        unchanged = [
            cb
            for (ca, cb), comp in comparisons.items()
            if ca == baseline_name
            and cb != baseline_name
            and comp.mean_exact_match == 1.0
        ]
        if changed:
            lines.append(
                f"Optimization parallel inference configurations **do** change LLM outputs. "
                f"{len(changed)} configuration(s) produced different outputs compared to "
                f"the baseline: {', '.join(changed)}."
            )
            if unchanged:
                lines.append(
                    f" {len(unchanged)} configuration(s) remained identical: "
                    f"{', '.join(unchanged)}."
                )
        else:
            lines.append(
                "No cross-configuration differences detected in the compared pairs."
            )
    lines.append("")

    # -- Configuration Summary --
    lines.append("## Configuration Summary")
    lines.append("")
    lines.append("| Config | Display Name | Status |")
    lines.append("|--------|-------------|--------|")
    for name in sorted(config_results.keys()):
        cr = config_results[name]
        dn = getattr(cr, "display_name", name)
        lines.append(f"| {name} | {dn} | {_status(cr)} |")
    lines.append("")

    # -- Determinism Verification --
    lines.append("## Determinism Verification")
    lines.append("")
    c9_key = None
    for (ca, cb), comp in comparisons.items():
        if "c9" in ca or "c9" in cb:
            c9_key = (ca, cb)
            break

    if c9_key:
        c9_comp = comparisons[c9_key]
        if c9_comp.mean_exact_match == 1.0:
            lines.append(
                f"C9 (determinism check): Baseline output is perfectly repeatable. "
                f"All {c9_comp.num_prompts_compared} prompt(s) produced exact matches "
                f"across multiple runs."
            )
        else:
            lines.append(
                f"C9 (determinism check): Non-deterministic output detected. "
                f"Exact match ratio: {_fmt(c9_comp.mean_exact_match)}, "
                f"BLEU: {_fmt(c9_comp.mean_bleu)}."
            )
    else:
        lines.append("C9 (determinism check): No determinism check data available.")
    lines.append("")

    # -- Cross-Configuration Comparison --
    lines.append("## Cross-Configuration Comparison")
    lines.append("")
    lines.append("| Config | Exact Match | BLEU | Edit Distance |")
    lines.append("|--------|------------|------|---------------|")
    for (ca, cb), comp in sorted(comparisons.items()):
        if ca == baseline_name and cb != baseline_name:
            lines.append(
                f"| {cb} | {_fmt(comp.mean_exact_match)} "
                f"| {_fmt(comp.mean_bleu)} "
                f"| {_fmt(comp.mean_edit_distance)} |"
            )
    lines.append("")

    # -- Key Findings --
    lines.append("## Key Findings")
    lines.append("")
    if all_identical:
        lines.append(
            "- All optimization configurations produce outputs identical to the baseline."
        )
    else:
        for (ca, cb), comp in sorted(comparisons.items()):
            if ca != cb:
                if comp.mean_exact_match == 1.0:
                    lines.append(
                        f"- **{cb}** vs {ca}: Outputs are identical "
                        f"(exact_match=1.0, BLEU=100.0)."
                    )
                else:
                    lines.append(
                        f"- **{cb}** vs {ca}: Outputs differ "
                        f"(exact_match={_fmt(comp.mean_exact_match)}, "
                        f"BLEU={_fmt(comp.mean_bleu)}, "
                        f"edit_distance={_fmt(comp.mean_edit_distance)})."
                    )
    lines.append("")

    # -- Detailed Metrics --
    lines.append("## Detailed Metrics")
    lines.append("")
    lines.append(
        "| Config A | Config B | Prompts | Exact Match | Token Match "
        "| BLEU | ROUGE-L | Edit Dist | Edit Dist (norm) | First Div |"
    )
    lines.append(
        "|----------|----------|---------|------------|------------"
        "|------|--------|-----------|------------------|-----------|"
    )
    for (ca, cb), comp in sorted(comparisons.items()):
        lines.append(
            f"| {ca} | {cb} | {comp.num_prompts_compared} "
            f"| {_fmt(comp.mean_exact_match)} "
            f"| {_fmt(comp.mean_token_match_ratio)} "
            f"| {_fmt(comp.mean_bleu)} "
            f"| {_fmt(comp.mean_rouge_l_f)} "
            f"| {_fmt(comp.mean_edit_distance)} "
            f"| {_fmt(comp.mean_edit_distance_normalized)} "
            f"| {_fmt(comp.mean_first_divergence_token)} |"
        )
    lines.append("")

    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8")
    logger.info("Wrote markdown report to %s", output_path)
    return content


def generate_all_reports(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    config_results: dict[str, Any],
    output_dir: str = "results/reports",
) -> dict[str, str]:
    """Generate all report artifacts and return path mapping."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    logger.info("Generating reports in %s ...", output_dir)

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    generate_csv(comparisons, csv_path)
    paths["metrics_summary.csv"] = csv_path

    per_prompt_path = os.path.join(output_dir, "per_prompt_comparison.csv")
    generate_per_prompt_csv(comparisons, per_prompt_path)
    paths["per_prompt_comparison.csv"] = per_prompt_path

    for metric in ("exact_match", "bleu", "edit_distance"):
        hm_path = os.path.join(output_dir, f"heatmap_{metric}.png")
        generate_heatmap(comparisons, metric, hm_path)
        paths[f"heatmap_{metric}.png"] = hm_path

    md_path = os.path.join(output_dir, "full_report.md")
    generate_markdown_report(comparisons, config_results, md_path)
    paths["full_report.md"] = md_path

    logger.info("Generated %d report files in %s", len(paths), output_dir)
    return paths


# Alias expected by evaluation/__init__.py
generate_report = generate_all_reports
