"""Pairwise comparison of LLM outputs across inference configurations."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.metrics import ComparisonResult, compare_texts


@dataclass
class PromptComparison:
    prompt_id: str
    config_a: str
    config_b: str
    result: ComparisonResult


@dataclass
class ConfigPairComparison:
    config_a: str
    config_b: str
    prompt_comparisons: list[PromptComparison]
    mean_exact_match: float
    mean_token_match_ratio: float
    mean_bleu: float
    mean_rouge_l_f: float
    mean_edit_distance: float
    mean_edit_distance_normalized: float
    mean_semantic_similarity: float
    mean_first_divergence_token: float
    num_prompts_compared: int
    num_exact_matches: int


def compare_pair(
    outputs_a: list[dict],
    outputs_b: list[dict],
    include_semantic: bool = True,
) -> ConfigPairComparison:
    """Compare outputs from two configs by matching on prompt_id.

    Raises ValueError if either list is empty or no prompt_ids overlap.
    """
    if not outputs_a or not outputs_b:
        raise ValueError("Output lists must not be empty")

    a_by_id = {o["prompt_id"]: o for o in outputs_a}
    b_by_id = {o["prompt_id"]: o for o in outputs_b}

    common_ids = sorted(set(a_by_id) & set(b_by_id))

    if not common_ids:
        raise ValueError("No matching prompt_ids found between output lists")

    prompt_comparisons: list[PromptComparison] = []
    for pid in common_ids:
        result = compare_texts(
            a_by_id[pid]["output_text"],
            b_by_id[pid]["output_text"],
            include_semantic,
        )
        prompt_comparisons.append(
            PromptComparison(
                prompt_id=pid,
                config_a="",
                config_b="",
                result=result,
            )
        )

    n = len(prompt_comparisons)
    return ConfigPairComparison(
        config_a="",
        config_b="",
        prompt_comparisons=prompt_comparisons,
        mean_exact_match=sum(pc.result.exact_match for pc in prompt_comparisons) / n,
        mean_token_match_ratio=sum(
            pc.result.token_match_ratio for pc in prompt_comparisons
        )
        / n,
        mean_bleu=sum(pc.result.bleu for pc in prompt_comparisons) / n,
        mean_rouge_l_f=sum(pc.result.rouge_l_f for pc in prompt_comparisons) / n,
        mean_edit_distance=sum(pc.result.edit_distance for pc in prompt_comparisons)
        / n,
        mean_edit_distance_normalized=sum(
            pc.result.edit_distance_normalized for pc in prompt_comparisons
        )
        / n,
        mean_semantic_similarity=sum(
            pc.result.semantic_similarity for pc in prompt_comparisons
        )
        / n,
        mean_first_divergence_token=sum(
            pc.result.first_divergence_token for pc in prompt_comparisons
        )
        / n,
        num_prompts_compared=n,
        num_exact_matches=sum(
            1 for pc in prompt_comparisons if pc.result.exact_match == 1.0
        ),
    )


def compare_all_configs(
    all_outputs: dict[str, list[dict]],
    include_semantic: bool = True,
) -> dict[tuple[str, str], ConfigPairComparison]:
    """Compare every unique pair of configs (including self-pairs).

    Returns dict mapping (config_a_name, config_b_name) -> ConfigPairComparison.
    For N configs, produces N*(N+1)/2 comparisons (N self-pairs + N choose 2 cross-pairs).
    """
    configs = sorted(all_outputs.keys())
    comparisons: dict[tuple[str, str], ConfigPairComparison] = {}

    for i, name_a in enumerate(configs):
        for j, name_b in enumerate(configs):
            if i > j:
                continue
            comp = compare_pair(
                all_outputs[name_a],
                all_outputs[name_b],
                include_semantic,
            )
            comp.config_a = name_a
            comp.config_b = name_b
            for pc in comp.prompt_comparisons:
                pc.config_a = name_a
                pc.config_b = name_b
            comparisons[(name_a, name_b)] = comp

    return comparisons


_METRIC_ATTR_MAP: dict[str, str] = {
    "exact_match": "mean_exact_match",
    "token_match_ratio": "mean_token_match_ratio",
    "bleu": "mean_bleu",
    "mean_rouge_l_f": "mean_rouge_l_f",
    "edit_distance": "mean_edit_distance",
    "edit_distance_normalized": "mean_edit_distance_normalized",
    "mean_semantic_similarity": "mean_semantic_similarity",
    "mean_first_divergence_token": "mean_first_divergence_token",
}


def comparison_matrix_to_dict(
    comparisons: dict[tuple[str, str], ConfigPairComparison],
    metric: str = "exact_match",
) -> dict[str, dict[str, float]]:
    """Extract a single metric from all comparisons into a 2D nested dict.

    Suitable for heatmap rendering or CSV export.
    Returns {config_a: {config_b: metric_value}}.
    """
    if metric not in _METRIC_ATTR_MAP:
        raise ValueError(
            f"Unknown metric: {metric!r}. Must be one of {list(_METRIC_ATTR_MAP)}"
        )

    attr = _METRIC_ATTR_MAP[metric]
    matrix: dict[str, dict[str, float]] = {}

    for (cfg_a, cfg_b), comp in comparisons.items():
        if cfg_a not in matrix:
            matrix[cfg_a] = {}
        matrix[cfg_a][cfg_b] = getattr(comp, attr)

    return matrix
