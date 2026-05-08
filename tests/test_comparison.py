"""Tests for evaluation.comparison module."""

import sys
import types
from dataclasses import fields as dc_fields

import pytest

# evaluation/__init__.py imports modules not yet created (report).
# Stub them so the package can be loaded during development.
for _mod_name in ("evaluation.report",):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.generate_report = None  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

from evaluation.comparison import (
    ConfigPairComparison,
    compare_all_configs,
    compare_pair,
    comparison_matrix_to_dict,
)
from evaluation.metrics import ComparisonResult


# -- compare_pair --


class TestComparePairIdenticalOutputs:
    def test_compare_pair_identical_outputs(self):
        outputs = [
            {"prompt_id": "p1", "output_text": "the cat sat on the mat"},
            {"prompt_id": "p2", "output_text": "hello world"},
        ]
        result = compare_pair(outputs, outputs, include_semantic=False)
        assert result.mean_exact_match == 1.0
        assert result.mean_bleu == 100.0
        assert result.mean_edit_distance == 0.0
        assert result.num_prompts_compared == 2
        assert result.num_exact_matches == 2


class TestComparePairDifferentOutputs:
    def test_compare_pair_different_outputs(self):
        outputs_a = [
            {"prompt_id": "p1", "output_text": "the cat sat on the mat"},
        ]
        outputs_b = [
            {"prompt_id": "p1", "output_text": "a dog ran in the park"},
        ]
        result = compare_pair(outputs_a, outputs_b, include_semantic=False)
        assert result.mean_exact_match == 0.0
        assert result.mean_bleu < 100.0
        assert result.mean_edit_distance > 0
        assert result.num_exact_matches == 0


class TestComparePairMissingPrompt:
    def test_compare_pair_missing_prompt(self):
        outputs_a = [
            {"prompt_id": "p1", "output_text": "hello"},
            {"prompt_id": "p2", "output_text": "world"},
        ]
        outputs_b = [
            {"prompt_id": "p2", "output_text": "world"},
            {"prompt_id": "p3", "output_text": "foo"},
        ]
        result = compare_pair(outputs_a, outputs_b, include_semantic=False)
        # Only p2 is common
        assert result.num_prompts_compared == 1
        assert result.mean_exact_match == 1.0


class TestComparePairEmptyLists:
    def test_compare_pair_empty_both(self):
        with pytest.raises(ValueError):
            compare_pair([], [], include_semantic=False)

    def test_compare_pair_one_empty(self):
        outputs = [{"prompt_id": "p1", "output_text": "hello"}]
        with pytest.raises(ValueError):
            compare_pair(outputs, [], include_semantic=False)  # type: ignore[arg-type]

    def test_compare_pair_no_overlapping_ids(self):
        outputs_a = [{"prompt_id": "p1", "output_text": "hello"}]
        outputs_b = [{"prompt_id": "p2", "output_text": "hello"}]
        with pytest.raises(ValueError):
            compare_pair(outputs_a, outputs_b, include_semantic=False)


# -- compare_all_configs --


class TestCompareAllConfigsThreeConfigs:
    def test_compare_all_configs_three_configs(self):
        all_outputs = {
            "config_a": [
                {"prompt_id": "p1", "output_text": "hello world"},
            ],
            "config_b": [
                {"prompt_id": "p1", "output_text": "hello world"},
            ],
            "config_c": [
                {"prompt_id": "p1", "output_text": "foo bar"},
            ],
        }
        result = compare_all_configs(all_outputs, include_semantic=False)
        # 3 self-pairs + 3 cross-pairs = 6
        assert len(result) == 6
        assert ("config_a", "config_a") in result
        assert ("config_a", "config_b") in result
        assert ("config_a", "config_c") in result
        assert ("config_b", "config_b") in result
        assert ("config_b", "config_c") in result
        assert ("config_c", "config_c") in result

    def test_self_pair_perfect_match(self):
        all_outputs = {
            "cfg": [{"prompt_id": "p1", "output_text": "hello"}],
        }
        result = compare_all_configs(all_outputs, include_semantic=False)
        comp = result[("cfg", "cfg")]
        assert comp.mean_exact_match == 1.0


class TestCompareAllConfigsSymmetricMetrics:
    def test_symmetric_metrics_across_direction(self):
        outputs_a = [{"prompt_id": "p1", "output_text": "the cat sat on the mat"}]
        outputs_b = [{"prompt_id": "p1", "output_text": "a dog ran in the park"}]

        result_ab = compare_pair(outputs_a, outputs_b, include_semantic=False)
        result_ba = compare_pair(outputs_b, outputs_a, include_semantic=False)

        # edit_distance is symmetric
        assert result_ab.mean_edit_distance == result_ba.mean_edit_distance
        assert (
            result_ab.mean_edit_distance_normalized
            == result_ba.mean_edit_distance_normalized
        )
        # exact_match is symmetric
        assert result_ab.mean_exact_match == result_ba.mean_exact_match


# -- comparison_matrix_to_dict --


class TestComparisonMatrixToDict:
    def test_comparison_matrix_to_dict(self):
        all_outputs = {
            "cfg_a": [{"prompt_id": "p1", "output_text": "hello"}],
            "cfg_b": [{"prompt_id": "p1", "output_text": "hello"}],
        }
        comparisons = compare_all_configs(all_outputs, include_semantic=False)
        matrix = comparison_matrix_to_dict(comparisons, metric="exact_match")

        assert isinstance(matrix, dict)
        assert "cfg_a" in matrix
        assert "cfg_b" in matrix["cfg_a"]
        # Self-pair should be perfect match
        assert matrix["cfg_a"]["cfg_a"] == 1.0
        assert matrix["cfg_b"]["cfg_b"] == 1.0

    def test_comparison_matrix_to_dict_all_metrics(self):
        all_outputs = {
            "cfg_a": [{"prompt_id": "p1", "output_text": "hello world"}],
            "cfg_b": [{"prompt_id": "p1", "output_text": "hello world"}],
        }
        comparisons = compare_all_configs(all_outputs, include_semantic=False)

        all_metrics = [
            "exact_match",
            "token_match_ratio",
            "bleu",
            "mean_rouge_l_f",
            "edit_distance",
            "edit_distance_normalized",
            "mean_semantic_similarity",
            "mean_first_divergence_token",
        ]
        for metric in all_metrics:
            matrix = comparison_matrix_to_dict(comparisons, metric=metric)
            assert isinstance(matrix, dict)
            assert len(matrix) > 0
            for inner_dict in matrix.values():
                for value in inner_dict.values():
                    assert isinstance(value, float)

    def test_comparison_matrix_to_dict_unknown_metric(self):
        comparisons = {
            ("a", "a"): ConfigPairComparison(
                config_a="a",
                config_b="a",
                prompt_comparisons=[],
                mean_exact_match=1.0,
                mean_token_match_ratio=1.0,
                mean_bleu=100.0,
                mean_rouge_l_f=1.0,
                mean_edit_distance=0.0,
                mean_edit_distance_normalized=0.0,
                mean_semantic_similarity=-1.0,
                mean_first_divergence_token=-1.0,
                num_prompts_compared=1,
                num_exact_matches=1,
            ),
        }
        with pytest.raises(ValueError, match="Unknown metric"):
            comparison_matrix_to_dict(comparisons, metric="nonexistent_metric")


# -- dataclass field tests --


class TestPromptComparisonFields:
    def test_prompt_comparison_fields(self):
        outputs_a = [{"prompt_id": "p1", "output_text": "hello"}]
        outputs_b = [{"prompt_id": "p1", "output_text": "world"}]
        result = compare_pair(outputs_a, outputs_b, include_semantic=False)
        assert len(result.prompt_comparisons) == 1

        pc = result.prompt_comparisons[0]
        assert pc.prompt_id == "p1"
        assert isinstance(pc.result, ComparisonResult)
        assert pc.result.exact_match == 0.0
        assert pc.result.num_tokens_a == 1
        assert pc.result.num_tokens_b == 1


class TestConfigPairComparisonFields:
    def test_config_pair_comparison_fields(self):
        expected_fields = {
            "config_a",
            "config_b",
            "prompt_comparisons",
            "mean_exact_match",
            "mean_token_match_ratio",
            "mean_bleu",
            "mean_rouge_l_f",
            "mean_edit_distance",
            "mean_edit_distance_normalized",
            "mean_semantic_similarity",
            "mean_first_divergence_token",
            "num_prompts_compared",
            "num_exact_matches",
        }
        actual_fields = {f.name for f in dc_fields(ConfigPairComparison)}
        assert actual_fields == expected_fields


class TestNumExactMatchesCount:
    def test_num_exact_matches_mixed(self):
        outputs_a = [
            {"prompt_id": "p1", "output_text": "hello world"},
            {"prompt_id": "p2", "output_text": "foo bar"},
            {"prompt_id": "p3", "output_text": "baz qux"},
        ]
        outputs_b = [
            {"prompt_id": "p1", "output_text": "hello world"},
            {"prompt_id": "p2", "output_text": "different text"},
            {"prompt_id": "p3", "output_text": "baz qux"},
        ]
        result = compare_pair(outputs_a, outputs_b, include_semantic=False)
        assert result.num_exact_matches == 2
        assert result.num_prompts_compared == 3

    def test_num_exact_matches_none(self):
        outputs_a = [{"prompt_id": "p1", "output_text": "hello"}]
        outputs_b = [{"prompt_id": "p1", "output_text": "world"}]
        result = compare_pair(outputs_a, outputs_b, include_semantic=False)
        assert result.num_exact_matches == 0
