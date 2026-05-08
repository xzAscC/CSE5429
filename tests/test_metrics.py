"""Tests for evaluation.metrics module."""

import sys
import types
from dataclasses import fields

import pytest

# evaluation/__init__.py imports modules not yet created (comparison, report).
# Stub them so the package can be loaded during development.
for _mod_name in ("evaluation.comparison", "evaluation.report"):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if "comparison" in _mod_name:
            _stub.compare_pair = None
            _stub.compare_all_configs = None
        if "report" in _mod_name:
            _stub.generate_report = None
        sys.modules[_mod_name] = _stub

from evaluation.metrics import (
    ComparisonResult,
    bleu_score,
    compare_texts,
    edit_distance,
    edit_distance_normalized,
    exact_match,
    first_divergence_token,
    rouge_scores,
    semantic_similarity,
    token_match_ratio,
)


# -- exact_match --


class TestExactMatch:
    def test_exact_match_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_exact_match_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_exact_match_case_sensitive(self):
        assert exact_match("Hello", "hello") == 0.0

    def test_exact_match_empty(self):
        assert exact_match("", "") == 1.0


# -- token_match_ratio --


class TestTokenMatchRatio:
    def test_token_match_ratio_identical(self):
        assert token_match_ratio("the cat sat", "the cat sat") == 1.0

    def test_token_match_ratio_partial(self):
        result = token_match_ratio("the cat sat", "the dog sat")
        assert abs(result - 2 / 3) < 0.01

    def test_token_match_ratio_different_length(self):
        assert token_match_ratio("a b c", "a b c d") == 1.0

    def test_token_match_ratio_no_match(self):
        assert token_match_ratio("a b c", "x y z") == 0.0

    def test_token_match_ratio_empty_both(self):
        assert token_match_ratio("", "") == 1.0

    def test_token_match_ratio_one_empty(self):
        assert token_match_ratio("hello", "") == 0.0


# -- bleu_score --


class TestBleuScore:
    def test_bleu_identical(self):
        assert bleu_score("the cat sat on the mat", "the cat sat on the mat") == 100.0

    def test_bleu_partial(self):
        result = bleu_score("the cat sat on the mat", "the dog sat on the rug")
        assert 0.0 < result < 100.0

    def test_bleu_no_overlap(self):
        result = bleu_score("alpha beta gamma", "one two three")
        assert result < 5.0

    def test_bleu_empty_reference(self):
        assert bleu_score("", "some text") == 0.0

    def test_bleu_empty_hypothesis(self):
        assert bleu_score("some text", "") == 0.0


# -- rouge_scores --


class TestRougeScores:
    def test_rouge_identical(self):
        scores = rouge_scores("the cat sat on the mat", "the cat sat on the mat")
        assert scores["rouge1"]["fmeasure"] == 1.0
        assert scores["rouge2"]["fmeasure"] == 1.0
        assert scores["rougeL"]["fmeasure"] == 1.0

    def test_rouge_partial(self):
        scores = rouge_scores("the cat sat on the mat", "the dog sat on the rug")
        for key in ("rouge1", "rouge2", "rougeL"):
            assert 0.0 < scores[key]["fmeasure"] < 1.0

    def test_rouge_has_all_keys(self):
        scores = rouge_scores("a b c", "x y z")
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}
        for key in scores:
            assert set(scores[key].keys()) == {"precision", "recall", "fmeasure"}

    def test_rouge_empty_strings(self):
        scores = rouge_scores("", "")
        assert scores["rouge1"]["fmeasure"] == 0.0


# -- edit_distance --


class TestEditDistance:
    def test_edit_distance_identical(self):
        assert edit_distance("hello", "hello") == 0

    def test_edit_distance_known(self):
        assert edit_distance("kitten", "sitting") == 3

    def test_edit_distance_empty(self):
        assert edit_distance("", "abc") == 3

    def test_edit_distance_both_empty(self):
        assert edit_distance("", "") == 0

    def test_edit_distance_single_char(self):
        assert edit_distance("a", "b") == 1


# -- edit_distance_normalized --


class TestEditDistanceNormalized:
    def test_edit_distance_normalized_identical(self):
        assert edit_distance_normalized("hello", "hello") == 0.0

    def test_edit_distance_normalized_completely_different(self):
        assert edit_distance_normalized("abc", "xyz") == 1.0

    def test_edit_distance_normalized_both_empty(self):
        assert edit_distance_normalized("", "") == 0.0

    def test_edit_distance_normalized_partial(self):
        result = edit_distance_normalized("abcde", "abXde")
        assert 0.0 < result < 1.0


# -- first_divergence_token --


class TestFirstDivergenceToken:
    def test_first_divergence_identical(self):
        assert first_divergence_token("a b c", "a b c") == -1

    def test_first_divergence_first_token(self):
        assert first_divergence_token("a b c", "x b c") == 0

    def test_first_divergence_middle(self):
        assert first_divergence_token("a b c", "a x c") == 1

    def test_first_divergence_different_length(self):
        # "a b c" vs "a b" — all shared-position tokens match, but lengths differ
        assert first_divergence_token("a b c", "a b") == 2

    def test_first_divergence_empty_both(self):
        assert first_divergence_token("", "") == -1

    def test_first_divergence_one_empty(self):
        assert first_divergence_token("a b c", "") == 0


# -- semantic_similarity --


class TestSemanticSimilarity:
    def test_semantic_identical(self):
        result = semantic_similarity(
            "The cat sat on the mat.", "The cat sat on the mat."
        )
        assert result > 0.99

    def test_semantic_similar_meaning(self):
        result = semantic_similarity(
            "The cat sat on the mat.", "A kitten was sitting on a rug."
        )
        assert result > 0.5

    def test_semantic_dissimilar(self):
        result = semantic_similarity(
            "The cat sat on the mat.",
            "Quantum physics explains subatomic particles.",
        )
        assert result < 0.5


# -- compare_texts --


class TestCompareTexts:
    def test_compare_texts_returns_comparison_result(self):
        result = compare_texts("hello world", "hello world", include_semantic=False)
        assert isinstance(result, ComparisonResult)
        expected_fields = {f.name for f in fields(ComparisonResult)}
        actual_fields = set(result.__dict__.keys())
        assert actual_fields == expected_fields

    def test_compare_texts_identical(self):
        text = "the cat sat on the mat"
        result = compare_texts(text, text, include_semantic=False)
        assert result.exact_match == 1.0
        assert result.token_match_ratio == 1.0
        assert result.bleu == 100.0
        assert result.rouge_1_f == 1.0
        assert result.rouge_2_f == 1.0
        assert result.rouge_l_f == 1.0
        assert result.edit_distance == 0
        assert result.edit_distance_normalized == 0.0
        assert result.first_divergence_token == -1
        assert result.num_tokens_a == result.num_tokens_b

    def test_compare_texts_without_semantic(self):
        result = compare_texts("hello", "world", include_semantic=False)
        assert result.semantic_similarity == -1.0

    def test_compare_texts_with_semantic(self):
        result = compare_texts("hello world", "hello world", include_semantic=True)
        assert result.semantic_similarity > 0.99

    def test_compare_texts_different(self):
        result = compare_texts("the cat sat", "a dog ran", include_semantic=False)
        assert result.exact_match == 0.0
        assert result.edit_distance > 0
        assert result.first_divergence_token == 0
