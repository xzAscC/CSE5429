"""Tests for evaluation.report module."""

import csv
import os
import sys
import types
from types import SimpleNamespace

import pytest

# Stub evaluation.report so evaluation/__init__.py can load. Then remove
# the stub so the real module can be imported.
if "evaluation.report" not in sys.modules:
    _stub = types.ModuleType("evaluation.report")
    _stub.generate_report = None  # type: ignore[attr-defined]
    sys.modules["evaluation.report"] = _stub

from evaluation.comparison import ConfigPairComparison, PromptComparison  # noqa: E402
from evaluation.metrics import ComparisonResult  # noqa: E402

# Clear any stub so the real evaluation.report module loads.
if "evaluation.report" in sys.modules:
    existing = sys.modules["evaluation.report"]
    if not hasattr(existing, "generate_csv"):
        del sys.modules["evaluation.report"]

from evaluation.report import (  # noqa: E402
    generate_all_reports,
    generate_csv,
    generate_heatmap,
    generate_markdown_report,
    generate_per_prompt_csv,
)


# -- Fixtures / Helpers --


def _make_cr_identical() -> ComparisonResult:
    return ComparisonResult(
        text_a="hello world",
        text_b="hello world",
        exact_match=1.0,
        token_match_ratio=1.0,
        bleu=100.0,
        rouge_1_f=1.0,
        rouge_2_f=1.0,
        rouge_l_f=1.0,
        edit_distance=0,
        edit_distance_normalized=0.0,
        semantic_similarity=-1.0,
        first_divergence_token=-1,
        num_tokens_a=2,
        num_tokens_b=2,
    )


def _make_cr_different() -> ComparisonResult:
    return ComparisonResult(
        text_a="hello world",
        text_b="hi there",
        exact_match=0.0,
        token_match_ratio=0.0,
        bleu=0.0,
        rouge_1_f=0.0,
        rouge_2_f=0.0,
        rouge_l_f=0.0,
        edit_distance=9,
        edit_distance_normalized=0.8182,
        semantic_similarity=-1.0,
        first_divergence_token=0,
        num_tokens_a=2,
        num_tokens_b=2,
    )


def _make_mock_comparisons() -> dict[tuple[str, str], ConfigPairComparison]:
    cr_id = _make_cr_identical()
    cr_diff = _make_cr_different()

    pc_id = PromptComparison(
        prompt_id="p1",
        config_a="c0_baseline",
        config_b="c0_baseline",
        result=cr_id,
    )
    pc_diff = PromptComparison(
        prompt_id="p1",
        config_a="c0_baseline",
        config_b="c1_chunked",
        result=cr_diff,
    )

    return {
        ("c0_baseline", "c0_baseline"): ConfigPairComparison(
            config_a="c0_baseline",
            config_b="c0_baseline",
            prompt_comparisons=[pc_id],
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
        ("c0_baseline", "c1_chunked"): ConfigPairComparison(
            config_a="c0_baseline",
            config_b="c1_chunked",
            prompt_comparisons=[pc_diff],
            mean_exact_match=0.0,
            mean_token_match_ratio=0.0,
            mean_bleu=0.0,
            mean_rouge_l_f=0.0,
            mean_edit_distance=9.0,
            mean_edit_distance_normalized=0.8182,
            mean_semantic_similarity=-1.0,
            mean_first_divergence_token=0.0,
            num_prompts_compared=1,
            num_exact_matches=0,
        ),
        ("c1_chunked", "c1_chunked"): ConfigPairComparison(
            config_a="c1_chunked",
            config_b="c1_chunked",
            prompt_comparisons=[
                PromptComparison(
                    prompt_id="p1",
                    config_a="c1_chunked",
                    config_b="c1_chunked",
                    result=cr_id,
                )
            ],
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


def _make_mock_config_results() -> dict[str, SimpleNamespace]:
    return {
        "c0_baseline": SimpleNamespace(
            config_name="c0_baseline",
            display_name="C0: Baseline",
            error=None,
            skipped=False,
            skip_reason=None,
        ),
        "c1_chunked": SimpleNamespace(
            config_name="c1_chunked",
            display_name="C1: Chunked",
            error=None,
            skipped=False,
            skip_reason=None,
        ),
    }


# -- generate_csv --


class TestGenerateCsv:
    def test_generate_csv(self, tmp_path):
        comparisons = _make_mock_comparisons()
        out = str(tmp_path / "metrics.csv")
        generate_csv(comparisons, out)
        assert os.path.isfile(out)

    def test_generate_csv_content(self, tmp_path):
        comparisons = _make_mock_comparisons()
        out = str(tmp_path / "metrics.csv")
        generate_csv(comparisons, out)

        with open(out) as fh:
            reader = csv.reader(fh)
            rows = list(reader)

        # Header + 3 data rows
        assert len(rows) == 4
        assert rows[0][0] == "config_a"

        # Find the cross-config row (c0_baseline, c1_chunked)
        cross_row = [
            r for r in rows[1:] if r[0] == "c0_baseline" and r[1] == "c1_chunked"
        ][0]
        assert cross_row[2] == "1"  # num_prompts
        assert cross_row[3] == "0"  # exact_matches
        assert float(cross_row[4]) == pytest.approx(0.0)  # exact_match_ratio
        assert float(cross_row[5]) == pytest.approx(0.0)  # token_match_ratio
        assert float(cross_row[6]) == pytest.approx(0.0)  # bleu
        assert float(cross_row[8]) == pytest.approx(9.0)  # edit_distance

        # Self-pair row (c0_baseline, c0_baseline)
        self_row = [
            r for r in rows[1:] if r[0] == "c0_baseline" and r[1] == "c0_baseline"
        ][0]
        assert self_row[3] == "1"  # exact_matches
        assert float(self_row[6]) == pytest.approx(100.0)  # bleu


# -- generate_heatmap --


class TestGenerateHeatmap:
    def test_generate_heatmap(self, tmp_path):
        comparisons = _make_mock_comparisons()
        out = str(tmp_path / "heatmap.png")
        generate_heatmap(comparisons, "exact_match", out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_generate_heatmap_multiple_metrics(self, tmp_path):
        comparisons = _make_mock_comparisons()
        for metric in ("exact_match", "bleu", "edit_distance"):
            out = str(tmp_path / f"heatmap_{metric}.png")
            generate_heatmap(comparisons, metric, out)
            assert os.path.isfile(out), f"Missing heatmap for {metric}"
            assert os.path.getsize(out) > 0


# -- generate_per_prompt_csv --


class TestGeneratePerPromptCsv:
    def test_generate_per_prompt_csv(self, tmp_path):
        comparisons = _make_mock_comparisons()
        out = str(tmp_path / "per_prompt.csv")
        generate_per_prompt_csv(comparisons, out)
        assert os.path.isfile(out)

        with open(out) as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        assert rows[0][0] == "config_a"
        assert "prompt_id" in rows[0]

    def test_generate_per_prompt_csv_rows(self, tmp_path):
        comparisons = _make_mock_comparisons()
        out = str(tmp_path / "per_prompt.csv")
        generate_per_prompt_csv(comparisons, out)

        with open(out) as fh:
            reader = csv.reader(fh)
            rows = list(reader)

        # 3 config pairs, each with 1 prompt comparison = 3 data rows
        assert len(rows) == 4  # 1 header + 3 data


# -- generate_markdown_report --


class TestGenerateMarkdownReport:
    def test_generate_markdown_report(self, tmp_path):
        comparisons = _make_mock_comparisons()
        config_results = _make_mock_config_results()
        out = str(tmp_path / "report.md")
        generate_markdown_report(comparisons, config_results, out)
        assert os.path.isfile(out)
        content = open(out).read()
        assert len(content) > 0

    def test_generate_markdown_report_sections(self, tmp_path):
        comparisons = _make_mock_comparisons()
        config_results = _make_mock_config_results()
        out = str(tmp_path / "report.md")
        content = generate_markdown_report(comparisons, config_results, out)

        assert "## Executive Summary" in content
        assert "## Configuration Summary" in content
        assert "## Determinism Verification" in content
        assert "## Cross-Configuration Comparison" in content
        assert "## Key Findings" in content
        assert "## Detailed Metrics" in content

    def test_generate_markdown_report_returns_content(self, tmp_path):
        comparisons = _make_mock_comparisons()
        config_results = _make_mock_config_results()
        out = str(tmp_path / "report.md")
        returned = generate_markdown_report(comparisons, config_results, out)

        with open(out) as fh:
            file_content = fh.read()
        assert returned == file_content


# -- generate_all_reports --


class TestGenerateAllReports:
    def test_generate_all_reports(self, tmp_path):
        comparisons = _make_mock_comparisons()
        config_results = _make_mock_config_results()
        output_dir = str(tmp_path / "reports")
        generate_all_reports(comparisons, config_results, output_dir)

        expected_files = [
            "metrics_summary.csv",
            "per_prompt_comparison.csv",
            "heatmap_exact_match.png",
            "heatmap_bleu.png",
            "heatmap_edit_distance.png",
            "full_report.md",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"Missing {fname}"

    def test_generate_all_reports_returns_paths(self, tmp_path):
        comparisons = _make_mock_comparisons()
        config_results = _make_mock_config_results()
        output_dir = str(tmp_path / "reports")
        paths = generate_all_reports(comparisons, config_results, output_dir)

        assert isinstance(paths, dict)
        expected_keys = [
            "metrics_summary.csv",
            "per_prompt_comparison.csv",
            "heatmap_exact_match.png",
            "heatmap_bleu.png",
            "heatmap_edit_distance.png",
            "full_report.md",
        ]
        for key in expected_keys:
            assert key in paths, f"Missing key {key}"
            assert paths[key].startswith(output_dir)
