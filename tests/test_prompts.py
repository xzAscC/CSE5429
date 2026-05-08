from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

# src/__init__.py imports modules that don't exist yet, so we load
# src.prompts directly from its file path to avoid triggering the package init.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_spec = importlib.util.spec_from_file_location(
    "src.prompts",
    _project_root + "/src/prompts.py",
)
_prompts_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.prompts"] = _prompts_mod
_spec.loader.exec_module(_prompts_mod)

Prompt = _prompts_mod.Prompt
get_all_categories = _prompts_mod.get_all_categories
get_all_prompts = _prompts_mod.get_all_prompts
get_prompt_by_id = _prompts_mod.get_prompt_by_id
get_prompts_by_category = _prompts_mod.get_prompts_by_category


def test_prompts_loaded():
    prompts = get_all_prompts()
    assert isinstance(prompts, list)
    assert len(prompts) > 0


def test_prompt_count():
    assert len(get_all_prompts()) == 20


def test_prompt_categories():
    categories = get_all_categories()
    expected = {
        "short_factual",
        "medium_creative",
        "long_reasoning",
        "code_generation",
        "edge_case",
        "repetition_sensitive",
    }
    assert set(categories) == expected
    assert len(categories) == 6


def test_prompt_structure():
    field_names = {f.name for f in fields(Prompt)}
    expected_fields = {"text", "category", "id", "expected_min_tokens"}
    assert field_names == expected_fields
    for p in get_all_prompts():
        assert isinstance(p.text, str)
        assert isinstance(p.category, str)
        assert isinstance(p.id, str)
        assert isinstance(p.expected_min_tokens, int)


def test_prompt_subset_deterministic():
    for cat in get_all_categories():
        first = get_prompts_by_category(cat)
        second = get_prompts_by_category(cat)
        assert first == second


def test_prompt_texts_are_strings():
    for p in get_all_prompts():
        assert isinstance(p.text, str)
        assert len(p.text) > 0


def test_prompt_ids_unique():
    ids = [p.id for p in get_all_prompts()]
    assert len(ids) == len(set(ids))


def test_get_prompt_by_id_found():
    prompt = get_prompt_by_id("sf_01")
    assert prompt is not None
    assert prompt.text == "What is the capital of Japan?"
    assert prompt.category == "short_factual"


def test_get_prompt_by_id_not_found():
    assert get_prompt_by_id("nonexistent_id") is None


def test_category_sizes():
    expected_sizes = {
        "short_factual": 4,
        "medium_creative": 4,
        "long_reasoning": 4,
        "code_generation": 3,
        "edge_case": 3,
        "repetition_sensitive": 2,
    }
    for category, expected_count in expected_sizes.items():
        prompts = get_prompts_by_category(category)
        assert len(prompts) == expected_count, (
            f"Category '{category}' has {len(prompts)} prompts, expected {expected_count}"
        )
