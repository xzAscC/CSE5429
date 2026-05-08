"""Evaluation metrics for comparing LLM outputs across inference configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer

        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


@dataclass
class ComparisonResult:
    text_a: str
    text_b: str
    exact_match: float
    token_match_ratio: float
    bleu: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    edit_distance: int
    edit_distance_normalized: float
    semantic_similarity: float
    first_divergence_token: int
    num_tokens_a: int
    num_tokens_b: int


def exact_match(text_a: str, text_b: str) -> float:
    return 1.0 if text_a == text_b else 0.0


def token_match_ratio(text_a: str, text_b: str) -> float:
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    min_len = min(len(tokens_a), len(tokens_b))
    if min_len == 0:
        return 1.0 if len(tokens_a) == len(tokens_b) else 0.0
    matches = sum(1 for i in range(min_len) if tokens_a[i] == tokens_b[i])
    return round(matches / min_len, 4)


def bleu_score(text_a: str, text_b: str) -> float:
    import sacrebleu

    if not text_a or not text_b:
        return 0.0
    bleu = sacrebleu.sentence_bleu(text_b, [text_a])
    return round(bleu.score, 4)


def rouge_scores(text_a: str, text_b: str) -> dict[str, dict[str, float]]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(text_a, text_b)
    result: dict[str, dict[str, float]] = {}
    for key in ("rouge1", "rouge2", "rougeL"):
        s = scores[key]
        result[key] = {
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "fmeasure": round(s.fmeasure, 4),
        }
    return result


def edit_distance(text_a: str, text_b: str) -> int:
    from Levenshtein import distance

    return distance(text_a, text_b)


def edit_distance_normalized(text_a: str, text_b: str) -> float:
    dist = edit_distance(text_a, text_b)
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 0.0
    return round(dist / max_len, 4)


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = _get_semantic_model()
    embeddings = model.encode([text_a, text_b])
    a = embeddings[0]
    b = embeddings[1]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Clamp to [0.0, 1.0] range
    return round(float(max(0.0, min(1.0, cos_sim))), 4)


def first_divergence_token(text_a: str, text_b: str) -> int:
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    min_len = min(len(tokens_a), len(tokens_b))
    for i in range(min_len):
        if tokens_a[i] != tokens_b[i]:
            return i
    # All shared-position tokens match; check if lengths differ
    if len(tokens_a) != len(tokens_b):
        return min_len
    return -1


def compare_texts(
    text_a: str, text_b: str, include_semantic: bool = True
) -> ComparisonResult:
    rouge = rouge_scores(text_a, text_b)
    sem_sim = semantic_similarity(text_a, text_b) if include_semantic else -1.0

    tokens_a = text_a.split()
    tokens_b = text_b.split()

    return ComparisonResult(
        text_a=text_a,
        text_b=text_b,
        exact_match=exact_match(text_a, text_b),
        token_match_ratio=token_match_ratio(text_a, text_b),
        bleu=bleu_score(text_a, text_b),
        rouge_1_f=rouge["rouge1"]["fmeasure"],
        rouge_2_f=rouge["rouge2"]["fmeasure"],
        rouge_l_f=rouge["rougeL"]["fmeasure"],
        edit_distance=edit_distance(text_a, text_b),
        edit_distance_normalized=edit_distance_normalized(text_a, text_b),
        semantic_similarity=sem_sim,
        first_divergence_token=first_divergence_token(text_a, text_b),
        num_tokens_a=len(tokens_a),
        num_tokens_b=len(tokens_b),
    )
