from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S | re.I)
CHAT_MARKER_RE = re.compile(r"<\|im_(?:start|end)\|>(?:assistant|user|system)?\n?", re.I)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-]*")
THINK_TAG_RE = re.compile(r"</?think>", re.I)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*.*?\s*</tool_response>", re.S | re.I)
TOOL_RESPONSE_CONTENT_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S | re.I)

EXPLORATION_TERMS = (
    "perhaps",
    "might",
    "could",
    "maybe",
    "possibly",
    "consider",
    "considering",
    "try",
    "trying",
    "alternative",
    "alternatively",
    "option",
    "instead",
    "however",
    "rather than",
    "on the other hand",
    "wait",
    "actually",
    "reconsider",
    "reconsidering",
    "revise",
    "revising",
    "unavailable",
    "not ideal",
    "fallback",
)
SELF_VERIFICATION_TERMS = (
    "check",
    "checks",
    "checked",
    "checking",
    "verify",
    "verifies",
    "verified",
    "verifying",
    "validate",
    "validates",
    "validated",
    "validating",
    "confirm",
    "confirms",
    "confirmed",
    "confirming",
    "ensure",
    "ensures",
    "ensured",
    "ensuring",
    "recheck",
    "rechecked",
    "consistent",
    "consistency",
    "inconsistent",
    "inconsistency",
    "stock_ok",
    "stock ok",
)
ADAPTIVE_GRANULARITY_TERMS = (
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "accordingly",
    "as a result",
    "which means",
    "this suggests",
    "this indicates",
    "suggesting",
    "indicating",
    "given that",
    "due to",
    "in order to",
    "so that",
    "thereby",
)
REQUIRED_REWARD_FLAGS = (
    "parse_ok",
    "route_tree_valid",
    "root_target_match",
    "non_trivial_route",
    "all_leaf_in_emolecules",
    "core_chemical_valid",
    "final_route_stock_checked_true",
    "tool_trace_healthy",
    "final_answer_parse_ok",
)


def strict_candidate_ok(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    if row.get("status") != "completed":
        reasons.append(f"status_{row.get('status')}")
    if not (bool(row.get("success")) or bool(reward.get("success"))):
        reasons.append("not_success")
    for key in REQUIRED_REWARD_FLAGS:
        if not bool(reward.get(key)):
            reasons.append(f"{key}_false")
    return not reasons, reasons


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    lower = (text or "").lower()
    total = 0
    for term in terms:
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        total += len(re.findall(rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])", lower))
    return total


def normalized_keyword_score(
    text: str, terms: tuple[str, ...], expected_per_1000_words: float
) -> tuple[float, int, int]:
    word_count = len(WORD_RE.findall(text or ""))
    term_count = count_terms(text, terms)
    denom = max(1.0, word_count / 1000.0 * expected_per_1000_words)
    return min(float(term_count) / denom, 1.0), term_count, word_count


def reasoning_text_for_quality(response: str) -> str:
    text = str(response or "")
    for pattern in (TOOL_RESPONSE_RE, TOOL_CALL_RE, ANSWER_RE, CHAT_MARKER_RE, THINK_TAG_RE):
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def repeated_ngram_ratio(text: str, n: int = 5, max_words: int = 8000) -> float:
    words = WORD_RE.findall(text.lower())
    if len(words) > max_words:
        step = max(1, len(words) // max_words)
        words = words[::step][:max_words]
    if len(words) <= n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def response_quality(row: dict[str, Any], answer_steps: list[str] | None = None) -> dict[str, float | int]:
    response = str(row.get("response") or "")
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    tool_calls = int(reward.get("tool_call_count") or len(TOOL_CALL_RE.findall(response)))
    reasoning_text = reasoning_text_for_quality(response)
    word_count = len(WORD_RE.findall(reasoning_text))
    elaborated = word_count / 8000.0
    verification, verification_terms, _ = normalized_keyword_score(reasoning_text, SELF_VERIFICATION_TERMS, 6.0)
    exploration, exploration_terms, _ = normalized_keyword_score(reasoning_text, EXPLORATION_TERMS, 6.0)
    exploratory = 0.80 * exploration + 0.20 * min(tool_calls / 3.0, 1.0)
    adaptive, adaptive_terms, _ = normalized_keyword_score(reasoning_text, ADAPTIVE_GRANULARITY_TERMS, 8.0)
    score = 0.30 * elaborated + 0.20 * verification + 0.25 * exploratory + 0.25 * adaptive
    return {
        "elaborated_reasoning": round(elaborated, 6),
        "self_verification": round(verification, 6),
        "exploratory_approach": round(exploratory, 6),
        "adaptive_granularity": round(adaptive, 6),
        "reasoning_word_count": word_count,
        "self_verification_terms": verification_terms,
        "exploratory_terms": exploration_terms,
        "adaptive_granularity_terms": adaptive_terms,
        "repeated_ngram_ratio": round(repeated_ngram_ratio(response), 6),
        "cot_quality_score": round(score, 6),
    }


def stock_sequence_from_response(response: str) -> list[bool]:
    sequence = []
    for raw in TOOL_RESPONSE_CONTENT_RE.findall(response or ""):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "stock_ok" in payload:
            sequence.append(bool(payload["stock_ok"]))
    return sequence


def triangular_score(value: float, low: float, peak: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return max(0.0, min(1.0, (value - low) / max(1e-9, peak - low)))
    return max(0.0, min(1.0, (high - value) / max(1e-9, high - peak)))


def balanced_dimension_score(quality: dict[str, Any]) -> float:
    dims = [
        float(quality.get(key) or 0.0)
        for key in (
            "elaborated_reasoning",
            "self_verification",
            "exploratory_approach",
            "adaptive_granularity",
        )
    ]
    capped = [min(max(value, 0.0), 1.0) for value in dims]
    total = sum(capped)
    if total <= 0:
        return 0.0
    probs = [value / total for value in capped if value > 0]
    return -sum(prob * math.log(prob) for prob in probs) / math.log(4)


def style_diversity_score(row: dict[str, Any], quality: dict[str, Any], response: str) -> dict[str, float | int | str]:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    tool_calls = int(reward.get("tool_call_count") or len(TOOL_CALL_RE.findall(response)))
    sequence = stock_sequence_from_response(response)
    if sequence and sequence[-1] and any(not value for value in sequence[:-1]):
        stock_score, stock_pattern = 1.0, "false_to_true"
    elif sequence and all(sequence):
        stock_score, stock_pattern = 0.45, "all_true"
    elif sequence:
        stock_score, stock_pattern = 0.2, "other_tool_sequence"
    else:
        stock_score, stock_pattern = 0.0, "no_tool_response"
    try:
        nll = float(row.get("sampled_token_nll"))
    except (ValueError, TypeError, OverflowError):
        nll = 0.0
    nll_score = triangular_score(nll, low=0.02, peak=0.45, high=1.25)
    tool_shape = min(tool_calls / 3.0, 1.0)
    balance = balanced_dimension_score(quality)
    score = 0.35 * balance + 0.25 * nll_score + 0.20 * tool_shape + 0.20 * stock_score
    return {
        "reasoning_style_diversity_score": round(score, 6),
        "dimension_balance_score": round(balance, 6),
        "nll_style_score": round(nll_score, 6),
        "tool_shape_score": round(tool_shape, 6),
        "stock_pattern_score": round(stock_score, 6),
        "stock_pattern": stock_pattern,
        "sampled_token_nll": round(nll, 6),
        "stock_check_count": len(sequence),
    }


def stable_tie(text: str) -> float:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def combined_cot_score(row: dict[str, Any], quality: dict[str, Any], diversity: dict[str, Any], key: str) -> float:
    return (
        0.85 * float(quality.get("cot_quality_score") or 0.0)
        + 0.15 * float(diversity.get("reasoning_style_diversity_score") or 0.0)
        - 0.30 * float(quality.get("repeated_ngram_ratio") or 0.0)
        + 0.0005 * stable_tie(key)
    )
