from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .scoring import (
    TOOL_CALL_RE,
    combined_cot_score,
    response_quality,
    stable_tie,
    strict_candidate_ok,
    style_diversity_score,
)
from .trajectories import (
    build_conversations_from_response,
    canonical_targets,
    extract_answer_steps,
    format_for_token_count,
    has_answer_before_tool_response,
    render_prompt,
    token_count_text,
    validate_function_calls,
)


class CandidateRejected(ValueError):
    pass


def build_candidate(
    row: dict[str, Any],
    file_label: str,
    line_number: int,
    prompt_template: str,
    tools_text: str,
    tokenizer: Any | None,
    max_chat_template_tokens: int,
) -> dict[str, Any]:
    target = str(row.get("target") or "")
    canonical = canonical_targets(target)
    if canonical is None:
        raise CandidateRejected("invalid_target")
    response = str(row.get("response") or "")
    if has_answer_before_tool_response(response):
        raise CandidateRejected("answer_before_tool_response")
    answer_steps = extract_answer_steps(response)
    if not answer_steps:
        raise CandidateRejected("invalid_answer")
    try:
        conversations, warnings = build_conversations_from_response(render_prompt(prompt_template, target), response)
        validate_function_calls(conversations)
    except (ValueError, TypeError, AttributeError) as exc:
        raise CandidateRejected("conversation_parse_or_function_call_invalid") from exc
    token_length = None
    if max_chat_template_tokens > 0:
        try:
            token_length = token_count_text(tokenizer, format_for_token_count(conversations, tools_text))
        except Exception as exc:
            raise CandidateRejected("tokenizer_failed") from exc
        if token_length > max_chat_template_tokens:
            raise CandidateRejected("chat_template_tokens_gt_max")
    quality = response_quality(row, answer_steps)
    diversity = style_diversity_score(row, quality, response)
    if any(isinstance(value, float) and not math.isfinite(value) for value in diversity.values()):
        raise CandidateRejected("non_finite_score")
    key = f"{file_label}:{line_number}:{row.get('doc_id')}:{row.get('trial_index')}:{row.get('sample_index')}"
    score = combined_cot_score(row, quality, diversity, key)
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return {
        "source_file": file_label,
        "source": str(row.get("dataset") or "unknown"),
        "doc_id": str(row.get("doc_id") or ""),
        "source_index": row.get("source_index"),
        "target": target,
        "canonical_target": canonical[0],
        "canonical_noiso": canonical[1],
        "trial_index": int(row.get("trial_index", row.get("sample_index", 0)) or 0),
        "sample_index": int(row.get("sample_index", row.get("trial_index", 0)) or 0),
        "line_number": line_number,
        "response_length": int(row.get("response_length") or 0),
        "answer_steps": answer_steps,
        "route_step_count": len(answer_steps),
        "tool_call_count": int(reward.get("tool_call_count") or len(TOOL_CALL_RE.findall(response))),
        "tool_response_count": int(reward.get("tool_response_count") or response.lower().count("<tool_response>")),
        "quality": quality,
        "diversity": diversity,
        "combined_cot_score": round(score, 6),
        "chat_template_token_length": token_length,
        "conversations": conversations,
        "parse_warnings": warnings,
    }


def _store_candidate(connection: sqlite3.Connection, candidate: dict[str, Any]) -> bool:
    key = candidate["canonical_noiso"]
    score = candidate["combined_cot_score"]
    tie = stable_tie(f"{candidate['doc_id']}:{candidate['trial_index']}:{candidate['sample_index']}")
    previous = connection.execute("SELECT score, tie FROM winners WHERE target = ?", (key,)).fetchone()
    if previous is not None:
        connection.execute("UPDATE winners SET seen = seen + 1 WHERE target = ?", (key,))
        if (score, tie) <= previous:
            return False
    payload = json.dumps(candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    connection.execute(
        "INSERT INTO winners (target, score, tie, target_tie, payload, seen) VALUES (?, ?, ?, ?, ?, 1) "
        "ON CONFLICT(target) DO UPDATE SET score = excluded.score, tie = excluded.tie, payload = excluded.payload",
        (key, score, tie, stable_tie(key), payload),
    )
    return True


def _export(connection: sqlite3.Connection, directory: Path, prefix: str, tools_text: str) -> tuple[int, Counter]:
    warnings: Counter[str] = Counter()
    count = 0
    connection.execute("CREATE INDEX ranking ON winners (score DESC, target_tie DESC)")
    with (
        (directory / f"{prefix}.json").open("w", encoding="utf-8") as output,
        (directory / f"{prefix}.selected_cots.jsonl").open("w", encoding="utf-8") as scores,
    ):
        output.write("[\n")
        cursor = connection.execute("SELECT payload, seen FROM winners ORDER BY score DESC, target_tie DESC, rowid ASC")
        for payload, seen in cursor:
            candidate = json.loads(payload)
            conversations = candidate.pop("conversations")
            warnings.update(candidate.pop("parse_warnings"))
            count += 1
            candidate["rank"] = count
            candidate["strict_samples_seen_for_target"] = seen
            scores.write(json.dumps(candidate, ensure_ascii=False, allow_nan=False) + "\n")
            record = {
                "conversations": conversations,
                "tools": tools_text,
                "source": "reasoning_bootstrapping",
                "doc_id": candidate["doc_id"],
                "trial_index": candidate["trial_index"],
                "sample_index": candidate["sample_index"],
                "target": candidate["target"],
                "selection": {
                    key: candidate[key]
                    for key in (
                        "rank",
                        "source",
                        "source_file",
                        "line_number",
                        "canonical_target",
                        "canonical_noiso",
                        "combined_cot_score",
                        "quality",
                        "diversity",
                        "response_length",
                        "chat_template_token_length",
                        "route_step_count",
                        "tool_call_count",
                        "strict_samples_seen_for_target",
                    )
                },
            }
            record["selection"]["cot_quality_score"] = candidate["quality"]["cot_quality_score"]
            record["selection"]["reasoning_style_diversity_score"] = candidate["diversity"][
                "reasoning_style_diversity_score"
            ]
            if count > 1:
                output.write(",\n")
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
        output.write("\n]\n")
        for handle in (output, scores):
            handle.flush()
            os.fsync(handle.fileno())
    return count, warnings


def curate_trajectories(
    samples: Sequence[Path | str],
    *,
    output_dir: Path | str,
    prompt_template: str,
    tools_text: str,
    tokenizer: Any | None = None,
    max_response_length: int = 32000,
    max_chat_template_tokens: int = 32000,
    prefix: str = "curated",
    sample_labels: Sequence[str] | None = None,
    work_dir: Path | str | None = None,
    progress_every: int = 5000,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in samples]
    if not paths:
        raise ValueError("at least one rollout JSONL file is required")
    if not prefix or prefix in {".", ".."} or "/" in prefix or "\\" in prefix:
        raise ValueError("prefix must be a filename component")
    if max_chat_template_tokens > 0 and tokenizer is None:
        raise ValueError("provide a tokenizer or explicitly disable max_chat_template_tokens with 0")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    labels = list(sample_labels) if sample_labels is not None else [str(path) for path in paths]
    if len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError("sample labels must be unique and match the number of inputs")
    if "{target}" not in prompt_template and "{TARGET_MOLECULE}" not in prompt_template:
        raise ValueError("prompt template must contain {target} or {TARGET_MOLECULE}")
    schema = json.loads(tools_text)
    if not isinstance(schema, list) or not schema:
        raise ValueError("tool schema must be a nonempty JSON list")
    tools_text = json.dumps(schema, ensure_ascii=False, allow_nan=False)
    output_dir = Path(output_dir)
    names = (f"{prefix}.json", f"{prefix}.selected_cots.jsonl", f"{prefix}.report.json")
    destinations = {(output_dir / name).resolve() for name in names}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.resolve() in destinations:
            raise ValueError("output must not replace an input file")
    output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    with (
        tempfile.TemporaryDirectory(prefix="curation-", dir=work_dir) as workspace,
        tempfile.TemporaryDirectory(prefix=f".{prefix}-", dir=output_dir) as staging,
    ):
        connection = sqlite3.connect(str(Path(workspace) / "winners.sqlite3"))
        try:
            connection.execute("PRAGMA cache_size = -8192")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                "CREATE TABLE winners (target TEXT PRIMARY KEY, score REAL NOT NULL, tie REAL NOT NULL, "
                "target_tie REAL NOT NULL, payload TEXT NOT NULL, seen INTEGER NOT NULL)"
            )
            for path, label in zip(paths, labels):
                counters["files"] += 1
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        counters["records"] += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            counters["bad_json"] += 1
                            row = None
                        else:
                            if not isinstance(row, dict):
                                counters["non_dict"] += 1
                                row = None
                        if row is not None:
                            counters["rows"] += 1
                            if not isinstance(row.get("status"), str) or not row["status"].strip():
                                raise ValueError(
                                    f"{label}:{line_number}: missing or invalid generation status; preserve the "
                                    "rollout producer's explicit status. A successful reward does not prove completion."
                                )
                            ok, reasons = strict_candidate_ok(row)
                            if not ok:
                                counters["strict_rejected"] += 1
                                rejects.update(
                                    "status_not_completed" if r.startswith("status_") else r for r in reasons
                                )
                            else:
                                try:
                                    length = int(row.get("response_length") or 0)
                                    if max_response_length > 0 and length > max_response_length:
                                        raise CandidateRejected("response_length_gt_max")
                                    candidate = build_candidate(
                                        row,
                                        label,
                                        line_number,
                                        prompt_template,
                                        tools_text,
                                        tokenizer,
                                        max_chat_template_tokens,
                                    )
                                    updated = _store_candidate(connection, candidate)
                                except CandidateRejected as exc:
                                    counters["unusable"] += 1
                                    rejects[str(exc)] += 1
                                except (ValueError, TypeError, OverflowError, AttributeError):
                                    counters["unusable"] += 1
                                    rejects["invalid_record_fields"] += 1
                                else:
                                    counters["strict_usable_samples"] += 1
                                    counters["best_updated"] += int(updated)
                        if counters["records"] % 1000 == 0:
                            connection.commit()
                        if progress and progress_every and counters["records"] % progress_every == 0:
                            progress(dict(counters))
            connection.commit()
            count, warnings = _export(connection, Path(staging), prefix, tools_text)
        finally:
            connection.close()
        report = {
            "policy": {
                "target_filter": "none",
                "dedup_key": "canonical target without stereochemistry",
                "max_response_length": max_response_length,
                "max_chat_template_tokens": max_chat_template_tokens,
                "token_length_filter_enabled": max_chat_template_tokens > 0,
                "score": "0.85 * quality + 0.15 * diversity - 0.30 * repetition + 0.0005 * stable_tie",
            },
            "counters": dict(counters),
            "reject_reasons": dict(rejects),
            "selected_targets": count,
            "parse_warnings": dict(warnings),
        }
        with (Path(staging) / names[-1]).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for name in names:
            os.replace(Path(staging) / name, output_dir / name)
    return report
