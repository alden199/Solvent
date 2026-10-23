from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from solvent.chem_utils import canonical_parent_smiles
from solvent.data import atomic_jsonl, iter_records, normalize_record
from solvent.rl import generate_with_emol as core
from solvent.stock import validate_stock


def add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stock-db", dest="retro_emol_stock_db", type=Path)
    parser.add_argument("--template-file", dest="retro_emol_template_file", type=Path)
    parser.add_argument("--top-n-templates", dest="retro_emol_top_n_templates", type=int, default=10000)
    parser.add_argument("--reward-mode", dest="retro_emol_reward_mode", default="gated_success_format",
                        choices=["default", "gated_success_format"])
    parser.add_argument("--overlong-penalty", dest="retro_emol_overlong_penalty_enable", action="store_true")
    parser.set_defaults(retro_emol_overlong_start_len=16000, retro_emol_overlong_max_response_len=24000,
                        retro_emol_overlong_penalty_factor=0.5)


def validate_resources(args: Any) -> None:
    validate_stock(core._stock_db_path(args))
    path = core._template_file_path(args)
    if not path.is_file():
        raise ValueError(f"Template file does not exist: {path}")
    if core._top_n_templates(args) <= 0:
        raise ValueError("--top-n-templates must be positive.")
    if not core._compiled_retro_templates(path, core._top_n_templates(args)):
        raise ValueError("Template file must be gzip JSON containing usable reaction_smarts records.")


def pass_at_k(n: int, correct: int, k: int) -> float:
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= correct <= n and 1 <= k <= n.")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod((n - correct - i) / (n - i) for i in range(k))


def evaluate_file(source: Path, output: Path, args: Any, ks: tuple[int, ...] = (1,)) -> dict[str, Any]:
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Input and output must be different paths.")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Pass@k values must be positive.")
    validate_resources(args)
    groups = defaultdict(lambda: [0, 0])
    count = success_count = 0
    score_sum = 0.0
    with atomic_jsonl(output) as handle:
        for index, row in enumerate(iter_records(source)):
            label = normalize_record(row)
            if getattr(args, "for_curation", False):
                status = row.get("status")
                if not isinstance(status, str) or not status.strip():
                    raise ValueError(
                        f"Record {index + 1} needs its original generation status for curation; "
                        "evaluation cannot infer completed status."
                    )
            response = row.get("response")
            if not isinstance(response, str):
                raise ValueError(f"Record {index + 1} needs a string response containing the full tool trace.")
            length = row.get("response_length")
            if length is None and core._overlong_penalty_enabled(args):
                raise ValueError("response_length is required with --overlong-penalty.")
            if length is not None and (type(length) is not int or length < 0):
                raise ValueError("response_length must be a nonnegative integer.")
            sample = SimpleNamespace(label=label, response=response, response_length=length or 0)
            reward = core.compute_retro_emol_reward_with_args(args, sample)
            success = int(reward["success"])
            target = canonical_parent_smiles(label["target"])
            groups[target][0] += 1
            groups[target][1] += success
            count += 1
            success_count += success
            score_sum += reward["score"]
            handle.write(json.dumps({**row, "target": target, "label": label, "reward": reward,
                                     "success": success}) + "\n")
    summary = {
        "samples": count,
        "targets": len(groups),
        "success_rate": success_count / count if count else 0.0,
        "mean_score": score_sum / count if count else 0.0,
        "pass_at_k": {},
    }
    for k in ks:
        values = [pass_at_k(n, correct, k) for n, correct in groups.values() if n >= k]
        summary["pass_at_k"][str(k)] = {
            "value": sum(values) / len(values) if values else None,
            "eligible_targets": len(values),
            "excluded_targets": len(groups) - len(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score JSONL responses using the Solvent route verifier.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--for-curation", action="store_true",
                        help="Require original generation status for downstream trajectory curation.")
    add_resource_arguments(parser)
    args = parser.parse_args()
    try:
        summary = evaluate_file(args.input, args.output, args, tuple(args.k))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
