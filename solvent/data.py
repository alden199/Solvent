from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from solvent.chem_utils import canonical_parent_smiles, extract_leaf_nodes, route_record_to_ground_truth
from solvent.rl.generate_with_emol import _canonical_route_steps, _is_single_root_tree, build_route_graph


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = PROMPT_DIR / "cotv1_easy_toolonly_prompt.txt"
DEFAULT_TOOLS = PROMPT_DIR / "check_emolecules_stock_tool_schema.json"


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Install pyarrow to read parquet input.") from exc
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    if path.suffix != ".jsonl":
        raise ValueError("Input must be .jsonl or .parquet; JSON arrays are not accepted.")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {number}.")
            yield row


@contextmanager
def atomic_jsonl(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".solvent-", suffix=".jsonl", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yield handle
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_prompt_and_tools(prompt_path: Path = DEFAULT_PROMPT, tool_path: Path = DEFAULT_TOOLS):
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    tools = json.loads(Path(tool_path).read_text(encoding="utf-8"))
    if "{target}" not in prompt:
        raise ValueError("Prompt must contain {target}.")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tools must be a nonempty JSON list.")
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if function.get("name") != "check_emolecules_stock":
            raise ValueError("Only the check_emolecules_stock tool is supported.")
        parameters = function.get("parameters", {})
        if parameters.get("properties", {}).get("answer_route", {}).get("type") != "string":
            raise ValueError("Tool must accept a string answer_route parameter.")
    return prompt, tools


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("label") or (row.get("reward_model") or {}).get("ground_truth")
    if isinstance(label, str):
        label = json.loads(label)
    if not label:
        if "route" in row:
            label = route_record_to_ground_truth({"metadata": {"target": row.get("target"), "route": row["route"]}})
        else:
            label = route_record_to_ground_truth(row)
        if label.get("retro_routes") == [[]]:
            label["retro_routes"] = []
    if not isinstance(label, dict):
        raise ValueError("label must be an object or a JSON-encoded object.")
    label = dict(label)
    target = label.get("target") or row.get("target")
    if not isinstance(target, str) or not target.strip() or not canonical_parent_smiles(target):
        raise ValueError("Every record needs a valid target SMILES.")
    label["target"] = target
    routes = label.get("retro_routes") or []
    if not isinstance(routes, list):
        raise ValueError("retro_routes must be a list of routes.")
    for route in routes:
        if not isinstance(route, list) or not all(isinstance(step, str) for step in route):
            raise ValueError("Each route must be a nonempty list of PRODUCT>>PRECURSORS strings.")
        canonical, error = _canonical_route_steps(route)
        if error:
            raise ValueError(error)
        valid, root, error = _is_single_root_tree(*build_route_graph(canonical))
        if not valid or root != canonical_parent_smiles(target):
            raise ValueError(error or "Route root does not match target.")
    label["retro_routes"] = routes
    label["materials"] = label.get("materials") or (extract_leaf_nodes(routes[0]) if routes else [])
    return label


def prepare_data(source: Path, output: Path, *, prompt_path: Path = DEFAULT_PROMPT, tool_path: Path = DEFAULT_TOOLS):
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Input and output must be different paths.")
    prompt_template, tools = load_prompt_and_tools(prompt_path, tool_path)
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    count = 0
    with atomic_jsonl(output) as handle:
        for index, row in enumerate(iter_records(source)):
            try:
                label = normalize_record(row)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid record {index + 1}: {exc}") from exc
            target = label["target"]
            sample = {
                "prompt": prompt_template.replace("{target}", target),
                "label": label,
                "metadata": {
                    "doc_id": str(row.get("doc_id") or f"sample:{index}"),
                    "source_index": index,
                    "target": target,
                    "gt_route_len": len(label["retro_routes"][0]) if label["retro_routes"] else None,
                    "gt_leaf_count": len(label["materials"]),
                },
                "tools": tools_json,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return {"rows": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate user targets/routes and prepare Slime JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--tool-schema", type=Path, default=DEFAULT_TOOLS)
    args = parser.parse_args()
    try:
        result = prepare_data(args.input, args.output, prompt_path=args.prompt_template, tool_path=args.tool_schema)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
