from __future__ import annotations

import json
import re
import sqlite3
import os
import gzip
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from solvent.chem_utils import build_route_graph, canonical_parent_smiles, extract_answer_steps, extract_leaf_nodes


DEFAULT_MAX_TURNS = 5
DEFAULT_MAX_CONTEXT_LEN = 32768
DEFAULT_CONTEXT_SAFETY_MARGIN = 8
DEFAULT_TOOL_SCHEMA = Path(__file__).resolve().parents[1] / "prompts" / "check_emolecules_stock_tool_schema.json"
DEFAULT_TOP_N_TEMPLATES = 10_000
TOOL_NAME = "check_emolecules_stock"
CORE_SOURCE_ATOMIC_NUMS = {6, 7, 14, 15, 16}
STRICT_SOURCE_ATOMIC_NUMS = {5, 6, 7, 9, 14, 15, 16, 17, 35, 53}
_TEMPLATE_CACHE: dict[tuple[str, int], list[tuple[Chem.Mol, tuple[Chem.Mol, ...]]]] = {}
_CHEM_VALID_CACHE: dict[tuple[str, tuple[str, ...], str, int, int, str], tuple[bool, str]] = {}
_STOCK_CONN_CACHE: dict[str, sqlite3.Connection] = {}
_STOCK_HIT_CACHE: dict[tuple[str, str], bool] = {}

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    '<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, '
    '"arguments": <args-json-object>}}\n</tool_call>'
)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _max_turns(args: Any) -> int:
    return int(_get_arg(args, "retro_emol_max_turns", _get_arg(args, "max_turns", DEFAULT_MAX_TURNS)) or DEFAULT_MAX_TURNS)


def _context_limit(args: Any) -> int:
    configured = _get_arg(args, "retro_emol_max_context_len", DEFAULT_MAX_CONTEXT_LEN) or DEFAULT_MAX_CONTEXT_LEN
    rollout_limit = _get_arg(args, "rollout_max_context_len", None)
    if rollout_limit:
        configured = min(int(configured), int(rollout_limit))
    return min(int(configured), DEFAULT_MAX_CONTEXT_LEN)


def _context_safety_margin(args: Any) -> int:
    return max(0, int(_get_arg(args, "retro_emol_context_safety_margin", DEFAULT_CONTEXT_SAFETY_MARGIN) or 0))


def _stock_db_path(args: Any) -> Path:
    configured = _get_arg(args, "retro_emol_stock_db", None) or os.environ.get("SOLVENT_STOCK_DB") or os.environ.get("STOCK_DB")
    if not configured:
        raise ValueError("Configure retro_emol_stock_db or SOLVENT_STOCK_DB with a user-built inventory.")
    return Path(configured).expanduser().resolve()


def _template_file_path(args: Any) -> Path:
    configured = _get_arg(args, "retro_emol_template_file", None) or os.environ.get("SOLVENT_TEMPLATE_FILE") or os.environ.get("TEMPLATE_FILE")
    if not configured:
        raise ValueError("Configure retro_emol_template_file or SOLVENT_TEMPLATE_FILE with user-supplied templates.")
    return Path(configured).expanduser().resolve()


def _top_n_templates(args: Any) -> int:
    return int(_get_arg(args, "retro_emol_top_n_templates", DEFAULT_TOP_N_TEMPLATES) or DEFAULT_TOP_N_TEMPLATES)


def _reward_mode(args: Any) -> str:
    mode = str(_get_arg(args, "retro_emol_reward_mode", "default") or "default").strip().lower()
    if mode not in {"default", "gated_success_format"}:
        raise ValueError(f"Unsupported release reward mode: {mode}")
    return mode


def _is_gated_success_format_reward(args: Any) -> bool:
    return _reward_mode(args) == "gated_success_format"


def _success_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_success_reward_weight", 0.6) or 0.0)


def _final_route_stock_checked_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_final_route_stock_checked_reward_weight", 0.2) or 0.0)


def _tool_trace_valid_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_tool_trace_valid_reward_weight", 0.2) or 0.0)


def _gated_success_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_gated_success_reward_weight", 0.8) or 0.0)


def _gated_final_route_stock_checked_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_gated_final_route_stock_checked_reward_weight", 0.1) or 0.0)


def _gated_tool_trace_valid_weight(args: Any) -> float:
    return float(_get_arg(args, "retro_emol_gated_tool_trace_valid_reward_weight", 0.1) or 0.0)


def _reward_components(args: Any) -> tuple[tuple[str, float], ...]:
    if _is_gated_success_format_reward(args):
        return (
            ("r_success", _gated_success_weight(args)),
            ("r_final_route_stock_checked", _gated_final_route_stock_checked_weight(args)),
            ("r_tool_trace_valid", _gated_tool_trace_valid_weight(args)),
        )
    return (
        ("r_success", _success_weight(args)),
        ("r_final_route_stock_checked", _final_route_stock_checked_weight(args)),
        ("r_tool_trace_valid", _tool_trace_valid_weight(args)),
    )


def _overlong_penalty_enabled(args: Any) -> bool:
    return bool(_get_arg(args, "retro_emol_overlong_penalty_enable", False))


def _overlong_metrics(args: Any, sample: Sample) -> dict[str, float | int]:
    if not _overlong_penalty_enabled(args):
        return {
            "overlong": 0,
            "overlong_penalty": 0.0,
            "overlong_additive_reward": 0.0,
            "overlong_start_len": 0,
            "overlong_max_len": 0,
            "overlong_exceed_len": 0,
        }

    start_len = int(_get_arg(args, "retro_emol_overlong_start_len", 20000) or 20000)
    max_len = int(
        _get_arg(
            args,
            "retro_emol_overlong_max_response_len",
            _get_arg(args, "rollout_max_response_len", 0),
        )
        or 0
    )
    penalty_factor = float(_get_arg(args, "retro_emol_overlong_penalty_factor", 1.0) or 0.0)
    if max_len <= start_len:
        return {
            "overlong": 0,
            "overlong_penalty": 0.0,
            "overlong_additive_reward": 0.0,
            "overlong_start_len": int(start_len),
            "overlong_max_len": int(max_len),
            "overlong_exceed_len": 0,
        }

    response_len = int(sample.response_length or 0)
    exceed_len = max(response_len - start_len, 0)
    if exceed_len <= 0:
        penalty = 0.0
    else:
        penalty = -min(exceed_len / (max_len - start_len), 1.0) * penalty_factor

    return {
        "overlong": int(exceed_len > 0),
        "overlong_penalty": float(penalty),
        "overlong_additive_reward": float(penalty),
        "overlong_start_len": int(start_len),
        "overlong_max_len": int(max_len),
        "overlong_exceed_len": int(exceed_len),
    }


def _load_default_tools() -> list[dict[str, Any]]:
    return json.loads(DEFAULT_TOOL_SCHEMA.read_text(encoding="utf-8"))


def _normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return _load_default_tools()
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise TypeError(f"tools must be a list or JSON string, got {type(tools)}")
    return tools


def _qwen_tool_prompt(tools: list[dict[str, Any]]) -> str:
    tool_text = ""
    for tool in tools:
        wrapped = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
        tool_text += "\n" + json.dumps(wrapped, ensure_ascii=False)
    return QWEN_TOOL_PROMPT.format(tool_text=tool_text)


def render_qwen3_nothink_initial_prompt(user_prompt: str, tools: list[dict[str, Any]]) -> str:
    system_text = _qwen_tool_prompt(tools) if tools else ""
    prompt = ""
    if system_text:
        prompt += f"<|im_start|>system\n{system_text}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def render_qwen3_nothink_observation(observation: str, *, assistant_already_closed: bool = False) -> str:
    prefix = "" if assistant_already_closed else "<|im_end|>\n"
    return prefix + (
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{observation}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_last_tool_call(text: str) -> dict[str, Any] | None:
    matches = TOOL_CALL_RE.findall(text or "")
    for payload in reversed(matches):
        try:
            parsed = json.loads(payload.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed
    return None


def _has_final_answer(text: str) -> bool:
    return bool(ANSWER_RE.search(text or ""))


def _assistant_already_closed(text: str) -> bool:
    return (text or "").rstrip().endswith("<|im_end|>")


def _sqlite_hit(conn: sqlite3.Connection, canonical_parent: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM emolecules WHERE canonical_parent = ? LIMIT 1",
            (canonical_parent,),
        ).fetchone()
    )


def _cached_sqlite_hit(stock_db: Path, canonical_parent: str) -> bool:
    db_key = str(stock_db)
    cache_key = (db_key, canonical_parent)
    if cache_key in _STOCK_HIT_CACHE:
        return _STOCK_HIT_CACHE[cache_key]
    conn = _STOCK_CONN_CACHE.get(db_key)
    if conn is None:
        conn = sqlite3.connect(stock_db.resolve().as_uri() + "?mode=ro", uri=True)
        _STOCK_CONN_CACHE[db_key] = conn
    hit = _sqlite_hit(conn, canonical_parent)
    _STOCK_HIT_CACHE[cache_key] = hit
    return hit


def _route_key(steps: list[str] | None) -> tuple[str, ...] | None:
    if not steps:
        return None
    keys: list[str] = []
    for step in steps:
        try:
            parent, precursors = step.split(">>", 1)
        except ValueError:
            return None
        canonical_parent = canonical_parent_smiles(parent)
        if not canonical_parent:
            return None
        canonical_children = []
        for child in precursors.split("."):
            child = child.strip()
            if not child:
                continue
            canonical_child = canonical_parent_smiles(child)
            if not canonical_child:
                return None
            canonical_children.append(canonical_child)
        if not canonical_children:
            return None
        keys.append(f"{canonical_parent}>>{'.'.join(sorted(canonical_children))}")
    return tuple(sorted(keys))


def _canonical_route_steps(steps: list[str] | None) -> tuple[list[str] | None, str | None]:
    if not steps:
        return None, "missing final answer route"
    canonical_steps: list[str] = []
    for step in steps:
        try:
            parent, children = step.split(">>", 1)
        except ValueError as exc:
            return None, f"step parse error: {exc}"
        canonical_parent = canonical_parent_smiles(parent)
        if not canonical_parent:
            return None, f"invalid parent SMILES: {parent}"
        canonical_children: list[str] = []
        for child in children.split("."):
            child = child.strip()
            if not child:
                continue
            canonical_child = canonical_parent_smiles(child)
            if not canonical_child:
                return None, f"invalid precursor SMILES: {child}"
            canonical_children.append(canonical_child)
        if not canonical_children:
            return None, f"step has no precursors: {step}"
        if canonical_parent in canonical_children:
            return None, f"parent equals child after canonicalization: {canonical_parent}"
        canonical_steps.append(f"{canonical_parent}>>{'.'.join(sorted(canonical_children))}")
    return canonical_steps, None


def _is_single_root_tree(adj: dict[str, set[str]], indeg: dict[str, int], nodes: set[str]) -> tuple[bool, str | None, str | None]:
    roots = sorted(node for node in nodes if indeg.get(node, 0) == 0)
    if len(roots) != 1:
        return False, roots[0] if roots else None, f"invalid_root_count:{len(roots)}"
    root = roots[0]
    for node in nodes:
        if node == root:
            continue
        if indeg.get(node, 0) != 1:
            return False, root, f"node_indegree_not_one:{node}:{indeg.get(node, 0)}"

    visited: set[str] = set()
    active: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        active.add(node)
        for child in adj.get(node, ()):
            if child not in visited:
                if not dfs(child):
                    return False
            elif child in active:
                return False
        active.remove(node)
        return True

    if not dfs(root):
        return False, root, "cycle_detected"
    if visited != nodes:
        return False, root, "not_all_nodes_reachable"
    return True, root, None


def _validate_final_route(
    steps: list[str] | None,
    label: dict[str, Any],
    stock_db: Path,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "parse_ok": False,
        "route_tree_valid": False,
        "root_target_match": False,
        "non_trivial_route": False,
        "all_leaf_in_emolecules": False,
        "root": None,
        "canonical_target": None,
        "leaves": [],
        "route_error": None,
    }
    canonical_steps, error = _canonical_route_steps(steps)
    if error or canonical_steps is None:
        details["route_error"] = error
        return details
    details["parse_ok"] = True

    try:
        adj, indeg, nodes = build_route_graph(canonical_steps)
        tree_ok, root, tree_error = _is_single_root_tree(adj, indeg, nodes)
    except Exception as exc:
        details["route_error"] = f"graph build error: {type(exc).__name__}: {exc}"
        return details

    details["route_tree_valid"] = bool(tree_ok)
    details["root"] = root
    if tree_error:
        details["route_error"] = tree_error

    target = label.get("target") or (label.get("metadata") or {}).get("target")
    canonical_target = canonical_parent_smiles(str(target or ""))
    details["canonical_target"] = canonical_target
    details["root_target_match"] = bool(root and canonical_target and root == canonical_target)
    details["non_trivial_route"] = bool(len(canonical_steps) > 0 and len(nodes) > 1)

    leaves = sorted(node for node in nodes if not adj.get(node))
    details["leaves"] = leaves
    if leaves and stock_db.exists():
        details["all_leaf_in_emolecules"] = all(_cached_sqlite_hit(stock_db, leaf) for leaf in leaves)
    return details


def _all_route_smiles_valid(steps: list[str] | None) -> bool:
    if not steps:
        return False
    try:
        _adj, _indeg, nodes = build_route_graph(steps)
    except Exception:
        return False
    return all(canonical_parent_smiles(node) is not None for node in nodes)


def _canonical_smiles_noiso(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _atom_counts(mols: list[Chem.Mol]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 1:
                continue
            counts[atomic_num] = counts.get(atomic_num, 0) + 1
    return counts


def _atom_source_sanity(
    product_mol: Chem.Mol,
    precursor_mols: list[Chem.Mol],
    max_missing_oxygen: int,
    mode: str,
) -> tuple[bool, str]:
    product_counts = _atom_counts([product_mol])
    precursor_counts = _atom_counts(precursor_mols)
    deficits = {
        atomic_num: product_count - precursor_counts.get(atomic_num, 0)
        for atomic_num, product_count in product_counts.items()
        if product_count > precursor_counts.get(atomic_num, 0)
    }
    if not deficits:
        return True, ""
    if mode not in {"core", "strict"}:
        return False, f"unknown atom sanity mode: {mode}"
    checked_atomic_nums = STRICT_SOURCE_ATOMIC_NUMS if mode == "strict" else CORE_SOURCE_ATOMIC_NUMS
    fatal: list[str] = []
    periodic_table = Chem.GetPeriodicTable()
    for atomic_num, deficit in sorted(deficits.items()):
        if atomic_num in checked_atomic_nums:
            fatal.append(f"{periodic_table.GetElementSymbol(atomic_num)}+{deficit}")
        elif atomic_num == 8 and deficit > max_missing_oxygen:
            fatal.append(f"O+{deficit}")
    if fatal:
        return False, "product has missing atom sources: " + ", ".join(fatal)
    return True, ""


def _split_smarts_side(side: str) -> list[str]:
    return [part for part in side.split(".") if part]


def _stream_reaction_smarts(path: Path, limit: int) -> list[str]:
    pattern = re.compile(r'"reaction_smarts"\s*:\s*"((?:\\.|[^"\\])*)"')
    templates: list[str] = []
    carry = ""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                text = carry
                carry = ""
            else:
                text = carry + chunk
            matches = list(pattern.finditer(text))
            carry = text[matches[-1].end() :] if matches else (text[-1000:] if chunk else "")
            for match in matches:
                try:
                    smarts = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    smarts = match.group(1)
                templates.append(smarts)
                if limit > 0 and len(templates) >= limit:
                    return templates
            if not chunk:
                break
    return templates


def _compiled_retro_templates(template_file: Path, limit: int) -> list[tuple[Chem.Mol, tuple[Chem.Mol, ...]]]:
    cache_key = (str(template_file), int(limit))
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    compiled: list[tuple[Chem.Mol, tuple[Chem.Mol, ...]]] = []
    if template_file.exists():
        for smarts in _stream_reaction_smarts(template_file, limit):
            if ">>" not in smarts:
                continue
            lhs, rhs = smarts.split(">>", 1)
            lhs_query = Chem.MolFromSmarts(lhs)
            if lhs_query is None:
                continue
            rhs_queries: list[Chem.Mol] = []
            for fragment in _split_smarts_side(rhs):
                rhs_query = Chem.MolFromSmarts(fragment)
                if rhs_query is None:
                    rhs_queries = []
                    break
                rhs_queries.append(rhs_query)
            if rhs_queries:
                compiled.append((lhs_query, tuple(rhs_queries)))
    _TEMPLATE_CACHE[cache_key] = compiled
    return compiled


def _rhs_matches_precursors(rhs_queries: tuple[Chem.Mol, ...], precursor_mols: list[Chem.Mol]) -> bool:
    if not rhs_queries:
        return False
    used = [False] * len(precursor_mols)
    queries = sorted(rhs_queries, key=lambda query: query.GetNumAtoms(), reverse=True)

    def backtrack(index: int) -> bool:
        if index == len(queries):
            return True
        query = queries[index]
        for mol_index, mol in enumerate(precursor_mols):
            if used[mol_index]:
                continue
            try:
                matched = mol.HasSubstructMatch(query)
            except Exception:
                matched = False
            if not matched:
                continue
            used[mol_index] = True
            if backtrack(index + 1):
                return True
            used[mol_index] = False
        return False

    return backtrack(0)


def _normalize_step_for_chem(step: str) -> tuple[str, tuple[str, ...], Chem.Mol, list[Chem.Mol]] | tuple[None, None, None, None]:
    try:
        product, precursors = step.split(">>", 1)
    except ValueError:
        return None, None, None, None
    product_can = _canonical_smiles_noiso(product.strip())
    product_mol = _mol_from_smiles(product.strip())
    if product_can is None or product_mol is None:
        return None, None, None, None
    precursor_cans: list[str] = []
    precursor_mols: list[Chem.Mol] = []
    for precursor in precursors.split("."):
        precursor = precursor.strip()
        if not precursor:
            continue
        precursor_can = _canonical_smiles_noiso(precursor)
        precursor_mol = _mol_from_smiles(precursor)
        if precursor_can is None or precursor_mol is None:
            return None, None, None, None
        precursor_cans.append(precursor_can)
        precursor_mols.append(precursor_mol)
    if not precursor_cans:
        return None, None, None, None
    return product_can, tuple(sorted(precursor_cans)), product_mol, precursor_mols


def _step_core_chemical_valid(step: str, args: Any) -> tuple[bool, str]:
    normalized = _normalize_step_for_chem(step)
    product_can, precursor_cans, product_mol, precursor_mols = normalized
    if product_can is None or precursor_cans is None or product_mol is None or precursor_mols is None:
        return False, "step SMILES cannot be parsed"

    template_file = _template_file_path(args)
    top_n_templates = _top_n_templates(args)
    max_missing_oxygen = int(_get_arg(args, "retro_emol_max_missing_oxygen", 2) or 2)
    atom_sanity_mode = str(_get_arg(args, "retro_emol_atom_sanity_mode", "core") or "core")
    cache_key = (product_can, precursor_cans, str(template_file), int(top_n_templates), max_missing_oxygen, atom_sanity_mode)
    cached = _CHEM_VALID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    atom_ok, atom_reason = _atom_source_sanity(product_mol, precursor_mols, max_missing_oxygen, atom_sanity_mode)
    if not atom_ok:
        result = (False, atom_reason)
        _CHEM_VALID_CACHE[cache_key] = result
        return result

    templates = _compiled_retro_templates(template_file, top_n_templates)
    if not templates:
        result = (False, f"no usable retro templates loaded from {template_file}")
        _CHEM_VALID_CACHE[cache_key] = result
        return result

    for lhs_query, rhs_queries in templates:
        try:
            product_match = product_mol.HasSubstructMatch(lhs_query)
        except Exception:
            product_match = False
        if product_match and _rhs_matches_precursors(rhs_queries, precursor_mols):
            result = (True, "")
            _CHEM_VALID_CACHE[cache_key] = result
            return result

    result = (False, "no top-frequency retro template matched this step")
    _CHEM_VALID_CACHE[cache_key] = result
    return result


def _route_core_chemical_valid(steps: list[str] | None, args: Any) -> dict[str, Any]:
    if not bool(_get_arg(args, "retro_emol_core_chemical_validity_enable", True)):
        return {
            "core_chemical_valid": True,
            "core_chemical_valid_enabled": False,
            "core_chemical_valid_steps": [],
            "core_chemical_invalid_reasons": [],
        }
    if not steps:
        return {
            "core_chemical_valid": False,
            "core_chemical_valid_enabled": True,
            "core_chemical_valid_steps": [],
            "core_chemical_invalid_reasons": ["missing final answer route"],
        }
    step_results: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for index, step in enumerate(steps):
        ok, reason = _step_core_chemical_valid(step, args)
        step_results.append({"index": index, "valid": int(ok), "reason": reason})
        if not ok:
            invalid_reasons.append(f"step_{index}: {reason}")
    return {
        "core_chemical_valid": all(item["valid"] for item in step_results),
        "core_chemical_valid_enabled": True,
        "core_chemical_valid_steps": step_results,
        "core_chemical_invalid_reasons": invalid_reasons,
    }


def _normalize_label(label: Any) -> dict[str, Any]:
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            return {}
    return label if isinstance(label, dict) else {}


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in TOOL_CALL_RE.findall(text or ""):
        record: dict[str, Any] = {
            "raw": payload,
            "parse_ok": False,
            "name_ok": False,
            "arguments_ok": False,
            "route_parse_ok": False,
            "route_key": None,
            "error": None,
        }
        try:
            parsed = json.loads(payload.strip())
        except json.JSONDecodeError as exc:
            record["error"] = f"tool_call JSON parse error: {exc}"
            records.append(record)
            continue

        if not isinstance(parsed, dict):
            record["error"] = "tool_call payload is not a JSON object"
            records.append(record)
            continue

        record["parse_ok"] = True
        record["name"] = parsed.get("name")
        record["name_ok"] = parsed.get("name") == TOOL_NAME

        arguments = parsed.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"answer_route": arguments}

        if not isinstance(arguments, dict) or "answer_route" not in arguments:
            record["error"] = "tool_call missing arguments.answer_route"
            records.append(record)
            continue

        record["arguments_ok"] = True
        answer_route = str(arguments.get("answer_route", ""))
        record["answer_route"] = answer_route
        route_steps = extract_answer_steps(answer_route)
        record["route_parse_ok"] = bool(route_steps)
        record["route_key"] = _route_key(route_steps)
        if not record["name_ok"]:
            record["error"] = f"unknown tool {parsed.get('name')!r}"
        elif not route_steps:
            record["error"] = "answer_route is not a parseable route"
        records.append(record)
    return records


def _parse_tool_responses(text: str) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for payload in TOOL_RESPONSE_RE.findall(text or ""):
        record: dict[str, Any] = {"raw": payload, "parse_ok": False, "observation": None, "error": None}
        try:
            parsed = json.loads(payload.strip())
        except json.JSONDecodeError as exc:
            record["error"] = f"tool_response JSON parse error: {exc}"
            responses.append(record)
            continue
        if not isinstance(parsed, dict):
            record["error"] = "tool_response payload is not a JSON object"
            responses.append(record)
            continue
        record["parse_ok"] = True
        record["observation"] = parsed
        if parsed.get("error"):
            record["error"] = str(parsed.get("error"))
        responses.append(record)
    return responses


def _tool_trace_metrics(text: str, final_route_key: tuple[str, ...] | None) -> dict[str, Any]:
    calls = _parse_tool_calls(text)
    responses = _parse_tool_responses(text)
    paired = []
    final_route_stock_checked_true = False
    for idx, call in enumerate(calls):
        response = responses[idx] if idx < len(responses) else None
        observation = response.get("observation") if response else None
        paired.append({"call": call, "response": response})
        if (
            final_route_key is not None
            and call.get("route_key") == final_route_key
            and isinstance(observation, dict)
            and observation.get("stock_ok") is True
        ):
            final_route_stock_checked_true = True

    all_calls_ok = bool(calls) and all(
        call.get("parse_ok")
        and call.get("name_ok")
        and call.get("arguments_ok")
        and call.get("route_parse_ok")
        and call.get("route_key") is not None
        and not call.get("error")
        for call in calls
    )
    all_responses_ok = len(responses) == len(calls) and all(
        response.get("parse_ok")
        and isinstance(response.get("observation"), dict)
        and not response.get("error")
        for response in responses
    )

    return {
        "tool_call_count": len(calls),
        "tool_response_count": len(responses),
        "tool_trace_healthy": int(all_calls_ok and all_responses_ok),
        "final_route_stock_checked_true": int(final_route_stock_checked_true),
        "tool_pairs": paired,
        "tool_errors": [
            item
            for item in [*(call.get("error") for call in calls), *(resp.get("error") for resp in responses)]
            if item
        ],
    }


def check_emolecules_stock(answer_route: str, stock_db: Path) -> str:
    if not stock_db.exists():
        return json.dumps({"stock_ok": False, "error": f"eMolecules stock database not found: {stock_db}"})

    steps = extract_answer_steps(answer_route)
    if not steps:
        return json.dumps(
            {
                "stock_ok": False,
                "error": "Could not parse answer_route as a non-empty JSON list of PRODUCT>>PRECURSORS steps.",
            },
            ensure_ascii=False,
        )

    try:
        leaves = extract_leaf_nodes(steps)
        adj, _indeg, nodes = build_route_graph(steps)
    except Exception as exc:
        return json.dumps({"stock_ok": False, "error": f"Could not parse route graph: {exc}"}, ensure_ascii=False)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []

    conn = sqlite3.connect(stock_db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for node in sorted(nodes):
            all_nodes.append(
                {
                    "node": node,
                    "canonical_parent": canonical_parent_smiles(node),
                    "is_leaf": not bool(adj.get(node)),
                }
            )

        for leaf in leaves:
            canonical_parent = canonical_parent_smiles(leaf)
            if not canonical_parent:
                item = {"leaf": leaf, "canonical_parent": None, "in_emolecules": False}
            else:
                item = {
                    "leaf": leaf,
                    "canonical_parent": canonical_parent,
                    "in_emolecules": _sqlite_hit(conn, canonical_parent),
                }
            rows.append(item)
            if not item["in_emolecules"]:
                missing.append(item)
    finally:
        conn.close()

    return json.dumps(
        {
            "stock_ok": not missing,
            "leaf_count": len(leaves),
            "missing_count": len(missing),
            "missing": missing[:20],
            "leaves": rows,
            "all_nodes": all_nodes,
            "interpretation": (
                "stock_ok=false means at least one predicted leaf is not locally available in eMolecules and the "
                "route must be revised. stock_ok=true only checks availability of predicted leaves; it is not a "
                "correctness check."
            ),
        },
        ensure_ascii=False,
    )


def _execute_tool_call(tool_call: dict[str, Any], stock_db: Path) -> tuple[str, dict[str, Any]]:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"answer_route": arguments}

    if name != TOOL_NAME:
        observation = {"stock_ok": False, "error": f"Unknown tool: {name}. Only {TOOL_NAME} is available."}
        return json.dumps(observation, ensure_ascii=False), observation

    if not isinstance(arguments, dict) or "answer_route" not in arguments:
        observation = {"stock_ok": False, "error": "Missing required argument: answer_route."}
        return json.dumps(observation, ensure_ascii=False), observation

    observation_text = check_emolecules_stock(str(arguments.get("answer_route", "")), stock_db)
    try:
        observation = json.loads(observation_text)
    except json.JSONDecodeError:
        observation = {"stock_ok": False, "error": "Tool returned non-JSON observation.", "raw": observation_text}
    return observation_text, observation


def _safe_execute_tool_call(tool_call: dict[str, Any], stock_db: Path) -> tuple[str, dict[str, Any]]:
    try:
        return _execute_tool_call(tool_call, stock_db)
    except Exception as exc:
        observation = {
            "stock_ok": False,
            "error": f"Tool execution failed: {type(exc).__name__}: {exc}",
        }
        return json.dumps(observation, ensure_ascii=False), observation


def _append_tokens(
    *,
    sample: Sample,
    tokenizer,
    text: str,
    token_ids: list[int] | None,
    log_probs: list[float] | None,
    response_tokens: list[int],
    loss_mask_val: int,
    max_context_len: int,
) -> bool:
    from slime.utils.types import Sample

    if token_ids is None:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if log_probs is None:
        log_probs = [0.0] * len(token_ids)

    remaining = max_context_len - len(sample.tokens)
    if remaining <= 0:
        sample.status = Sample.Status.TRUNCATED
        return False
    if len(token_ids) > remaining:
        token_ids = token_ids[:remaining]
        log_probs = log_probs[:remaining]
        sample.status = Sample.Status.TRUNCATED

    sample.tokens.extend(token_ids)
    response_tokens.extend(token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([loss_mask_val] * len(token_ids))
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend(log_probs)
    sample.response_length = len(response_tokens)
    return sample.status != Sample.Status.TRUNCATED


def _finish_sample(sample: Sample, tokenizer, response_tokens: list[int]) -> Sample:
    from slime.utils.types import Sample

    sample.response = tokenizer.decode(response_tokens)
    sample.response_length = len(response_tokens)
    sample.reward = None
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    if sample.loss_mask is None:
        sample.loss_mask = []
    if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != sample.response_length:
        sample.status = Sample.Status.ABORTED
    if len(sample.loss_mask) != sample.response_length:
        sample.status = Sample.Status.ABORTED
    return sample


def compute_retro_emol_reward(sample: Sample) -> dict[str, Any]:
    return compute_retro_emol_reward_with_args(None, sample)


def compute_retro_emol_reward_with_args(args: Any, sample: Sample) -> dict[str, Any]:
    label = _normalize_label(sample.label)
    response = sample.response or ""

    final_steps = extract_answer_steps(response)
    final_route_key = _route_key(final_steps)
    route_details = _validate_final_route(final_steps, label, _stock_db_path(args))
    chem_details = _route_core_chemical_valid(final_steps, args)
    tool_metrics = _tool_trace_metrics(response, final_route_key)
    overlong_metrics = _overlong_metrics(args, sample) if args is not None else _overlong_metrics(None, sample)

    parse_ok = int(route_details["parse_ok"])
    route_tree_valid = int(route_details["route_tree_valid"])
    root_target_match = int(route_details["root_target_match"])
    non_trivial_route = int(route_details["non_trivial_route"])
    all_leaf_in_emolecules = int(route_details["all_leaf_in_emolecules"])
    core_chemical_valid = int(chem_details["core_chemical_valid"])
    final_route_stock_checked_true = int(tool_metrics["final_route_stock_checked_true"])
    tool_trace_healthy = int(tool_metrics["tool_trace_healthy"])

    r_success = int(
        bool(
            parse_ok
            and route_tree_valid
            and root_target_match
            and non_trivial_route
            and all_leaf_in_emolecules
            and core_chemical_valid
        )
    )
    r_stock_tree = int(
        bool(parse_ok and route_tree_valid and root_target_match and non_trivial_route and all_leaf_in_emolecules)
    )
    r_chem = int(bool(parse_ok and route_tree_valid and root_target_match and non_trivial_route and core_chemical_valid))
    raw_r_final_route_stock_checked = final_route_stock_checked_true
    raw_r_tool_trace_valid = int(bool(tool_trace_healthy and parse_ok))
    gate_format_reward = _is_gated_success_format_reward(args)
    if gate_format_reward and not r_success:
        r_final_route_stock_checked = 0
        r_tool_trace_valid = 0
    else:
        r_final_route_stock_checked = raw_r_final_route_stock_checked
        r_tool_trace_valid = raw_r_tool_trace_valid

    component_values = {
        "r_success": float(r_success),
        "r_stock_tree": float(r_stock_tree),
        "r_chem": float(r_chem),
        "r_final_route_stock_checked": float(r_final_route_stock_checked),
        "r_tool_trace_valid": float(r_tool_trace_valid),
    }
    base_score = sum(weight * component_values[name] for name, weight in _reward_components(args))
    score = base_score + float(overlong_metrics["overlong_additive_reward"])

    return {
        "score": float(score),
        "base_score": float(base_score),
        "r_success": float(r_success),
        "r_stock": float(r_stock_tree),
        "r_stock_tree": float(r_stock_tree),
        "r_chem": float(r_chem),
        "r_final_route_stock_checked": float(r_final_route_stock_checked),
        "r_tool_trace_valid": float(r_tool_trace_valid),
        "raw_r_final_route_stock_checked": float(raw_r_final_route_stock_checked),
        "raw_r_tool_trace_valid": float(raw_r_tool_trace_valid),
        "reward_mode_gated_success_format": int(_is_gated_success_format_reward(args)),
        "format_reward_gate_success": int(gate_format_reward),
        "success": r_success,
        "stock_tree_success": r_stock_tree,
        "chem_success": r_chem,
        "parse_ok": parse_ok,
        "route_tree_valid": route_tree_valid,
        "root_target_match": root_target_match,
        "non_trivial_route": non_trivial_route,
        "all_leaf_in_emolecules": all_leaf_in_emolecules,
        "core_chemical_valid": core_chemical_valid,
        "final_route_stock_checked_true": final_route_stock_checked_true,
        "tool_trace_healthy": tool_trace_healthy,
        "final_answer_parse_ok": int(final_steps is not None),
        "final_route_key_ok": int(final_route_key is not None),
        "tool_call_count": tool_metrics["tool_call_count"],
        "tool_response_count": tool_metrics["tool_response_count"],
        "tool_errors": tool_metrics["tool_errors"],
        "route": route_details,
        "core_chemical_validity": chem_details,
        "reward_weights": dict(_reward_components(args)),
        **overlong_metrics,
    }


def _ensure_reward_dict(sample: Sample, args: Any = None) -> dict[str, Any]:
    if isinstance(sample.reward, dict):
        return sample.reward
    reward = compute_retro_emol_reward_with_args(args, sample)
    sample.reward = reward
    return reward


def _iter_samples(obj: Any):
    from slime.utils.types import Sample

    if isinstance(obj, Sample):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_samples(item)


def mask_truncated_samples_for_loss(args: Any, samples: list[Sample] | list[list[Sample]]) -> None:
    from slime.utils.types import Sample

    if not bool(_get_arg(args, "retro_emol_mask_truncated_samples_for_loss", True)):
        return

    for sample in _iter_samples(samples):
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata["retro_emol_truncated_loss_masked"] = 1


def dynamic_filter_drop_all_success_or_all_parse_fail(
    args: Any, samples: list[Sample] | list[list[Sample]], **kwargs
) -> DynamicFilterOutput:
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    flat_samples = list(_iter_samples(samples))
    if not flat_samples:
        return DynamicFilterOutput(keep=False, reason="empty_group")

    rewards = [_ensure_reward_dict(sample, args) for sample in flat_samples]
    successes = [int(float(reward.get("success", 0.0)) == 1.0) for reward in rewards]
    final_answer_parse_ok = [int(float(reward.get("final_answer_parse_ok", 0.0)) == 1.0) for reward in rewards]

    if all(successes):
        return DynamicFilterOutput(keep=False, reason="all_success")
    if not any(final_answer_parse_ok):
        return DynamicFilterOutput(keep=False, reason="all_final_answer_parse_fail")
    return DynamicFilterOutput(keep=True)


def dynamic_filter_drop_all_zero_or_all_full_stock_chem(
    args: Any, samples: list[Sample] | list[list[Sample]], **kwargs
) -> DynamicFilterOutput:
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    flat_samples = list(_iter_samples(samples))
    if not flat_samples:
        return DynamicFilterOutput(keep=False, reason="empty_group")

    rewards = [_ensure_reward_dict(sample, args) for sample in flat_samples]
    stock_chem_states = [
        (
            int(float(reward.get("r_stock_tree", reward.get("stock_tree_success", 0.0))) == 1.0),
            int(float(reward.get("r_chem", reward.get("chem_success", 0.0))) == 1.0),
        )
        for reward in rewards
    ]

    if all(stock == 0 and chem == 0 for stock, chem in stock_chem_states):
        return DynamicFilterOutput(keep=False, reason="all_stock_chem_zero")
    if all(stock == 1 and chem == 1 for stock, chem in stock_chem_states):
        return DynamicFilterOutput(keep=False, reason="all_stock_chem_full")
    return DynamicFilterOutput(keep=True)


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    assert not args.partial_rollout, "Partial rollout is not supported for retro eMolecules rollouts."

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_context_len = _context_limit(args)
    max_turns = _max_turns(args)
    stock_db = _stock_db_path(args)

    tools = _normalize_tools(sample.metadata.get("tools") if sample.metadata else None)
    prompt_text = render_qwen3_nothink_initial_prompt(str(sample.prompt), tools)
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    sample.prompt = prompt_text
    sample.tokens = list(prompt_tokens)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.status = Sample.Status.PENDING
    sample.reward = None
    sample.metadata = sample.metadata or {}

    trace = {
        "max_turns": max_turns,
        "max_context_len": max_context_len,
        "tool_call_count": 0,
        "stock_ok_history": [],
        "feedback_mode": "true",
        "tool_observations": [],
        "final_answer_found": False,
        "finish_reasons": [],
        "errors": [],
    }
    response_tokens: list[int] = []

    if len(sample.tokens) >= max_context_len:
        sample.status = Sample.Status.TRUNCATED
        sample.metadata["retro_emol_rollout"] = trace
        return _finish_sample(sample, tokenizer, response_tokens)

    for _turn_idx in range(max_turns):
        remaining = max_context_len - len(sample.tokens)
        if remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        per_turn_sampling_params = dict(sampling_params)
        if (
            per_turn_sampling_params.get("stop") is None
            and per_turn_sampling_params.get("stop_token_ids") is None
        ):
            per_turn_sampling_params["stop"] = ["<|im_end|>"]
        configured_max_new = per_turn_sampling_params.get(
            "max_new_tokens",
            _get_arg(args, "rollout_max_response_len", remaining),
        )
        safe_remaining = remaining - _context_safety_margin(args)
        if safe_remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        per_turn_sampling_params["max_new_tokens"] = max(1, min(int(configured_max_new), safe_remaining))
        repetition_penalty = float(_get_arg(args, "retro_emol_repetition_penalty", 1.0) or 1.0)
        if repetition_penalty != 1.0:
            per_turn_sampling_params["repetition_penalty"] = repetition_penalty

        payload = {
            "input_ids": sample.tokens,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)
        finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type", "unknown")
        trace["finish_reasons"].append(finish_type)

        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            break

        logprob_rows = output.get("meta_info", {}).get("output_token_logprobs")
        if not logprob_rows:
            trace["errors"].append("missing output_token_logprobs")
            sample.status = Sample.Status.ABORTED
            break

        cur_token_ids = [item[1] for item in logprob_rows]
        cur_log_probs = [item[0] for item in logprob_rows]
        cur_text = tokenizer.decode(cur_token_ids)
        if not _append_tokens(
            sample=sample,
            tokenizer=tokenizer,
            text=cur_text,
            token_ids=cur_token_ids,
            log_probs=cur_log_probs,
            response_tokens=response_tokens,
            loss_mask_val=1,
            max_context_len=max_context_len,
        ):
            break

        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        tool_call = _extract_last_tool_call(cur_text)
        if tool_call is None:
            trace["final_answer_found"] = _has_final_answer(tokenizer.decode(response_tokens))
            break

        observation_text, observation = _safe_execute_tool_call(tool_call, stock_db)
        route_text = ""
        arguments = tool_call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"answer_route": arguments}
        if isinstance(arguments, dict):
            route_text = str(arguments.get("answer_route", ""))
        delivered_observation = observation
        trace["tool_call_count"] += 1
        trace["stock_ok_history"].append(observation.get("stock_ok"))
        trace["tool_observations"].append(
            {
                "turn": _turn_idx + 1,
                "queried_route": route_text,
                "true": observation,
                "delivered": delivered_observation,
            }
        )
        if "error" in observation:
            trace["errors"].append(str(observation["error"]))

        obs_turn = render_qwen3_nothink_observation(
            observation_text,
            assistant_already_closed=_assistant_already_closed(cur_text),
        )
        if not _append_tokens(
            sample=sample,
            tokenizer=tokenizer,
            text=obs_turn,
            token_ids=None,
            log_probs=None,
            response_tokens=response_tokens,
            loss_mask_val=0,
            max_context_len=max_context_len,
        ):
            break

    trace["final_answer_found"] = trace["final_answer_found"] or _has_final_answer(tokenizer.decode(response_tokens))
    sample.metadata["retro_emol_rollout"] = trace
    return _finish_sample(sample, tokenizer, response_tokens)


async def reward_func(args: Any, sample: Sample | list[Sample], **kwargs) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(sample, list):
        return [compute_retro_emol_reward_with_args(args, item) for item in sample]
    return compute_retro_emol_reward_with_args(args, sample)
