from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)


def parse_route_step(step: str) -> tuple[str, list[str]]:
    parent, precursors = step.split(">>", 1)
    children = [item.strip() for item in precursors.split(".") if item.strip()]
    return parent.strip(), children


def extract_answer_steps(text: str | None) -> list[str] | None:
    if not isinstance(text, str):
        return None
    matches = list(ANSWER_TAG_RE.finditer(text))
    payload = matches[-1].group(1).strip() if matches else text.strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not value:
        return None
    steps = []
    for item in value:
        if not isinstance(item, str) or ">>" not in item:
            return None
        steps.append(item.strip())
    return steps


def build_route_graph(route: Iterable[str]) -> tuple[dict[str, set[str]], dict[str, int], set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    indeg: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for step in route:
        parent, children = parse_route_step(step)
        nodes.add(parent)
        adj.setdefault(parent, set())
        indeg.setdefault(parent, indeg.get(parent, 0))
        for child in children:
            nodes.add(child)
            if child not in adj[parent]:
                adj[parent].add(child)
                indeg[child] += 1
            adj.setdefault(child, set())
            indeg.setdefault(child, indeg.get(child, 0))
    return dict(adj), dict(indeg), nodes


def extract_leaf_nodes(route: Iterable[str]) -> list[str]:
    adj, _indeg, nodes = build_route_graph(route)
    return sorted(node for node in nodes if not adj.get(node))


def route_root(route: Iterable[str]) -> str:
    _adj, indeg, nodes = build_route_graph(route)
    roots = sorted(node for node in nodes if indeg.get(node, 0) == 0)
    return roots[0] if roots else ""


def reachable_nodes_from_root(route: Iterable[str], root: str | None = None) -> set[str]:
    adj, indeg, nodes = build_route_graph(route)
    root = root or route_root(route)
    if not root or root not in nodes:
        return set()
    seen = {root}
    queue: deque[str] = deque([root])
    while queue:
        node = queue.popleft()
        for child in adj.get(node, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def canonical_parent_smiles(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    try:
        parent = rdMolStandardize.FragmentParent(mol)
        parent = uncharger.uncharge(parent)
        return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_step_key(step: str) -> str | None:
    try:
        parent, children = parse_route_step(step)
    except Exception:
        return None
    canonical_parent = canonical_parent_smiles(parent)
    if not canonical_parent:
        return None
    canonical_children = []
    for child in children:
        canonical_child = canonical_parent_smiles(child)
        if not canonical_child:
            return None
        canonical_children.append(canonical_child)
    return f"{canonical_parent}>>{'.'.join(sorted(canonical_children))}"


def route_record_to_ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    route = list(metadata.get("route") or [])
    target = str(metadata.get("target") or row.get("target") or route_root(route))
    return {
        "target": target,
        "materials": extract_leaf_nodes(route),
        "retro_routes": [route],
    }
