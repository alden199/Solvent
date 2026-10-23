from __future__ import annotations

import json
import re
from typing import Any

from rdkit import Chem

from .scoring import ANSWER_RE, CHAT_MARKER_RE, TOOL_CALL_RE

TOOL_NAME = "check_emolecules_stock"
ALLOWED_ATOMS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"}
TOOL_RESPONSE_SPLIT_RE = re.compile(
    r"<\|im_end\|>\s*<\|im_start\|>user\s*<tool_response>\s*(.*?)\s*</tool_response>\s*"
    r"<\|im_end\|>\s*<\|im_start\|>assistant\n?",
    re.S,
)


def canonical_targets(smiles: str) -> tuple[str, str] | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None
    if len(fragments) != 1:
        return None
    mol = fragments[0]
    if {atom.GetSymbol() for atom in mol.GetAtoms()} - ALLOWED_ATOMS:
        return None
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
    )


def extract_answer_steps(text: str) -> list[str] | None:
    matches = list(ANSWER_RE.finditer(text))
    payload = matches[-1].group(1).strip() if matches else text.strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, str) or ">>" not in item for item in value):
        return None
    return [item.strip() for item in value]


def has_answer_before_tool_response(response: str) -> bool:
    position = 0
    for match in TOOL_RESPONSE_SPLIT_RE.finditer(response or ""):
        if ANSWER_RE.search(response[position : match.start()]):
            return True
        position = match.end()
    return False


def parse_tool_payload(payload: str) -> dict[str, Any]:
    data = json.loads(payload.strip())
    if not isinstance(data, dict):
        raise ValueError("tool payload is not a JSON object")
    name = data.get("name") or (data.get("function") or {}).get("name")
    args = data.get("arguments")
    if args is None:
        args = (data.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {"answer_route": args}
        except json.JSONDecodeError:
            args = {"answer_route": args}
    if not isinstance(args, dict):
        args = {"answer_route": json.dumps(args, ensure_ascii=False)}
    if isinstance(args.get("answer_route"), list):
        args["answer_route"] = json.dumps(args["answer_route"], ensure_ascii=False)
    return {"name": str(name or TOOL_NAME), "arguments": args}


def function_call_value(assistant_text: str, payload: dict[str, Any]) -> str:
    reasoning = CHAT_MARKER_RE.sub("", TOOL_CALL_RE.sub("", assistant_text or "")).strip()
    if not reasoning:
        reasoning = "<think>\n</think>"
    elif not reasoning.lower().startswith("<think>"):
        reasoning = f"<think>\n{reasoning}\n</think>"
    return f"{reasoning}\n\n{json.dumps(payload, ensure_ascii=False)}"


def validate_function_calls(conversations: list[dict[str, str]]) -> None:
    thought_re = re.compile(r"<think>\n(.*?)\n</think>\n\n", re.S)
    for message in conversations:
        if message.get("from") != "function_call":
            continue
        content = str(message.get("value") or "")
        thought = thought_re.search(content)
        if thought:
            content = content.replace(thought.group(0), "")
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            parsed = [parsed]
        for call in parsed:
            if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
                raise ValueError("invalid function_call payload")


def build_conversations_from_response(prompt: str, response: str) -> tuple[list[dict[str, str]], list[str]]:
    conversations = [{"from": "human", "value": prompt}]
    warnings = []
    position = 0
    tool_turns = 0
    for match in TOOL_RESPONSE_SPLIT_RE.finditer(response):
        assistant_text = response[position : match.start()]
        payloads = TOOL_CALL_RE.findall(assistant_text)
        if not payloads:
            raise ValueError("assistant segment before tool response has no tool_call")
        if len(payloads) > 1:
            warnings.append("multiple_tool_calls_in_one_assistant_segment")
        payload = parse_tool_payload(payloads[-1])
        if payload.get("name") != TOOL_NAME:
            warnings.append("unexpected_tool_name")
        conversations.append({"from": "function_call", "value": function_call_value(assistant_text, payload)})
        conversations.append({"from": "observation", "value": match.group(1).strip()})
        tool_turns += 1
        position = match.end()
    final_text = CHAT_MARKER_RE.sub("", response[position:] or "").strip()
    matches = list(ANSWER_RE.finditer(final_text))
    if not matches:
        raise ValueError("final assistant segment has no answer block")
    final_text = final_text[: matches[-1].end()].strip()
    if "<tool_response>" in final_text.lower():
        raise ValueError("final assistant segment contains tool_response")
    conversations.append({"from": "gpt", "value": final_text})
    if tool_turns == 0:
        raise ValueError("no parsed tool turns")
    return conversations, warnings


def render_prompt(template: str, target: str) -> str:
    return template.replace("{target}", target).replace("{TARGET_MOLECULE}", target)


def format_for_token_count(conversations: list[dict[str, str]], tools: str) -> str:
    chunks = [f"<tools>\n{tools}\n</tools>"]
    for message in conversations:
        role = str(message.get("from") or "unknown")
        value = str(message.get("value") or "")
        chunks.append(f"<{role}>\n{value}\n</{role}>")
    return "\n\n".join(chunks)


def token_count_text(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    return len(encoded["input_ids"])
