from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from solvent.chem_utils import canonical_parent_smiles, extract_answer_steps
from solvent.data import DEFAULT_PROMPT, DEFAULT_TOOLS, atomic_jsonl, load_prompt_and_tools
from solvent.evaluate import add_resource_arguments, validate_resources
from solvent.rl import generate_with_emol as core
from solvent.stock import validate_stock


def load_local_tokenizer(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--tokenizer must point to a local Hugging Face tokenizer directory.")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Inference requires transformers==4.57.1, matching the pinned RL dependencies.") from exc
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


class SGLangClient:
    def __init__(self, endpoint: str, tokenizer: Any, *, timeout: float = 600.0,
                 api_key: str | None = None, transport: Any = None):
        import httpx

        self.tokenizer = tokenizer
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=endpoint.rstrip("/"), timeout=timeout, headers=headers, transport=transport)

    def close(self) -> None:
        self.client.close()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict) or value.get("error"):
            raise ValueError(f"SGLang returned an invalid response from {path}.")
        return value

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
            raise ValueError("Tokenizer must return a flat input_ids list.")
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


def infer_target(
    target: str,
    client: SGLangClient,
    args: Any,
    *,
    prompt_template: str,
    tools: list[dict[str, Any]],
    sampling_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(target, str) or not canonical_parent_smiles(target):
        raise ValueError("Target must be a valid nonempty SMILES string.")
    stock_db = core._stock_db_path(args)
    validate_stock(stock_db)
    max_turns = core._max_turns(args)
    max_context_len = core._context_limit(args)
    if max_turns < 1 or max_context_len < 1:
        raise ValueError("Turn and context limits must be positive.")
    prompt = prompt_template.replace("{target}", target)
    rendered = core.render_qwen3_nothink_initial_prompt(prompt, tools)
    tokens = client.encode(rendered)
    initial_length = len(tokens)
    response_tokens: list[int] = []
    generated_token_count = 0
    sampled_token_nll_count = 0
    sampled_token_nll_invalid_count = 0
    sampled_token_nll = 0.0
    observations = []
    finish_reasons = []
    status = "completed"
    params = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 30000, **(sampling_params or {})}
    if int(params["max_new_tokens"]) < 1:
        raise ValueError("max_new_tokens must be positive.")
    if params.get("stop") is None and params.get("stop_token_ids") is None:
        params["stop"] = ["<|im_end|>"]
    for turn in range(max_turns):
        remaining = max_context_len - len(tokens) - core._context_safety_margin(args)
        if remaining <= 0:
            status = "truncated"
            break
        turn_params = {**params, "max_new_tokens": min(int(params["max_new_tokens"]), remaining)}
        result = client.post("/generate", {
            "input_ids": tokens,
            "sampling_params": turn_params,
            "return_logprob": True,
        })
        metadata = result.get("meta_info", {})
        finish_type = (metadata.get("finish_reason") or {}).get("type", "unknown")
        finish_reasons.append(finish_type)
        if finish_type == "abort":
            status = "aborted"
            break
        rows = metadata.get("output_token_logprobs")
        if not rows:
            raise ValueError("SGLang /generate omitted output_token_logprobs.")
        generated = [row[1] for row in rows]
        if not all(type(token) is int for token in generated):
            raise ValueError("SGLang returned invalid output token IDs.")
        available = max_context_len - len(tokens)
        clipped = len(generated) > available
        generated = generated[:available]
        generated_token_count += len(generated)
        for row in rows[:len(generated)]:
            logprob = row[0]
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                sampled_token_nll_invalid_count += 1
                continue
            sampled_token_nll_count += 1
            sampled_token_nll = (
                sampled_token_nll * ((sampled_token_nll_count - 1) / sampled_token_nll_count)
                - logprob / sampled_token_nll_count
            )
        tokens.extend(generated)
        response_tokens.extend(generated)
        if clipped or finish_type == "length":
            status = "truncated"
            break
        text = client.decode(generated)
        tool_call = core._extract_last_tool_call(text)
        if tool_call is None:
            break
        observation_text, observation = core._safe_execute_tool_call(tool_call, stock_db)
        observations.append({"turn": turn + 1, "call": tool_call, "observation": observation})
        observation_turn = core.render_qwen3_nothink_observation(
            observation_text, assistant_already_closed=core._assistant_already_closed(text)
        )
        observation_tokens = client.encode(observation_turn)
        available = max_context_len - len(tokens)
        clipped = len(observation_tokens) > available
        tokens.extend(observation_tokens[:available])
        response_tokens.extend(observation_tokens[:available])
        if clipped:
            status = "truncated"
            break
    response = client.decode(response_tokens) if response_tokens else ""
    return {
        "target": target,
        "label": {"target": target},
        "prompt": prompt,
        "response": response,
        "response_length": len(response_tokens),
        "generated_token_count": generated_token_count,
        "sampled_token_nll": sampled_token_nll if sampled_token_nll_count else None,
        "sampled_token_nll_count": sampled_token_nll_count,
        "sampled_token_nll_invalid_count": sampled_token_nll_invalid_count,
        "prompt_length": initial_length,
        "status": status,
        "route": extract_answer_steps(response),
        "metadata": {"tool_observations": observations, "finish_reasons": finish_reasons,
                     "final_answer_found": core._has_final_answer(response)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inventory-assisted inference against an SGLang HTTP server.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-context-len", dest="retro_emol_max_context_len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--tool-schema", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--score", action="store_true")
    add_resource_arguments(parser)
    args = parser.parse_args()
    client = None
    try:
        if args.score:
            validate_resources(args)
        prompt, tools = load_prompt_and_tools(args.prompt_template, args.tool_schema)
        tokenizer = load_local_tokenizer(args.tokenizer)
        client = SGLangClient(args.endpoint, tokenizer, timeout=args.timeout, api_key=os.environ.get("SGLANG_API_KEY"))
        result = infer_target(args.target, client, args, prompt_template=prompt, tools=tools, sampling_params={
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
        })
        if args.score:
            result["reward"] = core.compute_retro_emol_reward_with_args(args, SimpleNamespace(**result))
            result["success"] = result["reward"]["success"]
        with atomic_jsonl(args.output) as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps({"status": result["status"], "route": result["route"]}))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
