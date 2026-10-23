from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .curation import curate_trajectories


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curate one strict-success reasoning trajectory per canonical target.")
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-labels", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="curated")
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--tool-schema", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-response-length", type=int, default=32000)
    parser.add_argument("--max-chat-template-tokens", type=int, default=32000)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_chat_template_tokens > 0 and args.tokenizer is None:
        parser.error("--tokenizer is required unless --max-chat-template-tokens 0 is explicitly set")
    try:
        tokenizer = None
        if args.max_chat_template_tokens > 0:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
            tokenizer.model_max_length = 10**9
        outputs = {
            (args.output_dir / f"{args.prefix}{suffix}").resolve()
            for suffix in (".json", ".selected_cots.jsonl", ".report.json")
        }
        if any(path.resolve() in outputs for path in (args.prompt_template, args.tool_schema)):
            raise ValueError("output must not replace a prompt or tool schema")
        report = curate_trajectories(
            args.samples,
            output_dir=args.output_dir,
            prompt_template=args.prompt_template.read_text(encoding="utf-8"),
            tools_text=args.tool_schema.read_text(encoding="utf-8"),
            tokenizer=tokenizer,
            max_response_length=args.max_response_length,
            max_chat_template_tokens=args.max_chat_template_tokens,
            prefix=args.prefix,
            sample_labels=args.sample_labels,
            work_dir=args.work_dir,
            progress_every=args.progress_every,
            progress=lambda counts: print(json.dumps({"stage": "scan", **counts}), file=sys.stderr, flush=True),
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"curation failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
