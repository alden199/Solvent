# LLaMA-Factory SFT source snapshot

This directory contains the complete tracked framework package and its install
entrypoints, vendored from the actual SFT working tree. It is not a new upstream
release. See `PROVENANCE.json` for source revisions, local modifications and
dependency metadata, and `SOURCE_MANIFEST.json` for per-file SHA-256 hashes.

The framework is Apache-2.0 licensed. `LICENSE` and upstream source attribution
notices are retained. Original upstream project:
https://github.com/hiyouga/LLaMA-Factory.

The five locally modified framework files retain answer-only masking, weighted
SFT support and supervised-token entropy diagnostics. The release recipes disable
answer-only and weighted loss. Packaging removes local explanatory comments and
docstrings from these additions, without deleting upstream license notices.

The allowlist contains `src/llamafactory/**/*.py`, `src/train.py`, `src/api.py`,
`src/webui.py`, `setup.py`, `pyproject.toml`, `requirements.txt`, `MANIFEST.in`, and
`LICENSE`. No datasets, checkpoints, Git history, caches, logs, credentials, or
original experiment launch scripts are included. Additional files in this directory
are release documentation, provenance, and CPU-only packaging tests.

Use `../../scripts/train_sft.sh` as documented in `../../docs/sft.md`. The
unmodified upstream CLI remains available, but its distributed argument joining
does not preserve whitespace in paths. The release launcher uses an argument
array and invokes the vendored training entrypoint directly instead.
