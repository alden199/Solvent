# Solvent

Long-horizon reasoning for multi-step retrosynthesis, with supervised fine-tuning,
agentic reinforcement learning, inventory feedback, and route evaluation.

The repository includes the project code and modified LLaMA-Factory, Slime, and
Megatron-LM sources. Model weights, datasets, building-block inventories, and
reaction templates must be obtained separately under their respective licenses.
Run the commands below from the repository root.

## Environment

Use separate environments for SFT and RL. Training and model serving require
NVIDIA GPUs and compatible CUDA drivers. Data preparation and offline scoring
can run on CPU.

For data preparation and scoring only, install the core package with Python 3.10
or newer: `python -m pip install -e .`. Parquet input additionally needs `pyarrow`.

### SFT

```bash
python3.10 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sft.txt
python -m pip install --no-deps -e third_party/LLaMA-Factory
python -m pip install -e .
```

Install a matching CUDA toolkit and C++ compiler for DeepSpeed extensions.

### RL and Inference

Start a separate shell or deactivate the SFT environment first.

```bash
python3.12 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install 'torch==2.9.1+cu129' 'torchvision==0.24.1+cu129' 'torchaudio==2.9.1+cu129' --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements/rl.txt
python -m pip install --no-deps -e third_party/slime
python -m pip install --no-deps -e third_party/Megatron-LM
python -m pip install -e .
mkdir -p external
git clone https://github.com/sgl-project/sglang.git external/sglang
git -C external/sglang checkout --detach 24c91001cf99ba642be791e099d358f4dfe955f5
git -C external/sglang apply --check ../../third_party/slime/patches/sglang.patch
git -C external/sglang apply ../../third_party/slime/patches/sglang.patch
python -m pip install --no-deps -e external/sglang/python
python -m pip check
```

Apply the SGLang patch once to the specified revision. Bundled Megatron-LM is
already patched.

## Inventory, Templates and Targets

Create a UTF-8 inventory file with one building-block SMILES per line and no
header. Build the stock database and set the resource paths:

```bash
python scripts/build_stock.py --smiles ./data/building_blocks.smi --output ./data/stock.sqlite
python scripts/build_stock.py --validate ./data/stock.sqlite
export SOLVENT_STOCK_DB="$PWD/data/stock.sqlite"
export SOLVENT_TEMPLATE_FILE="$PWD/data/retro.templates.json.gz"
```

Templates are gzip-compressed JSON records with a `reaction_smarts` string in
`PRODUCT_PATTERN>>PRECURSOR_PATTERNS` direction.

RL target input is JSONL with a `target` SMILES and an optional `route` list of
reaction strings. For example, a target-only record is `{"target":"CCO"}`.

```bash
python scripts/prepare_data.py --input ./data/targets.jsonl --output ./data/prepared.jsonl
```

Use `prepared.jsonl` as `TRAIN_DATA` for RL.

## Inference and Evaluation

Start an SGLang server on an appropriate GPU host:

```bash
python -m sglang.launch_server --model-path ./models/Solvent --port 30000
```

In another terminal with the same environment and resource variables:

```bash
python scripts/infer.py --tokenizer ./models/Solvent --target CCO \
  --endpoint http://127.0.0.1:30000 --output ./outputs/prediction.jsonl
python scripts/evaluate.py --input ./outputs/prediction.jsonl \
  --output ./outputs/scored.jsonl --k 1
```

Set `--tokenizer` to the served model's local directory. The endpoint is the
SGLang server root, not `/v1`.
Inference requires stock; scoring additionally requires templates. Add `--score`
to inference to score immediately. Optional authentication uses `SGLANG_API_KEY`.

Evaluation accepts JSONL with `target` or `label.target` and the full `response`
transcript. Use `--k 1 5` to report Pass@1 and Pass@5 with at least five samples
per target. For bootstrapping, use `--for-curation` as shown below.

## SFT

Register your data in `dataset_info.json` in the directory given by `DATASET_DIR`.
Knowledge examples use `messages` with `role`/`content`; agent examples use
`conversations` with `from`/`value` and a JSON-string `tools` field:

```json
{
  "knowledge": {
    "file_name": "knowledge.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
  },
  "agent": {
    "file_name": "agent.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "tools": "tools"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "function_tag": "function_call", "observation_tag": "observation"}
  }
}
```

Function-call values are JSON strings with `name` and `arguments`. The
`answer_route` argument is a string containing a JSON list of reaction strings.

```bash
export MODEL=./models/pretrained-backbone
export DATASET_DIR=./my-data
export DATASET=knowledge
export OUTPUT_DIR=./outputs/knowledge
export NPROC_PER_NODE=8
bash scripts/train_sft.sh --dry-run knowledge
bash scripts/train_sft.sh knowledge
```

Select `cold_start` for teacher trajectories and `bootstrap` for curated seed
policy trajectories. Update `MODEL`, `DATASET`, and `OUTPUT_DIR` for each stage.
For both cold-start and bootstrap SFT, set `MODEL` to the knowledge-adapted
checkpoint.

Trailing `key=value` arguments override recipe settings. For example, disable
validation with `val_size=0 eval_strategy=no load_best_model_at_end=false`.
Resume with
`resume_from_checkpoint=./outputs/stage/checkpoint-N` and the same output path.

For multiple hosts launch once per host with identical `NNODES`,
`NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`, and shared input/output paths,
and distinct zero-based `NODE_RANK`. Do not wrap this script in another torchrun.

## RL

The supplied conversion and RL presets use Qwen3-8B. Run on GPU hosts without
an existing Ray session:

```bash
export MODEL_PATH="$PWD/models/cold-start"
export MEGATRON_LOAD="$PWD/artifacts/cold-start-megatron"
bash scripts/convert_checkpoint.sh hf-to-megatron \
  --input "$MODEL_PATH" --output "$MEGATRON_LOAD" --gpus 1

export TRAIN_DATA="$PWD/data/prepared.jsonl"
export SAVE_DIR="$PWD/outputs/round1-rl"
export GPU_PER_NODE=8
export NODE_COUNT=1
export NODE_RANK=0
bash scripts/train_rl.sh --preset round1 --dry-run
bash scripts/train_rl.sh --preset round1 --preflight
bash scripts/train_rl.sh --preset round1
```

Set `EVAL_DATA` to prepared validation JSONL if needed. Use `--dry-run` to preview
the command and `--preflight` to check the configuration before training.
`MEGATRON_LOAD` must be the root containing
`latest_checkpointed_iteration.txt`, not its `release/` child. `REF_LOAD` defaults
to that root.

After bootstrap SFT, convert its chosen checkpoint, update the input/output
variables, and use `--preset final`. Override configuration values with
`--override-config ./my-config.yaml`, for example:

```yaml
training:
  tensor_model_parallel_size: 1
  sequence_parallel: false
  num_epoch: null
  num_rollout: 50
```

Set `GPU_PER_NODE` to your allocation. Preserve `rollout_batch_size * n_samples_per_prompt ==
global_batch_size * num_steps_per_rollout` when changing batches. Multi-host
runs require the same `NODE_COUNT`, `GPU_PER_NODE`, reachable `MASTER_ADDR`,
fresh `RUN_ID`, and shared paths, with distinct `NODE_RANK`. Launch once per host.

Export trained weights to a new Hugging Face directory:

```bash
export MODEL_PATH=./models/bootstrap
bash scripts/convert_checkpoint.sh megatron-to-hf \
  --input ./outputs/final-rl --output ./models/Solvent
```

The input can also be a selected iteration directory. `MODEL_PATH` supplies
matching architecture/tokenizer metadata.

## Reasoning Bootstrapping

Save seed-policy trajectories as JSONL with `target`, `response`, and generation
`status`. Score them, then select one trajectory per target:

```bash
python scripts/evaluate.py --input ./data/seed-rollouts.jsonl \
  --output ./data/scored-rollouts.jsonl --for-curation
python scripts/curate_trajectories.py \
  --samples ./data/scored-rollouts.jsonl --sample-labels seed-rollouts \
  --prompt-template solvent/prompts/cotv1_easy_toolonly_prompt.txt \
  --tool-schema solvent/prompts/check_emolecules_stock_tool_schema.json \
  --tokenizer ./models/student --output-dir ./outputs/bootstrapping
```

Use the same prompt, tool schema, and tokenizer as SFT. Register the output
`curated.json` in `dataset_info.json` using the agent ShareGPT schema above, then
train with the `bootstrap` preset. Selection counts are saved in
`curated.report.json`.

## License

Solvent's original code is licensed under the [Apache License 2.0](LICENSE).
For bundled frameworks, see [third-party notices](third_party/NOTICE.md).
