from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
SLIME = ROOT / "third_party" / "slime"
MEGATRON_REVISION = "3714d81d418c9f1bca4594fc35f9e8289f652862"
SGLANG_REVISION = "24c91001cf99ba642be791e099d358f4dfe955f5"


def path_env(name, default):
    return str(Path(os.environ.get(name, default)).expanduser().resolve())


def load_config(preset):
    import yaml

    folder = Path(__file__).parent
    with (folder / "common.yaml").open() as handle:
        config = yaml.safe_load(handle)
    with (folder / f"{preset}.yaml").open() as handle:
        override = yaml.safe_load(handle)
    for section, values in override.items():
        config[section].update(values)
    return config


def flags(values):
    result = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def topology():
    nodes = int(os.environ.get("NODE_COUNT", "1"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    gpus = int(os.environ.get("GPU_PER_NODE", "8"))
    if nodes < 1 or gpus < 1 or not 0 <= rank < nodes:
        raise ValueError("Require NODE_COUNT >= 1, GPU_PER_NODE >= 1, and 0 <= NODE_RANK < NODE_COUNT")
    return nodes, rank, gpus


def paths():
    return {
        "model": path_env("MODEL_PATH", ROOT / "artifacts/model"),
        "load": path_env("MEGATRON_LOAD", ROOT / "artifacts/megatron"),
        "save": path_env("SAVE_DIR", ROOT / "outputs/rl"),
        "data": path_env("TRAIN_DATA", ROOT / "data/rl/train.jsonl"),
        "stock": path_env("SOLVENT_STOCK_DB", os.environ.get("STOCK_DB", ROOT / "data/stock.sqlite")),
        "templates": path_env("SOLVENT_TEMPLATE_FILE", os.environ.get("TEMPLATE_FILE", ROOT / "data/templates.json.gz")),
        "megatron": path_env("MEGATRON_PATH", ROOT / "third_party/Megatron-LM"),
        "sglang": path_env("SGLANG_PATH", ROOT / "external/sglang"),
    }


def runtime_environment(locations):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(SLIME), locations["megatron"], str(Path(locations["sglang"]) / "python")]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    environment["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["RAY_USAGE_STATS_ENABLED"] = "0"
    return environment


def training_plan(args):
    locations = paths()
    nodes, rank, gpus = topology()
    config = load_config(args.preset)
    training = config["training"]
    reward = config["reward"]
    override = {}
    if args.override_config:
        import yaml

        with Path(args.override_config).open() as handle:
            override = yaml.safe_load(handle)
        if not isinstance(override, dict) or not set(override) <= {"training", "reward", "model"}:
            raise ValueError("Override YAML must contain only training, reward, and/or model mappings")
        for section, values in override.items():
            if not isinstance(values, dict):
                raise ValueError(f"Override {section} must be a mapping")
    if os.environ.get("TENSOR_MODEL_PARALLEL_SIZE"):
        training["tensor_model_parallel_size"] = int(os.environ["TENSOR_MODEL_PARALLEL_SIZE"])
    if "tensor_model_parallel_size" in override.get("training", {}):
        training["tensor_model_parallel_size"] = override["training"]["tensor_model_parallel_size"]
    tp = training["tensor_model_parallel_size"]
    if type(tp) is not int or tp < 1 or gpus % tp:
        raise ValueError("GPU_PER_NODE must be divisible by TENSOR_MODEL_PARALLEL_SIZE")
    training["sequence_parallel"] = tp > 1
    run_id = os.environ.get("RUN_ID", "dry-run" if args.dry_run or args.preflight else uuid.uuid4().hex)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RUN_ID must contain only letters, digits, underscores, and hyphens")
    run_dir = Path(locations["save"]) / ".launcher" / run_id
    training.update(
        actor_num_nodes=nodes,
        actor_num_gpus_per_node=gpus,
        rollout_num_gpus=nodes * gpus,
        hf_checkpoint=locations["model"],
        load=locations["load"],
        ref_load=path_env("REF_LOAD", locations["load"]),
        save=locations["save"],
        prompt_data=locations["data"],
        custom_config_path=str(run_dir / "reward.yaml"),
    )
    reward.update(retro_emol_stock_db=locations["stock"], retro_emol_template_file=locations["templates"])
    if args.smoke:
        groups = max(1, math.ceil((nodes * gpus // tp) / 2))
        training.pop("num_epoch")
        training.update(
            num_rollout=1,
            rollout_batch_size=groups,
            n_samples_per_prompt=2,
            num_steps_per_rollout=1,
            global_batch_size=groups * 2,
            rollout_max_context_len=4096,
            rollout_max_response_len=1024,
            max_tokens_per_gpu=4096,
            dynamic_sampling_filter_path=None,
            save_interval=1,
        )
        reward.update(
            max_turns=2,
            retro_emol_max_context_len=4096,
            retro_emol_overlong_penalty_enable=False,
            retro_emol_mask_truncated_samples_for_loss=False,
        )
    elif os.environ.get("EVAL_DATA"):
        training.update(
            eval_interval=25,
            eval_prompt_data=["validation", path_env("EVAL_DATA", "")],
            eval_function_path="slime.rollout.sglang_rollout.generate_rollout",
            eval_input_key="prompt",
            eval_label_key="label",
            eval_tool_key="tools",
            n_samples_per_eval_prompt=1,
            eval_max_context_len=training["rollout_max_context_len"],
            eval_max_response_len=training["rollout_max_response_len"],
            eval_temperature=training["rollout_temperature"],
            eval_top_p=training["rollout_top_p"],
            eval_top_k=training["rollout_top_k"],
            eval_reward_key="success",
        )
    if args.override_config:
        for section, values in override.items():
            config[section].update(values)
        print(f"EXPLICIT CONFIG OVERRIDE: {Path(args.override_config).resolve()}", file=sys.stderr)
    tp = training["tensor_model_parallel_size"]
    if tp < 1 or gpus % tp:
        raise ValueError("GPU_PER_NODE must be divisible by tensor_model_parallel_size")
    if tp == 1 and training.get("sequence_parallel"):
        raise ValueError("sequence_parallel must be false with tensor_model_parallel_size=1")
    samples = training["rollout_batch_size"] * training["n_samples_per_prompt"]
    if samples != training["global_batch_size"] * training["num_steps_per_rollout"]:
        raise ValueError("rollout_batch_size * n_samples_per_prompt must equal global_batch_size * num_steps_per_rollout")
    command = [sys.executable, str(SLIME / "train.py"), *flags(training), *flags(config["model"])]
    return locations, config, run_dir, command


def check_source(folder, revision, patch):
    errors = []
    folder = Path(folder)
    if not folder.is_dir():
        return [f"Missing dependency source: {folder}"]
    if folder.resolve() != (ROOT / "third_party/Megatron-LM").resolve():
        result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != revision:
            errors.append(f"{folder.name}: require source revision {revision}")
    fingerprints = json.loads((patch.parent / "source_fingerprints.json").read_text())[patch.stem]
    for relative, expected in fingerprints["files"].items():
        source = folder / relative
        if not source.is_file() or semantic_hash(source.read_text()) != expected:
            errors.append(f"{folder.name}: required patched source differs: {relative}")
    return errors


def semantic_hash(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
            node.body = [child for child in node.body if not isinstance(child, ast.Pass)]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def preflight(locations, config=None, check_dependencies=True):
    errors = []
    required = [Path(locations["model"]) / "config.json", Path(locations["megatron"]) / "megatron/training/arguments.py"]
    if config:
        required += [Path(locations[key]) for key in ("data", "stock", "templates")]
        required += [ROOT / "solvent/rl/generate_with_emol.py", ROOT / "solvent/chem_utils.py"]
        for key in ("load", "ref_load"):
            folder = Path(config["training"][key])
            if folder.name == "release":
                errors.append(f"{key} must be the checkpoint ROOT, not its release/ subdirectory")
            required.append(folder / "latest_checkpointed_iteration.txt")
        if config["training"].get("eval_prompt_data"):
            required.append(Path(config["training"]["eval_prompt_data"][1]))
        if Path(locations["data"]).is_file():
            count = 0
            with Path(locations["data"]).open() as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not {"prompt", "label", "tools"} <= row.keys():
                            raise ValueError("require prompt, label, and tools fields")
                    except (ValueError, TypeError) as error:
                        errors.append(f"TRAIN_DATA line {number}: {error}")
                        break
                    count += 1
            if count < config["training"]["rollout_batch_size"]:
                errors.append("TRAIN_DATA has fewer records than rollout_batch_size")
        if not os.environ.get("EVAL_DATA"):
            print("NOTICE: no EVAL_DATA supplied; periodic validation is disabled, unlike the recorded runs.", file=sys.stderr)
    errors.extend(f"Missing required file: {path}" for path in required if not path.is_file())
    model_config = Path(locations["model"]) / "config.json"
    if model_config.is_file():
        with model_config.open() as handle:
            model = json.load(handle)
        expected = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
                    "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8,
                    "vocab_size": 151936, "head_dim": 128, "rope_theta": 1000000}
        for key, value in expected.items():
            if model.get(key) != value:
                errors.append(f"MODEL_PATH config {key} must be {value!r}, got {model.get(key)!r}")
    if check_dependencies:
        if sys.version_info[:2] != (3, 12):
            errors.append("The pinned RL environment requires Python 3.12")
        for name, expected in {"torch": "2.9.1+cu129", "ray": "2.53.0", "transformers": "4.57.1",
                               "mbridge": "0.15.1", "transformer-engine": "2.10.0", "flash-attn": "2.7.4.post1",
                               "torch-memory-saver": "0.0.9", "sglang-router": "0.3.2"}.items():
            try:
                found = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                found = "not installed"
            if found != expected:
                errors.append(f"{name}: expected {expected}, found {found}")
        if not shutil.which("ray"):
            errors.append("ray executable is not on PATH")
        if not shutil.which("git"):
            errors.append("git executable is not on PATH")
        else:
            errors += check_source(locations["megatron"], MEGATRON_REVISION, SLIME / "patches/megatron.patch")
            errors += check_source(locations["sglang"], SGLANG_REVISION, SLIME / "patches/sglang.patch")
    if errors:
        raise ValueError("Preflight failed:\n" + "\n".join(f"  {error}" for error in errors))
    print("CPU preflight passed. GPU kernels, CUDA memory, network fabric, and model execution are not tested.")


def wait_for(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(2)


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def run_training(args, locations, config, run_dir, command):
    import yaml

    nodes, rank, gpus = topology()
    if nodes > 1 and (not os.environ.get("RUN_ID") or not os.environ.get("MASTER_ADDR")):
        raise ValueError("Multinode runs require identical RUN_ID and MASTER_ADDR on every node")
    timeout = int(os.environ.get("RAY_WAIT_TIMEOUT_SEC", "900"))
    max_runtime = int(os.environ.get("MAX_RUN_SECONDS", "86400" if args.smoke else "604800"))
    head = os.environ.get("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
    node_ip = os.environ.get("NODE_IP", head if rank == 0 else socket.gethostbyname(socket.gethostname()))
    port = int(os.environ.get("RAY_HEAD_PORT", "6379"))
    dashboard_port = int(os.environ.get("RAY_DASHBOARD_PORT", "8265"))
    environment = runtime_environment(locations)
    existing = subprocess.run(["ray", "status", "--address=auto"], env=environment, capture_output=True)
    if existing.returncode == 0:
        raise ValueError("A Ray cluster is already running locally. Use dedicated allocated nodes with no existing Ray session.")
    ready = run_dir / "ready.json"
    done = run_dir / "done.json"
    started = False
    exit_code = 1
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
            with (run_dir / "reward.yaml").open("w") as handle:
                yaml.safe_dump(config["reward"], handle)
            write_status(run_dir / "resolved.json", {"smoke": args.smoke, "preset": args.preset, "config": config, "command": command})
            start = ["ray", "start", "--head", f"--port={port}", f"--dashboard-port={dashboard_port}", "--dashboard-host=127.0.0.1"]
        else:
            wait_for(ready, timeout)
            if done.exists():
                raise ValueError("RUN_ID was already completed; choose a new ID for every launch")
            start = ["ray", "start", f"--address={head}:{port}"]
        start += [f"--node-ip-address={node_ip}", f"--num-gpus={gpus}", "--disable-usage-stats"]
        started = True
        subprocess.run(start, check=True, env=environment, timeout=timeout)
        if rank:
            wait_for(done, max_runtime)
            exit_code = int(json.loads(done.read_text())["exit_code"])
            return exit_code
        write_status(ready, {"head": head, "port": port})
        probe = (
            "import ray,sys,time; ray.init(address=sys.argv[1], logging_level='ERROR'); "
            "deadline=time.monotonic()+int(sys.argv[4]); "
            "exec(\"while True:\\n"
            " nodes=[n for n in ray.nodes() if n['Alive'] and n.get('Resources',{}).get('GPU',0)>=int(sys.argv[3])]\\n"
            " if len(nodes)>=int(sys.argv[2]): break\\n"
            " if time.monotonic()>=deadline: raise TimeoutError('Waiting for GPU nodes')\\n"
            " time.sleep(2)\"); ray.shutdown()"
        )
        subprocess.run([sys.executable, "-c", probe, f"{head}:{port}", str(nodes), str(gpus), str(timeout)],
                       env=environment, check=True, timeout=timeout + 30)
        runtime = {"env_vars": {key: environment[key] for key in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONUNBUFFERED")}}
        submission = ["ray", "job", "submit", f"--address=http://127.0.0.1:{dashboard_port}",
                      "--runtime-env-json", json.dumps(runtime), "--", *command]
        child = subprocess.Popen(submission, env=environment, cwd=ROOT)
        exit_code = child.wait(timeout=max_runtime)
        return exit_code
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if rank == 0 and started and run_dir.is_dir():
            write_status(done, {"exit_code": exit_code})
        if started:
            subprocess.run(["ray", "stop"], env=environment, timeout=90, check=False)


def conversion_plan(args):
    locations = paths()
    model_args = load_config("final")["model"]
    source = Path(args.input).expanduser().resolve()
    target = Path(args.output).expanduser().resolve()
    if source == target:
        raise ValueError("Conversion input and output must be different directories")
    if args.direction == "hf-to-megatron":
        locations["model"] = str(source)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={args.gpus}", str(SLIME / "tools/convert_hf_to_torch_dist.py"),
                   "--hf-checkpoint", str(source), "--save", str(target), "--bf16", "--use-cpu-initialization",
                   *flags(model_args)]
    else:
        if (source / "latest_checkpointed_iteration.txt").is_file():
            iteration = (source / "latest_checkpointed_iteration.txt").read_text().strip()
            if iteration == "release":
                source = source / "release"
            elif iteration.isdigit():
                source = source / f"iter_{int(iteration):07d}"
            else:
                raise ValueError("Invalid Megatron checkpoint tracker")
        command = [sys.executable, str(SLIME / "tools/convert_torch_dist_to_hf.py"),
                   "--input-dir", str(source), "--output-dir", str(target),
                   "--origin-hf-dir", locations["model"], "--vocab-size", "151936"]
    return locations, source, target, command


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portable Solvent RL launchers. Help, dry-run, and preflight never import torch or start Ray.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    train = subparsers.add_parser("train", description="Start one Ray process per allocated node; run this command on every node.")
    train.add_argument("--preset", "--stage", choices=["final", "round1"], default="final")
    train.add_argument("--smoke", action="store_true", help="Explicit non-paper one-rollout smoke settings; requires allocated GPUs")
    train.add_argument("--override-config", type=Path, help="Explicit YAML training/reward/model overrides, applied after preset and smoke settings")
    convert = subparsers.add_parser("convert")
    convert.add_argument("direction", choices=["hf-to-megatron", "megatron-to-hf"])
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--gpus", type=int, default=1)
    for subparser in (train, convert):
        mode = subparser.add_mutually_exclusive_group()
        mode.add_argument("--dry-run", action="store_true", help="Print the fully resolved command without checking artifacts or starting processes")
        mode.add_argument("--preflight", action="store_true", help="Check local files, dependency revisions, patches, and metadata without GPU execution")
    args = parser.parse_args(argv)
    try:
        if args.action == "train":
            locations, config, run_dir, command = training_plan(args)
            print(json.dumps({"preset": args.preset, "smoke": args.smoke, "topology": topology(), "config": config}, indent=2))
            print(shlex.join(command))
            if args.smoke:
                print("SMOKE ONLY: reduced batches/lengths/turns; filtering, truncation masking, overlong penalties, and evaluation disabled.")
            if args.dry_run:
                return 0
            preflight(locations, config)
            if args.preflight:
                return 0
            return run_training(args, locations, config, run_dir, command)
        if args.gpus < 1 or args.gpus > 36:
            raise ValueError("Conversion --gpus must be between 1 and 36")
        locations, source, target, command = conversion_plan(args)
        print(shlex.join(command))
        if args.dry_run:
            return 0
        if target.exists():
            raise ValueError("Refusing to overwrite conversion output; choose a new directory")
        if args.direction == "megatron-to-hf" and not (source / "common.pt").is_file():
            raise ValueError("Input must contain common.pt or be a checkpoint root with a tracker")
        preflight(locations)
        if args.preflight:
            return 0
        return subprocess.run(command, env=runtime_environment(locations), cwd=ROOT).returncode
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
