# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B generation
with PyPTO kernels on Ascend NPUs. It includes a reusable Python runtime,
Qwen3-14B executor glue, CLI entry points, and tests for batching and config
handling.

## Layout

```text
python/
  cli/                         pypto-serving CLI implementation
  core/                        engine, scheduler, KV cache, model loading
pypto-lib/                     submodule providing Qwen3-14B PyPTO kernels
examples/
  pypto-serving                executable CLI wrapper
  model/qwen3_14b/
    cpu_generate.py            CPU reference generation example
    npu_generate.py            NPU generation/profiling example
    npu_serving.json           sample interactive serving config
    runner/                    Qwen3 executors and runner glue
tests/                         CLI and batching tests
```

## Quick Checks

Initialize the kernel submodule after cloning:

```bash
git submodule update --init --recursive
```

Run the unit tests:

```bash
python -m pytest tests/test_cli.py tests/test_batching.py
```

Show CLI help:

```bash
./examples/pypto-serving --help
python -m python.cli --help
```

## NPU Generation

One-shot generation, non-L3 path:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5"
```

One-shot generation, L3 path:

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5 \
    --l3"
```

Interactive generation:

```bash
task-submit --run -i \
  "./examples/pypto-serving \
    --config examples/model/qwen3_14b/npu_serving.json \
    --device 0 \
    --interactive"
```

At the `[user]` prompt, enter a prompt such as `Huawei is`; use `/exit` or
`/quit` to leave the interactive session.

## Notes

- The sample config points at `/data/linyifan/models/Qwen3-14B`; edit
  `examples/model/qwen3_14b/npu_serving.json` or pass another config if your
  model path differs.
- `./examples/pypto-serving --device <id>` overrides `npu.device_id` from the
  JSON config for both one-shot and interactive serving.
- Generated kernel artifacts are written under `build_output/` and are ignored
  by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the
  local Ascend runtime environment to be available in the active Python
  environment.
