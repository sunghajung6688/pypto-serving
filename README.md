# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B generation
with PyPTO kernels on Ascend NPUs. It includes a reusable Python runtime,
Qwen3-14B executor glue, CLI entry points, and tests for batching and config
handling.

## Layout

```text
python/
  cli/                         pypto-serving CLI implementation
  core/                        engine, scheduler, KV cache, model loading, async serving
  runtime/                     Simpler worker wrapper for NPU dispatch
pypto-lib/                     submodule providing Qwen3-14B PyPTO kernels
examples/
  pypto-serving                executable CLI wrapper
  model/qwen3_14b/
    cpu_generate.py            CPU reference generation example
    npu_generate.py            NPU generation/profiling example
    npu_serving.json           sample serving config
    runner/                    Qwen3 executors and runner glue
<<<<<<< HEAD
tests/                         CLI and batching tests
=======
    src/                       PyPTO kernel/program builders
tests/                         CLI, batching, E2E serving, and benchmark tests
>>>>>>> b2fca02 (Add serving V2: multiprocess worker + dynamic batching + HTTP API)
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

## HTTP Serving (OpenAI-compatible API)

Start the serving server with multiprocess worker:

```bash
task-submit --device auto --max-time 0 --run \
  "python -m python.cli.main \
    --config examples/model/qwen3_14b/npu_serving.json \
    --serve --port 8899 --device {}"
```

Test with curl:

```bash
# Health check
curl http://localhost:8899/health

# Completion
curl http://localhost:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "temperature": 0.0}'

# Streaming
curl http://localhost:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "stream": true}'

# Chat completion
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 32}'
```

Run the serving benchmark:

```bash
python tests/bench_serving.py --port 8899 --stream -n 8 -c 4 --max-tokens 16
```

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
- HTTP serving mode additionally requires `fastapi`, `uvicorn`, and `pydantic`.
  The benchmark script requires `aiohttp`.
