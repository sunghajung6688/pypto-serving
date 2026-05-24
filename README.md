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
    src/                       PyPTO kernel/program builders
tests/                         CLI, batching, E2E serving, and benchmark tests
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

## Offline Mode

### One-shot Generation

```bash
# non-L3 path
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5"
```

```bash
# L3 path
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

### Interactive Generation

```bash
task-submit --run -i \
  "./examples/pypto-serving \
    --config examples/model/qwen3_14b/npu_serving.json \
    --device 0 \
    --interactive"
```

At the `[user]` prompt, enter a prompt such as `Huawei is`; use `/exit` or
`/quit` to leave the interactive session.

### Enabling TurboQuant (Offline)

```bash
task-submit --device auto --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
  python examples/model/qwen3_14b/npu_generate.py \
    --model-dir /data/linyifan/models/Qwen3-14B \
    --prompt 'Huawei is' \
    --platform a2a3 \
    --max-seq-len 512 \
    --max-new-tokens 5 \
    --kv-quant"
```

Customizable TurboQuant parameters:

| Flag | Default | Description |
|---|---|---|
| `--kv-quant-key-bits` | 4 | Key quantization bits |
| `--kv-quant-value-bits` | 2 | Value quantization bits |
| `--kv-quant-residual-window` | 128 | Residual quantization window size |
| `--kv-quant-protected-layers` | 4 | Number of protected (unquantized) layers from the bottom |
| `--kv-quant-protected-bits` | 8 | Bit width for protected layers |

## Online Mode (HTTP Serving)

### Start the Server (OpenAI-compatible API)

```bash
task-submit --device auto --max-time 0 --run \
  "python -m python.cli.main \
    --config examples/model/qwen3_14b/npu_serving.json \
    --serve --port 8899 --device {}"
```

### Test Requests

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

### Benchmark

```bash
python tests/bench_serving.py --port 8899 --stream -n 8 -c 4 --max-tokens 16
```

### Enabling TurboQuant (Online)

Add a `kv_quant` section to your config JSON (same as offline mode):

```json
{
  "model": { ... },
  "runtime": { ... },
  "kv_quant": {
    "enabled": true,
    "key_bits": 4,
    "value_bits": 2,
    "residual_window": 128
  }
}
```

When enabled, `[TurboQuant]` log lines will appear in the worker output,
indicating KV cache compression is active.

## Common CLI Flags

| Flag | Description |
|---|---|
| `--device <id>` | Override NPU device ID from config |
| `--max-num-running-reqs <n>` | Max concurrent running requests |
| `--long-prefill-token-threshold <n>` | Token threshold for chunked prefill |
| `--disable-prefix-cache` | Disable prefix caching |
| `--disable-chunk-prefill` | Disable chunked prefill |

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
