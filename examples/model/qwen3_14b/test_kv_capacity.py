"""Test KV cache capacity: find the max prompt length before OOM, with and without TurboQuant.

Usage:
    # Step 1: Find max prompt length without TurboQuant
    python examples/model/qwen3_14b/test_kv_capacity.py \
        --model-dir /data/linyifan/models/Qwen3-14B --platform a2a3

    # Step 2: Try with TurboQuant (adjust --start-len to the failing length)
    python examples/model/qwen3_14b/test_kv_capacity.py \
        --model-dir /data/linyifan/models/Qwen3-14B --platform a2a3 \
        --kv-quant --start-len 400 --step 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PTO2_RING_HEAP", str(2 * 1024 ** 3))


def _bootstrap_package_root() -> None:
    this_file = Path(__file__).resolve()
    for candidate in (this_file, *this_file.parents):
        if (candidate / "python" / "core").is_dir() and (candidate / "examples" / "model" / "qwen3_14b" / "runner").is_dir():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return
    raise RuntimeError(f"Unable to locate the pypto-serving repo root from {this_file}")


_bootstrap_package_root()

from python.core import GenerateConfig, LLMEngine, RuntimeConfig
from python.core.kv_cache import KvCacheManager
from examples.model.qwen3_14b.runner.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor
from python.core.types import KvQuantConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test KV cache capacity limits.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", default="qwen3-14b-local")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Controls KV cache pool size. Smaller = fewer pages = OOM sooner.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Smaller = fewer pages = OOM sooner.")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="Only generate 1 token per test, we care about prompt length.")
    parser.add_argument("--start-len", type=int, default=32,
                        help="Starting prompt token count.")
    parser.add_argument("--step", type=int, default=32,
                        help="Increase prompt tokens by this much each iteration.")
    parser.add_argument("--max-len", type=int, default=4096,
                        help="Stop testing at this prompt length.")
    parser.add_argument("--kv-quant", action="store_true", help="Enable TurboQuant.")
    parser.add_argument("--kv-key-bits", type=int, default=4)
    parser.add_argument("--kv-value-bits", type=int, default=2)
    parser.add_argument("--kv-residual-window", type=int, default=128)
    parser.add_argument("--kv-protected-layers", type=int, default=4)
    parser.add_argument("--kv-protected-bits", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()

    kv_quant_config = KvQuantConfig(
        enabled=args.kv_quant,
        key_bits=args.kv_key_bits,
        value_bits=args.kv_value_bits,
        residual_window=args.kv_residual_window,
        protected_layers=args.kv_protected_layers,
        protected_bits=args.kv_protected_bits,
    ) if args.kv_quant else None

    kv_cache_manager = KvCacheManager()
    executor = PyptoExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_id=args.device_id,
        save_kernels_dir=None,
        l3_mode=False,
        l3_trace=False,
    )
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )

    print(f"[test] Initializing model (max_seq_len={args.max_seq_len}, max_batch={args.max_batch_size}) ...", flush=True)
    print(f"[test] TurboQuant: {'ENABLED' if kv_quant_config else 'DISABLED'}", flush=True)

    engine.init_model(
        model_id=args.model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=256,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
            kv_quant_config=kv_quant_config,
        ),
    )

    # Print KV cache pool info
    pool = kv_cache_manager._pool(args.model_id)
    total_pages = len(pool.free_pages) + (pool.key_pages.shape[1] - len(pool.free_pages))
    tokens_per_page = pool.page_size
    max_tokens = total_pages * tokens_per_page
    print(f"[test] KV cache: {total_pages} pages, {tokens_per_page} tokens/page, max {max_tokens} tokens total", flush=True)

    # Generate a long dummy text for padding prompts
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200

    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    prompt_len = args.start_len
    last_success = 0

    while prompt_len <= args.max_len:
        # Build a prompt of approximately prompt_len tokens by repeating dummy text
        # Rough: 1 token ≈ 4 chars for English
        char_count = prompt_len * 4
        prompt = (dummy_text * ((char_count // len(dummy_text)) + 1))[:char_count]

        try:
            print(f"[test] Trying prompt_len ≈ {prompt_len} tokens (str len={len(prompt)}) ... ", end="", flush=True)
            result = engine.generate_result(args.model_id, prompt, config)
            actual_tokens = len(result.token_ids)
            print(f"OK (generated {actual_tokens} tokens)", flush=True)
            last_success = prompt_len
            prompt_len += args.step
        except RuntimeError as e:
            if "Insufficient KV cache capacity" in str(e):
                print(f"OOM! {e}", flush=True)
                print(f"\n[test] Result: max prompt length = {last_success} tokens", flush=True)
                print(f"[test] Failed at: {prompt_len} tokens", flush=True)
                break
            else:
                raise
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            raise
    else:
        print(f"\n[test] Reached max test length {args.max_len} without OOM", flush=True)
        print(f"[test] Last successful: {last_success} tokens", flush=True)

    # Print TurboQuant stats if enabled
    if args.kv_quant:
        stats = kv_cache_manager.quantization_stats(args.model_id)
        bf16_bytes = pool.key_pages.numel() * pool.key_pages.element_size() + pool.value_pages.numel() * pool.value_pages.element_size()
        compressed_bytes = stats.get("compressed_bytes", 0)
        print(f"\n[TurboQuant] bf16 cache: {bf16_bytes / 1024**2:.1f} MiB", flush=True)
        print(f"[TurboQuant] compressed data: {compressed_bytes / 1024**2:.3f} MiB", flush=True)


if __name__ == "__main__":
    main()
