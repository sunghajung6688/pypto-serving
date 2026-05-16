# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
from examples.model.qwen3_14b.runner.cpu_executor import CpuModelExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU-only Qwen3-14B generation with the reference executor."
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory, e.g. a Hugging Face snapshot.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default="qwen3-14b-cpu-ref")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    kv_cache_manager = KvCacheManager()
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=CpuModelExecutor(kv_cache_manager),
    )
    engine.init_model(
        model_id=args.model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=64,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
    )
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
    )
    if args.stream:
        result = engine.generate(args.model_id, args.prompt, config)
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        result = engine.generate_result(args.model_id, args.prompt, config)
        print(f"text: {result.text}")
        print(f"token_ids: {result.token_ids}")
        print(f"finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    main()
