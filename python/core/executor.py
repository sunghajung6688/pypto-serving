# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import dataclasses
import logging
from abc import ABC, abstractmethod

import torch

from .kv_cache import KvCacheManager
from .types import (
    DecodeBatch,
    DecodeResult,
    GenerateConfig,
    GenerateResult,
    KvAllocation,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RequestState,
    RuntimeConfig,
    RuntimeModel,
)

logger = logging.getLogger(__name__)


class ModelExecutor(ABC):
    """Backend-neutral interface used by ``LLMEngine`` to execute generation."""

    def __init__(self, kv_cache_manager: KvCacheManager) -> None:
        """Store the KV cache manager shared with the engine."""
        self._kv_cache_manager = kv_cache_manager

    @contextlib.contextmanager
    def session(self):
        """Wrap one generation sequence in executor-specific runtime state."""
        yield

    @property
    def profile_verbose(self) -> bool:
        """Return whether model loading and execution should emit stage timings."""
        return False

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embedding rows for ``token_ids`` on the model runtime device."""
        token_ids = token_ids.to(device=model.runtime.device, dtype=torch.long)
        return model.embed_tokens.index_select(0, token_ids.view(-1)).view(
            *token_ids.shape,
            model.config.hidden_size,
        )

    def profile_peak_memory(
        self,
        model: RuntimeModel,
        kv_cache_manager: KvCacheManager,
        model_id: str,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> int:
        """Run a dummy forward pass and return peak device memory in bytes.

        Raises RuntimeError if profiling is not supported (CPU device, etc.).
        """
        device = runtime.device
        if not (device.startswith("npu") or device.startswith("cuda")):
            raise RuntimeError(
                f"Cannot profile peak memory for device '{device}'. "
                f"Profiling requires an NPU or CUDA device. "
                f"Set total_kv_pages explicitly in the config to skip profiling."
            )

        # Reset peak memory stats
        self._reset_peak_memory(device)

        # Allocate a temporary minimal KV pool for profiling (no TurboQuant)
        kv_cache_manager.register_model(model_id, config, dataclasses.replace(
            runtime, total_kv_pages=1, kv_quant_config=None,
        ))

        try:
            batch_size = 1
            seq_len = runtime.page_size
            device_obj = torch.device(device)

            dummy_tokens = torch.zeros(
                (batch_size, seq_len), dtype=torch.long, device=device_obj,
            )
            dummy_embeddings = torch.zeros(
                (batch_size, seq_len, config.hidden_size),
                dtype=model.embed_tokens.dtype, device=device_obj,
            )

            pool = kv_cache_manager._pool(model_id)
            dummy_alloc = KvAllocation(
                request_id="__profile__",
                model_id=model_id,
                page_ids=[0],
                tokens_capacity=pool.page_size,
                tokens_used=0,
            )

            block_table = torch.tensor(
                [[0]], dtype=torch.int32, device=device_obj,
            )
            slot_mapping = torch.arange(
                seq_len, dtype=torch.int32, device=device_obj,
            ).unsqueeze(0)

            with self.session():
                self.run_prefill(model, PrefillBatch(
                    request_ids=["__profile__"],
                    token_ids=dummy_tokens,
                    input_embeddings=dummy_embeddings,
                    seq_lens=torch.tensor(
                        [seq_len], dtype=torch.int32, device=device_obj,
                    ),
                    kv_allocations=[dummy_alloc],
                    positions=dummy_tokens.clone(),
                    block_table=block_table,
                    slot_mapping=slot_mapping,
                ))
        finally:
            # Remove temporary pool
            kv_cache_manager.unregister_model(model_id)

        peak = self._max_memory_allocated(device)
        logger.info(
            "[Profiling] Peak device memory after dummy forward: %.2f GiB "
            "(device=%s)", peak / (1024 ** 3), device,
        )
        return peak

    @staticmethod
    def _reset_peak_memory(device: str) -> None:
        """Reset peak memory tracking on the given device."""
        if device.startswith("npu"):
            torch.npu.reset_peak_memory_stats(device)
        elif device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def _max_memory_allocated(device: str) -> int:
        """Return peak memory in bytes allocated on the given device."""
        if device.startswith("npu"):
            return torch.npu.max_memory_allocated(device)
        elif device.startswith("cuda"):
            return torch.cuda.max_memory_allocated(device)
        return 0

    def validate_generate_batch(
        self,
        record: ModelRecord,
        batch_size: int,
        config: GenerateConfig,
    ) -> None:
        """Validate executor-specific limits before KV allocation begins."""
        return None

    def prompt_allocation_length(
        self,
        record: ModelRecord,
        prompt_len: int,
        config: GenerateConfig,
    ) -> int:
        """Return the initial KV allocation size for one prompt."""
        return prompt_len

    def try_generate_batch(
        self,
        record: ModelRecord,
        requests: list[RequestState],
        prefill_batch: PrefillBatch,
        config: GenerateConfig,
    ) -> list[GenerateResult] | None:
        """Optionally handle generation with an executor-specific fast path."""
        return None

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run prompt prefill and return logits for the next token."""
        raise NotImplementedError

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one decode step for active requests and return next-token logits."""
        raise NotImplementedError
