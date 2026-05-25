# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

from .types import KvAllocation, ModelConfig, RuntimeConfig

# Lazy import to avoid hard dependency when quantization is disabled.
_KVCOMPRESSOR = None


def _get_kv_compressor_cls():
    global _KVCOMPRESSOR
    if _KVCOMPRESSOR is None:
        from .turboquant.compressor import KVCompressor
        _KVCOMPRESSOR = KVCompressor
    return _KVCOMPRESSOR


@dataclass
class _CompressedSegment:
    """Compressed KV data for a contiguous range of old tokens in one layer."""

    token_start: int
    token_count: int
    compressed_k: dict
    compressed_v: dict


def compute_num_kv_pages(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_dtype: torch.dtype,
    device: str,
    gpu_memory_utilization: float = 0.9,
    peak_memory_bytes: int = 0,
) -> int:
    """Calculate the number of KV cache pages that fit in available device memory.

    Uses peak_memory_bytes (from dummy forward pass profiling) to account for
    both weight memory and activation memory.

    Raises RuntimeError if the device is not supported or memory info is unavailable.
    """
    if device.startswith("npu"):
        _, total_mem = torch.npu.mem_get_info(device)
    elif device.startswith("cuda"):
        _, total_mem = torch.cuda.mem_get_info(device)
    else:
        raise RuntimeError(
            f"Cannot compute KV cache pages for device '{device}'. "
            f"Set total_kv_pages explicitly in the config."
        )

    target = int(total_mem * gpu_memory_utilization)
    available = target - peak_memory_bytes
    if available <= 0:
        raise RuntimeError(
            f"Not enough device memory for KV cache: "
            f"target={target / 1024**3:.2f} GiB, "
            f"peak_memory={peak_memory_bytes / 1024**3:.2f} GiB, "
            f"available={available / 1024**3:.2f} GiB. "
            f"Reduce gpu_memory_utilization or set total_kv_pages explicitly."
        )

    dtype_size = torch.tensor([], dtype=kv_dtype).element_size()
    page_size_bytes = 2 * page_size * num_kv_heads * head_dim * dtype_size
    num_pages = max(available // (page_size_bytes * num_layers), 0)
    if num_pages == 0:
        raise RuntimeError(
            f"Available memory ({available / 1024**3:.2f} GiB) is too small "
            f"for even one KV cache page ({page_size_bytes * num_layers / 1024**2:.2f} MiB/page)."
        )
    return num_pages


@dataclass
class _CachePool:
    """Paged KV cache storage for one registered model."""

    page_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_blocks_per_seq: int
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    free_pages: list[int]
    # KV quantization (None when disabled)
    kv_compressor: object = None  # KVCompressor | None
    kv_quant_config: object = None  # KvQuantConfig | None
    # Compressed segments: {request_id: {layer_idx: _CompressedSegment}}
    compressed_segments: dict = None


class KvCacheManager:
    """Allocate and materialize paged KV cache for generation requests."""

    def __init__(self) -> None:
        """Create an empty registry of model-specific KV pools."""
        self._pools: dict[str, _CachePool] = {}

    def register_model(
        self,
        model_id: str,
        config: ModelConfig,
        runtime: RuntimeConfig,
        peak_memory_bytes: int = 0,
    ) -> None:
        """Create the KV page pool for a model if it is not already registered."""
        if model_id in self._pools:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        kv_dtype = getattr(torch, runtime.kv_dtype)
        num_pages = runtime.total_kv_pages
        if num_pages is None:
            num_pages = compute_num_kv_pages(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=runtime.page_size,
                kv_dtype=kv_dtype,
                device=runtime.device,
                gpu_memory_utilization=runtime.gpu_memory_utilization,
                peak_memory_bytes=peak_memory_bytes,
            )
            logger.info(
                "[KV Cache] Dynamic allocation from profiled peak memory: %d pages "
                "(peak_memory=%.2f GiB, gpu_memory_utilization=%.2f)",
                num_pages, peak_memory_bytes / (1024 ** 3),
                runtime.gpu_memory_utilization,
            )
        key_pages = torch.zeros(
            config.num_hidden_layers,
            num_pages,
            config.num_key_value_heads,
            runtime.page_size,
            config.head_dim,
            dtype=kv_dtype,
            device=runtime.device,
        )
        value_pages = torch.zeros_like(key_pages)
        # Initialize KV quantization if configured
        kv_compressor = None
        kv_quant_config = runtime.kv_quant_config
        compressed_segments = None
        if kv_quant_config is not None and kv_quant_config.enabled:
            print(f"[TurboQuant] Creating KVCompressor for model {model_id} ...", flush=True)
            KVCompressor = _get_kv_compressor_cls()
            kv_compressor = KVCompressor(
                head_dim=config.head_dim,
                num_layers=config.num_hidden_layers,
                config=kv_quant_config,
                device=runtime.device,
            )
            print(f"[TurboQuant] KVCompressor for model {model_id} created", flush=True)
            compressed_segments = {}

        self._pools[model_id] = _CachePool(
            page_size=runtime.page_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_blocks_per_seq=max_blocks_per_seq,
            key_pages=key_pages,
            value_pages=value_pages,
            free_pages=list(range(num_pages - 1, -1, -1)),
            kv_compressor=kv_compressor,
            kv_quant_config=kv_quant_config,
            compressed_segments=compressed_segments,
        )

        kv_bytes = key_pages.numel() * key_pages.element_size() * 2  # key + value
        per_page_bytes = 2 * config.num_key_value_heads * runtime.page_size * config.head_dim * key_pages.element_size()
        logger.info(
            "[KV Cache] Allocated KV cache: %d pages, %.2f GiB "
            "(per_page=%d bytes, %d layers)",
            num_pages, kv_bytes / (1024 ** 3),
            per_page_bytes, config.num_hidden_layers,
        )

    def unregister_model(self, model_id: str) -> None:
        """Remove a model's KV pool. Used for cleaning up profiling pools."""
        self._pools.pop(model_id, None)

    def allocate_for_prompt(self, model_id: str, request_id: str, prompt_len: int) -> KvAllocation:
        """Allocate enough KV pages to store a prompt of ``prompt_len`` tokens."""
        pool = self._pool(model_id)
        num_pages = max(1, math.ceil(prompt_len / pool.page_size))
        page_ids = self._take_pages(pool, num_pages)
        return KvAllocation(
            request_id=request_id,
            model_id=model_id,
            page_ids=page_ids,
            tokens_capacity=len(page_ids) * pool.page_size,
            tokens_used=0,
        )

    def allocate_with_page_ids(
        self, model_id: str, request_id: str, page_ids: list[int], tokens_used: int = 0
    ) -> KvAllocation:
        """Create a KvAllocation using externally-assigned page IDs from the scheduler."""
        pool = self._pool(model_id)
        return KvAllocation(
            request_id=request_id,
            model_id=model_id,
            page_ids=list(page_ids),
            tokens_capacity=len(page_ids) * pool.page_size,
            tokens_used=tokens_used,
        )

    def ensure_one_more_slot(self, alloc: KvAllocation) -> int:
        """Ensure a request has capacity for one more token and return its slot."""
        pool = self._pool(alloc.model_id)
        if alloc.tokens_used >= alloc.tokens_capacity:
            alloc.page_ids.extend(self._take_pages(pool, 1))
            alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
        return self.slot_mapping_for_request(alloc, alloc.tokens_used)

    def block_table_for_request(self, alloc: KvAllocation) -> torch.Tensor:
        """Return the page IDs for one request as an int32 tensor."""
        return torch.tensor(alloc.page_ids, dtype=torch.int32)

    def block_table_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return a dense ``[batch, max_blocks]`` block table for requests."""
        max_blocks = max((len(alloc.page_ids) for alloc in allocations), default=0)
        table = torch.full((len(allocations), max_blocks), -1, dtype=torch.int32)
        for row, alloc in enumerate(allocations):
            if alloc.page_ids:
                table[row, : len(alloc.page_ids)] = torch.tensor(alloc.page_ids, dtype=torch.int32)
        return table

    def block_table_for_batch_padded(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return a flat ``[B * max_blocks_per_seq]`` block table, -1 padded.

        This matches the paged-attention layout the bundled fused decode kernel
        expects: row ``b`` occupies ``[b * max_blocks_per_seq, (b+1) * max_blocks_per_seq)``,
        with unused trailing slots set to -1.
        """
        if not allocations:
            return torch.empty((0,), dtype=torch.int32)
        pool = self._pool(allocations[0].model_id)
        max_blocks = pool.max_blocks_per_seq
        table = torch.full((len(allocations) * max_blocks,), -1, dtype=torch.int32)
        for row, alloc in enumerate(allocations):
            if alloc.page_ids:
                row_start = row * max_blocks
                table[row_start : row_start + len(alloc.page_ids)] = torch.tensor(
                    alloc.page_ids, dtype=torch.int32,
                )
        return table

    def slot_mapping_for_request(self, alloc: KvAllocation, token_index: int | None = None) -> int:
        """Return the physical slot index for a request token."""
        pool = self._pool(alloc.model_id)
        logical_index = alloc.tokens_used if token_index is None else token_index
        page_idx = logical_index // pool.page_size
        offset = logical_index % pool.page_size
        return alloc.page_ids[page_idx] * pool.page_size + offset

    def slot_mapping_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return current decode slot mappings for a batch."""
        return torch.tensor(
            [self.slot_mapping_for_request(alloc) for alloc in allocations],
            dtype=torch.int32,
        )

    def slot_mapping_for_positions(self, alloc: KvAllocation, num_tokens: int, *, max_tokens: int | None = None) -> torch.Tensor:
        """Return per-position slot mappings, optionally padded with -1."""
        size = num_tokens if max_tokens is None else max_tokens
        mapping = torch.full((size,), -1, dtype=torch.int32)
        for token_index in range(num_tokens):
            mapping[token_index] = self.slot_mapping_for_request(alloc, token_index)
        return mapping

    def write_tokens(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write key/value rows for consecutive tokens into paged cache."""
        pool = self._pool(alloc.model_id)
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        for row in range(keys.shape[0]):
            token_index = start_token_index + row
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            pool.key_pages[layer_idx, physical_page, :, offset, :] = keys[row]
            pool.value_pages[layer_idx, physical_page, :, offset, :] = values[row]
        alloc.tokens_used = max(alloc.tokens_used, start_token_index + keys.shape[0])

    def ingest_prefill_cache(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        keys_flat: torch.Tensor,
        values_flat: torch.Tensor,
        *,
        max_seq: int,
        seq_len: int,
    ) -> None:
        """Import flattened prefill K/V tensors into the paged cache."""
        pool = self._pool(alloc.model_id)
        keys = keys_flat.view(pool.num_kv_heads, max_seq, pool.head_dim)[:, :seq_len, :].permute(1, 0, 2).contiguous()
        values = values_flat.view(pool.num_kv_heads, max_seq, pool.head_dim)[:, :seq_len, :].permute(1, 0, 2).contiguous()
        self.write_tokens(layer_idx, alloc, 0, keys, values)

    def read_context(self, layer_idx: int, alloc: KvAllocation, upto_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read contiguous K/V context for one request and layer."""
        pool = self._pool(alloc.model_id)
        token_count = alloc.tokens_used if upto_tokens is None else upto_tokens
        keys = torch.empty(
            token_count,
            pool.num_kv_heads,
            pool.head_dim,
            dtype=pool.key_pages.dtype,
            device=pool.key_pages.device,
        )
        values = torch.empty_like(keys)
        for token_index in range(token_count):
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            keys[token_index] = pool.key_pages[layer_idx, physical_page, :, offset, :]
            values[token_index] = pool.value_pages[layer_idx, physical_page, :, offset, :]
        return keys, values

    def materialize_single_layer_cache(self, model_id: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened K/V cache views for exactly one model layer.

        The returned tensors are zero-copy views over the selected layer of
        the paged cache, shaped ``[num_pages * num_kv_heads * page_size,
        head_dim]``. Use this API for kernels that receive one layer's cache
        at a time.
        """
        pool = self._pool(model_id)
        return (
            pool.key_pages[layer_idx].reshape(-1, pool.head_dim),
            pool.value_pages[layer_idx].reshape(-1, pool.head_dim),
        )

    def materialize_full_layer_cache(self, model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened K/V cache views stacked across every model layer.

        Use this API for fused or L3 decode kernels that select layer i via
        an arithmetic offset (layer_idx * cache_rows_per_layer) on a single
        cache tensor. The pool is already laid out as
        ``[num_layers, num_pages, num_kv_heads, page_size, head_dim]`` so the
        flat view is zero-copy.

        Returns:
            (key_cache_all, value_cache_all) each shaped
            [num_layers * num_pages * num_kv_heads * page_size, head_dim].
        """
        pool = self._pool(model_id)
        return (
            pool.key_pages.reshape(-1, pool.head_dim),
            pool.value_pages.reshape(-1, pool.head_dim),
        )

    def free(self, alloc: KvAllocation) -> None:
        """Return an allocation's pages to the model pool."""
        pool = self._pool(alloc.model_id)
        pool.free_pages.extend(alloc.page_ids)
        alloc.page_ids.clear()
        alloc.tokens_capacity = 0
        alloc.tokens_used = 0
        # Also clean up compressed segments for this request
        if pool.compressed_segments is not None:
            pool.compressed_segments.pop(alloc.request_id, None)

    # ── KV cache quantization ──

    def is_quantization_enabled(self, model_id: str) -> bool:
        """Return whether KV cache quantization is enabled for a model."""
        pool = self._pool(model_id)
        return pool.kv_compressor is not None

    def clear_compressed_segments(self, model_id: str, request_id: str) -> None:
        """Clear compressed segments for a request without freeing its pages."""
        pool = self._pool(model_id)
        if pool.compressed_segments is not None:
            pool.compressed_segments.pop(request_id, None)

    def restore_compressed_tokens(self, model_id: str, alloc: KvAllocation) -> None:
        """Decompress old tokens back into bf16 cache pages before kernel execution."""
        pool = self._pool(model_id)
        if pool.compressed_segments is None or pool.kv_compressor is None:
            return
        req_segments = pool.compressed_segments.get(alloc.request_id)
        if not req_segments:
            return

        print(f"[TurboQuant] Restoring compressed tokens for request {alloc.request_id}, {len(req_segments)} layers", flush=True)
        for layer_idx, segment in req_segments.items():
            keys, values = pool.kv_compressor.decompress_layer(
                layer_idx, segment.compressed_k, segment.compressed_v,
            )
            self._write_tokens_from_compressed(
                pool, layer_idx, alloc, segment.token_start, keys, values,
            )
        print(f"[TurboQuant] Restore complete, {alloc.tokens_used} tokens", flush=True)

    def compress_old_tokens(self, model_id: str, alloc: KvAllocation) -> None:
        """Compress old tokens (beyond residual_window) and store compressed data."""
        pool = self._pool(model_id)
        if pool.kv_compressor is None or pool.kv_quant_config is None:
            return

        residual_window = pool.kv_quant_config.residual_window
        tokens_used = alloc.tokens_used
        if tokens_used <= residual_window:
            return

        old_token_count = tokens_used - residual_window
        print(f"[TurboQuant] Compressing old tokens: total={tokens_used}, window={residual_window}, compressing={old_token_count} old tokens", flush=True)

        if pool.compressed_segments is None:
            pool.compressed_segments = {}
        if alloc.request_id not in pool.compressed_segments:
            pool.compressed_segments[alloc.request_id] = {}

        for layer_idx in range(pool.num_layers):
            old_keys, old_values = self.read_context(
                layer_idx, alloc, upto_tokens=old_token_count,
            )
            compressed_k, compressed_v = pool.kv_compressor.compress_layer(
                layer_idx, old_keys, old_values,
            )
            pool.compressed_segments[alloc.request_id][layer_idx] = _CompressedSegment(
                token_start=0,
                token_count=old_token_count,
                compressed_k=compressed_k,
                compressed_v=compressed_v,
            )

    def _write_tokens_from_compressed(
        self,
        pool: _CachePool,
        layer_idx: int,
        alloc: KvAllocation,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write decompressed tokens back into paged cache."""
        cache_dtype = pool.key_pages.dtype
        for row in range(keys.shape[0]):
            token_index = start_token_index + row
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            pool.key_pages[layer_idx, physical_page, :, offset, :] = keys[row].to(cache_dtype)
            pool.value_pages[layer_idx, physical_page, :, offset, :] = values[row].to(cache_dtype)

    def print_kv_cache_memory(self, model_id: str) -> None:
        """Calculate and print KV cache memory usage for a model."""
        pool = self._pool(model_id)

        single_tensor_bytes = pool.key_pages.numel() * pool.key_pages.element_size()
        total_kv_bytes = single_tensor_bytes * 2

        dtype_size = pool.key_pages.element_size()
        per_token_bytes = 2 * pool.num_layers * pool.num_kv_heads * pool.head_dim * dtype_size

        total_pages = pool.key_pages.shape[1]
        free_pages = len(pool.free_pages)
        used_pages = total_pages - free_pages
        max_seq_len = pool.max_blocks_per_seq * pool.page_size

        def _fmt(b: int) -> str:
            if b >= 1024 ** 3:
                return f"{b / 1024 ** 3:.2f} GiB"
            return f"{b / 1024 ** 2:.2f} MiB"

        print(f"[KV Cache] model_id = {model_id}", flush=True)
        print(f"[KV Cache] dtype = {pool.key_pages.dtype}, element_size = {dtype_size} bytes", flush=True)
        print(f"[KV Cache] shape per tensor = {list(pool.key_pages.shape)} "
              f"[layers={pool.num_layers}, pages={total_pages}, "
              f"kv_heads={pool.num_kv_heads}, page_size={pool.page_size}, head_dim={pool.head_dim}]", flush=True)
        print(f"[KV Cache] key_pages   = {_fmt(single_tensor_bytes)}", flush=True)
        print(f"[KV Cache] value_pages = {_fmt(single_tensor_bytes)}", flush=True)
        print(f"[KV Cache] total KV cache memory = {_fmt(total_kv_bytes)}", flush=True)
        print(f"[KV Cache] per-token memory = {per_token_bytes} bytes ({per_token_bytes / 1024:.2f} KiB)", flush=True)
        print(f"[KV Cache] pages: total={total_pages}, free={free_pages}, used={used_pages}", flush=True)
        print(f"[KV Cache] max_seq_len per request = {max_seq_len} tokens "
              f"(max_blocks_per_seq={pool.max_blocks_per_seq}, page_size={pool.page_size})", flush=True)

        if pool.compressed_segments is not None:
            stats = self.quantization_stats(model_id)
            compressed_bytes = stats.get("compressed_bytes", 0)
            print(f"[KV Cache] TurboQuant compressed data = {_fmt(compressed_bytes)}", flush=True)
            if total_kv_bytes > 0:
                print(f"[KV Cache] effective memory (bf16 + compressed) = {_fmt(total_kv_bytes + compressed_bytes)}", flush=True)

    def quantization_stats(self, model_id: str) -> dict:
        """Return memory usage stats for compressed KV cache."""
        pool = self._pool(model_id)
        if pool.compressed_segments is None:
            return {"enabled": False}
        total_compressed_bytes = 0
        total_segments = 0
        for req_segments in pool.compressed_segments.values():
            for segment in req_segments.values():
                for tensor in segment.compressed_k.values():
                    if isinstance(tensor, torch.Tensor):
                        total_compressed_bytes += tensor.numel() * tensor.element_size()
                for tensor in segment.compressed_v.values():
                    if isinstance(tensor, torch.Tensor):
                        total_compressed_bytes += tensor.numel() * tensor.element_size()
                total_segments += 1
        return {
            "enabled": True,
            "total_segments": total_segments,
            "compressed_bytes": total_compressed_bytes,
        }

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]

    @staticmethod
    def _take_pages(pool: _CachePool, num_pages: int) -> list[int]:
        """Remove and return free page IDs from a pool."""
        if len(pool.free_pages) < num_pages:
            raise RuntimeError(
                f"Insufficient KV cache capacity: requested {num_pages} pages, only {len(pool.free_pages)} available."
            )
        page_ids = pool.free_pages[-num_pages:]
        del pool.free_pages[-num_pages:]
        return page_ids
