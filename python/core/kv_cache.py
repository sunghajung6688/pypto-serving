# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

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
    freed_page_ids: list[int] = field(default_factory=list)


NONE_HASH = hash(("__none__",))


def hash_block_tokens(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """Return a chained prefix-cache hash for one full token block."""
    return hash((parent_hash, token_ids))


@dataclass(slots=True)
class KVCacheBlock:
    """Metadata for one physical KV cache page/block."""

    block_id: int
    ref_cnt: int = 0
    block_hash: int | None = None
    prev_free: "KVCacheBlock | None" = field(default=None, repr=False)
    next_free: "KVCacheBlock | None" = field(default=None, repr=False)


@dataclass(frozen=True)
class KVCacheBlocks:
    """Scheduler-facing KV blocks grouped by cache group."""

    blocks: tuple[list[KVCacheBlock], ...]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple([block.block_id for block in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        if len(self.blocks) != 1:
            raise ValueError("get_unhashed_block_ids requires one KV cache group")
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]


class FreeKVCacheBlockQueue:
    """Doubly-linked free block queue in eviction order."""

    def __init__(self) -> None:
        self.head: KVCacheBlock | None = None
        self.tail: KVCacheBlock | None = None
        self.count: int = 0

    def append(self, block: KVCacheBlock) -> None:
        block.prev_free = self.tail
        block.next_free = None
        if self.tail is not None:
            self.tail.next_free = block
        else:
            self.head = block
        self.tail = block
        self.count += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.append(block)

    def popleft(self) -> KVCacheBlock | None:
        if self.head is None:
            return None
        block = self.head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        if block != self.head and block != self.tail and block.prev_free is None and block.next_free is None:
            return
        prev_b = block.prev_free
        next_b = block.next_free
        if prev_b is not None:
            prev_b.next_free = next_b
        else:
            self.head = next_b
        if next_b is not None:
            next_b.prev_free = prev_b
        else:
            self.tail = prev_b
        block.prev_free = None
        block.next_free = None
        self.count -= 1

    def __len__(self) -> int:
        return self.count


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


class KvCacheManager:
    """Unified KV block metadata and paged KV tensor storage manager."""

    def __init__(
        self,
        *,
        num_blocks: int | None = None,
        block_size: int = 64,
        enable_prefix_cache: bool = True,
    ) -> None:
        """Create an empty registry of model-specific KV pools."""
        self._pools: dict[str, _CachePool] = {}
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[KVCacheBlock] = []
        self.free_queue = FreeKVCacheBlockQueue()
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        if num_blocks is not None:
            self._init_blocks(num_blocks, block_size)

    @property
    def num_free_blocks(self) -> int:
        """Return the number of immediately allocatable KV blocks."""
        return self.free_queue.count

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical KV blocks."""
        return len(self.blocks)

    def _init_blocks(self, num_blocks: int, block_size: int) -> None:
        if self.blocks:
            if len(self.blocks) != num_blocks or self.block_size != block_size:
                raise ValueError("KV block pool is already initialized with different dimensions")
            return
        self.block_size = block_size
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        for block in self.blocks:
            self.free_queue.append(block)

    def register_model(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """Create the KV page pool for a model if it is not already registered."""
        if model_id in self._pools:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        # Determine number of bf16 pages.  When TurboQuant is enabled we only
        # need a small pool: one resident segment per batch slot (for the
        # residual window) plus enough workspace pages for one full-length
        # sequence (used temporarily during decode to hold decompressed old
        # tokens).
        #
        # NOTE: The workspace is shared across batch slots — it is only needed
        # during the restore→kernel→compress cycle.  For batch=1 this formula
        # is exact.  For batch>1, the caller must serialize the restore/compress
        # per request so that workspace pages are reused across slots, or set
        # total_kv_pages explicitly to cover simultaneous workspace demand.
        kv_quant_config = runtime.kv_quant_config
        num_pages = runtime.total_kv_pages
        if num_pages is None and kv_quant_config is not None and kv_quant_config.enabled:
            # NOTE: The fused NPU decode kernel has the per-layer cache stride
            # (num_pages * num_kv_heads * page_size) baked in at compile time.
            # Reducing the page count would cause incorrect layer offsets and a
            # device hang.  Keep the full page count; TurboQuant saves memory by
            # compressing old tokens in-place (freeing bf16 pages back to the
            # pool for reuse by *other* requests on the same model).
            num_pages = runtime.max_batch_size * max_blocks_per_seq
            print(
                f"[TurboQuant] Using full bf16 pages ({num_pages}) to match "
                f"compiled kernel stride. Compression frees pages for reuse.",
                flush=True,
            )
        if num_pages is None:
            num_pages = runtime.max_batch_size * max_blocks_per_seq
        self._init_blocks(num_pages, runtime.page_size)
        kv_dtype = getattr(torch, runtime.kv_dtype)
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
        )

    def allocate_for_prompt(self, model_id: str, request_id: str, prompt_len: int) -> KvAllocation:
        """Allocate enough KV pages to store a prompt of ``prompt_len`` tokens."""
        pool = self._pool(model_id)
        num_pages = max(1, math.ceil(prompt_len / pool.page_size))
        blocks = self.allocate_blocks(num_pages)
        if blocks is None:
            raise RuntimeError("Insufficient KV cache blocks.")
        self.request_blocks[request_id] = blocks
        page_ids = [block.block_id for block in blocks]
        return KvAllocation(
            request_id=request_id,
            model_id=model_id,
            page_ids=page_ids,
            tokens_capacity=len(page_ids) * pool.page_size,
            tokens_used=0,
        )

    def allocate_blocks(self, num_blocks: int) -> list[KVCacheBlock] | None:
        """Allocate physical KV blocks, evicting stale prefix hashes as needed."""
        if num_blocks <= 0:
            return []
        if self.num_free_blocks < num_blocks:
            return None
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self.free_queue.popleft()
            if block is None:
                for allocated in blocks:
                    self.release(allocated)
                return None
            if block.block_hash is not None:
                self.hash_to_block.pop(block.block_hash, None)
                block.block_hash = None
            block.ref_cnt = 1
            blocks.append(block)
        return blocks

    def allocate_block_ids(self, num_blocks: int) -> list[int] | None:
        """Allocate physical KV blocks and return their IDs."""
        blocks = self.allocate_blocks(num_blocks)
        if blocks is None:
            return None
        return [block.block_id for block in blocks]

    def release_blocks_by_ids(self, *block_id_groups: list[int]) -> None:
        """Release request references for one or more groups of physical block IDs."""
        for block_ids in block_id_groups:
            for block_id in block_ids:
                self.release(self.blocks[block_id])

    def release_cached_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Release cached block objects returned by ``get_computed_blocks``."""
        for block in blocks:
            self.release(block)

    def release_request(self, request_id: str) -> None:
        """Release all blocks tracked for a request."""
        blocks = self.request_blocks.pop(request_id, [])
        for block in blocks:
            self.release(block)

    def get_cached_block(self, block_hash: int) -> KVCacheBlock | None:
        """Return and reference a cached block for one block hash."""
        if not self.enable_prefix_cache:
            return None
        block = self.hash_to_block.get(block_hash)
        if block is None:
            return None
        if block.ref_cnt == 0:
            self.free_queue.remove(block)
        block.ref_cnt += 1
        return block

    def cache_block(self, block: KVCacheBlock, block_hash: int) -> None:
        """Publish a full block to the prefix cache."""
        if not self.enable_prefix_cache:
            return
        if block.block_hash is not None and block.block_hash in self.hash_to_block:
            del self.hash_to_block[block.block_hash]
        block.block_hash = block_hash
        self.hash_to_block[block_hash] = block

    def cache_block_ids(self, block_ids: list[int], block_hashes: list[int], start: int, end: int) -> None:
        """Publish a range of full blocks to the prefix cache."""
        if not self.enable_prefix_cache:
            return
        for idx in range(start, end):
            if idx >= len(block_hashes) or idx >= len(block_ids):
                break
            self.cache_block(self.blocks[block_ids[idx]], block_hashes[idx])

    def release(self, block: KVCacheBlock) -> None:
        """Release one request reference to a block."""
        if block.ref_cnt <= 0:
            return
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_queue.append(block)

    def _iter_block_hashes(self, token_ids: list[int]):
        """Yield (block_index, block_hash) for each full block in the token sequence."""
        parent_hash = NONE_HASH
        num_full_blocks = len(token_ids) // self.block_size
        for i in range(num_full_blocks):
            start = i * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            parent_hash = hash_block_tokens(parent_hash, block_tokens)
            yield i, parent_hash

    def get_computed_blocks(self, token_ids: list[int]) -> list[KVCacheBlock]:
        """Find the longest full-block cached prefix for the token sequence."""
        if not self.enable_prefix_cache:
            return []
        hit_blocks: list[KVCacheBlock] = []
        for _, block_hash in self._iter_block_hashes(token_ids):
            block = self.get_cached_block(block_hash)
            if block is None:
                break
            hit_blocks.append(block)
        return hit_blocks

    def compute_block_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute chained hashes for all full blocks in the token sequence."""
        return [block_hash for _, block_hash in self._iter_block_hashes(token_ids)]

    def ensure_one_more_slot(self, alloc: KvAllocation) -> int:
        """Ensure a request has capacity for one more token and return its slot."""
        pool = self._pool(alloc.model_id)
        if alloc.tokens_used >= alloc.tokens_capacity:
            blocks = self.allocate_blocks(1)
            if blocks is None:
                raise RuntimeError("Insufficient KV cache blocks.")
            self.request_blocks.setdefault(alloc.request_id, []).extend(blocks)
            alloc.page_ids.extend(block.block_id for block in blocks)
            alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
        return self.slot_mapping_for_request(alloc, alloc.tokens_used)

    def block_table_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return a dense ``[batch, max_blocks]`` block table for requests."""
        return block_table_from_block_ids([alloc.page_ids for alloc in allocations])

    def slot_mapping_for_request(self, alloc: KvAllocation, token_index: int | None = None) -> int:
        """Return the physical slot index for a request token."""
        pool = self._pool(alloc.model_id)
        logical_index = alloc.tokens_used if token_index is None else token_index
        return slot_mapping_for_decode(alloc.page_ids, logical_index, pool.page_size)

    def slot_mapping_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return current decode slot mappings for a batch."""
        return torch.tensor(
            [self.slot_mapping_for_request(alloc) for alloc in allocations],
            dtype=torch.int32,
        )

    def slot_mapping_for_positions(
        self,
        alloc: KvAllocation,
        num_tokens: int,
        *,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        """Return per-position slot mappings, optionally padded with -1."""
        pool = self._pool(alloc.model_id)
        return slot_mapping_for_positions(alloc.page_ids, num_tokens, pool.page_size, max_tokens=max_tokens)

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

    def read_context(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        upto_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def materialize_single_layer_cache(
        self,
        model_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.release_request(alloc.request_id)
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

    def restore_compressed_tokens(self, model_id: str, alloc: KvAllocation, npu_runner=None) -> None:
        """Decompress old tokens back into pages and prepend before kernel execution.

        Takes pages from the free pool (same count as were freed during compression)
        so the total page count equals the pre-compression count, never exceeding
        max_blocks_per_seq.

        Args:
            npu_runner: If provided, use NPU kernels for decompress. Otherwise
                        fall back to pure Python/PyTorch.
        """
        pool = self._pool(model_id)
        if pool.compressed_segments is None or pool.kv_compressor is None:
            return
        req_segments = pool.compressed_segments.get(alloc.request_id)
        if not req_segments:
            return

        t_restore = time.perf_counter()

        # segment.token_count is page-aligned (set by compress_old_tokens).
        sample_segment = next(iter(req_segments.values()))
        compressed_token_count = sample_segment.token_count
        needed_pages = compressed_token_count // pool.page_size

        # Take pages from the free pool (same count as were freed by compress).
        restore_pages = self._take_pages(pool, needed_pages)
        print(
            f"[TurboQuant] Restoring compressed tokens for request {alloc.request_id}: "
            f"{compressed_token_count} old tokens into {needed_pages} pages {restore_pages}",
            flush=True,
        )

        # Build a temporary allocation over restore pages for writing.
        restore_alloc = KvAllocation(
            request_id=alloc.request_id,
            model_id=alloc.model_id,
            page_ids=restore_pages,
            tokens_capacity=needed_pages * pool.page_size,
            tokens_used=0,
        )

        # Decompress old tokens into restore pages.
        t_decompress = time.perf_counter()
        for layer_idx, segment in req_segments.items():
            if npu_runner is not None:
                keys, values = pool.kv_compressor.decompress_layer_npu(
                    layer_idx, segment.compressed_k, segment.compressed_v,
                    run_kv_dequantize_fn=npu_runner.run_tq_decompress,
                )
            else:
                keys, values = pool.kv_compressor.decompress_layer(
                    layer_idx, segment.compressed_k, segment.compressed_v,
                )
            self._write_tokens_from_compressed(
                pool, layer_idx, restore_alloc, 0, keys, values,
            )
        dt_decompress = (time.perf_counter() - t_decompress) * 1000
        print(
            f"[TurboQuant] restore: {len(req_segments)} layers decompressed "
            f"{dt_decompress:.1f} ms ({dt_decompress/len(req_segments):.1f} ms/layer)",
            flush=True,
        )

        # Prepend restore pages so old tokens come before resident tokens.
        # Total pages = needed_pages + resident pages.  Since ensure_one_more_slot
        # already allocated a page for the decode token (between run_decode calls),
        # the alloc may have grown by one.  Trim back to max_blocks_per_seq if needed
        # — the decode page from ensure_one_more_slot is now subsumed by the restore.
        alloc.page_ids = restore_pages + alloc.page_ids
        alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size

        if len(alloc.page_ids) > pool.max_blocks_per_seq:
            excess = len(alloc.page_ids) - pool.max_blocks_per_seq
            trimmed = alloc.page_ids[pool.max_blocks_per_seq:]
            alloc.page_ids = alloc.page_ids[:pool.max_blocks_per_seq]
            alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
            pool.free_pages.extend(trimmed)
            print(
                f"[TurboQuant] Trimmed {excess} excess pages to fit max_blocks={pool.max_blocks_per_seq}",
                flush=True,
            )

        dt_total = (time.perf_counter() - t_restore) * 1000
        print(
            f"[TurboQuant] restore complete: {dt_total:.1f} ms total, "
            f"{alloc.tokens_used} tokens, {len(alloc.page_ids)} pages",
            flush=True,
        )
        print(
            f"[TQ-DEBUG] restore: page_ids={alloc.page_ids}, "
            f"tokens_used={alloc.tokens_used}, tokens_capacity={alloc.tokens_capacity}, "
            f"max_blocks_per_seq={pool.max_blocks_per_seq}, "
            f"pool.num_pages={pool.key_pages.shape[1]}, "
            f"pool.free_pages remaining={len(pool.free_pages)}",
            flush=True,
        )

    def compress_old_tokens(self, model_id: str, alloc: KvAllocation, npu_runner=None) -> None:
        """Compress old tokens (beyond residual_window) and free their bf16 pages.

        Only fully-old pages are freed.  A boundary page that contains both old
        and residual tokens is kept as a resident page — this avoids having to
        split a page or lose residual data.

        Args:
            npu_runner: If provided, use NPU kernels for compress. Otherwise
                        fall back to pure Python/PyTorch.
        """
        pool = self._pool(model_id)
        if pool.kv_compressor is None or pool.kv_quant_config is None:
            return

        residual_window = pool.kv_quant_config.residual_window
        tokens_used = alloc.tokens_used
        if tokens_used <= residual_window:
            return

        old_token_count = tokens_used - residual_window
        # Use FLOOR division: only free pages that are FULLY occupied by old
        # tokens.  The boundary page (if any) may contain both old and residual
        # tokens and is kept as a resident page.
        old_page_count = old_token_count // pool.page_size
        if old_page_count == 0:
            # Not enough old tokens to fill even one page — nothing to free.
            return
        # Page-aligned count: the actual number of tokens we compress.
        compressed_token_count = old_page_count * pool.page_size
        print(
            f"[TurboQuant] Compressing old tokens: total={tokens_used}, "
            f"window={residual_window}, compressing={compressed_token_count} old tokens "
            f"({old_page_count} pages, {old_token_count - compressed_token_count} "
            f"boundary tokens kept in resident page)",
            flush=True,
        )

        t_compress = time.perf_counter()

        if pool.compressed_segments is None:
            pool.compressed_segments = {}
        if alloc.request_id not in pool.compressed_segments:
            pool.compressed_segments[alloc.request_id] = {}

        for layer_idx in range(pool.num_layers):
            old_keys, old_values = self.read_context(
                layer_idx, alloc, upto_tokens=compressed_token_count,
            )
            if layer_idx == 0:
                print(
                    f"[TQ-DEBUG] compress_old_tokens layer 0: "
                    f"old_keys shape={old_keys.shape}, dtype={old_keys.dtype}, "
                    f"nan={torch.isnan(old_keys).any().item()}, inf={torch.isinf(old_keys).any().item()}, "
                    f"min={old_keys.min().item():.6f}, max={old_keys.max().item():.6f}, "
                    f"mean={old_keys.mean().item():.6f}",
                    flush=True,
                )
            if npu_runner is not None:
                compressed_k, compressed_v = pool.kv_compressor.compress_layer_npu(
                    layer_idx, old_keys, old_values,
                    run_kv_quantize_fn=npu_runner.run_tq_compress,
                )
            else:
                compressed_k, compressed_v = pool.kv_compressor.compress_layer(
                    layer_idx, old_keys, old_values,
                )
            pool.compressed_segments[alloc.request_id][layer_idx] = _CompressedSegment(
                token_start=0,
                token_count=compressed_token_count,
                compressed_k=compressed_k,
                compressed_v=compressed_v,
            )

        dt_compress = (time.perf_counter() - t_compress) * 1000
        print(
            f"[TurboQuant] compress: {pool.num_layers} layers compressed "
            f"{dt_compress:.1f} ms ({dt_compress/pool.num_layers:.1f} ms/layer)",
            flush=True,
        )

        # Free old token pages: remove the leading pages that held old tokens
        # and return them to the pool.  tokens_used is kept at the full logical
        # sequence length so the kernel knows the total context length.
        freed_pages = alloc.page_ids[:old_page_count]
        alloc.page_ids = alloc.page_ids[old_page_count:]
        alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
        pool.free_pages.extend(freed_pages)
        # Store freed page IDs in every layer's segment so restore can reuse
        # the exact same physical pages (avoids exceeding max_blocks_per_seq).
        for seg in pool.compressed_segments[alloc.request_id].values():
            seg.freed_page_ids = freed_pages
        print(
            f"[TurboQuant] Freed {len(freed_pages)} bf16 pages, "
            f"{len(alloc.page_ids)} resident pages remain, "
            f"pool free_pages={len(pool.free_pages)}",
            flush=True,
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
        print(
            f"[TQ-DEBUG] _write_tokens_from_compressed: layer={layer_idx}, "
            f"keys shape={keys.shape}, keys dtype={keys.dtype}, "
            f"cache_dtype={cache_dtype}, pool.num_pages={pool.key_pages.shape[1]}, "
            f"alloc.page_ids len={len(alloc.page_ids)}, "
            f"alloc.page_ids max={max(alloc.page_ids) if alloc.page_ids else 'N/A'}, "
            f"keys nan={torch.isnan(keys).any().item()}, inf={torch.isinf(keys).any().item()}, "
            f"keys min={keys.min().item():.6f}, max={keys.max().item():.6f}",
            flush=True,
        )
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
            num_compressed_requests = stats.get("compressed_requests", 0)
            print(f"[KV Cache] TurboQuant compressed data = {_fmt(compressed_bytes)}", flush=True)
            print(f"[KV Cache] TurboQuant compressed requests = {num_compressed_requests}", flush=True)
            if total_kv_bytes > 0:
                print(f"[KV Cache] effective memory (bf16 + compressed) = {_fmt(total_kv_bytes + compressed_bytes)}", flush=True)
            # Compare actual pages vs pages needed per request without compression.
            # Without TurboQuant each sequence would need max_blocks_per_seq pages.
            pages_per_seq_uncompressed = pool.max_blocks_per_seq
            print(
                f"[KV Cache] pages per request: compressed pool={total_pages}, "
                f"uncompressed would need={pages_per_seq_uncompressed} per request",
                flush=True,
            )

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
            "compressed_requests": len(pool.compressed_segments),
        }

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]


# ---------------------------------------------------------------------------
# Standalone utility functions for block-table / slot-mapping computation.
# These are pure math (no allocation state) and can be used directly by
# the serving worker without a KvCacheManager instance.
# ---------------------------------------------------------------------------

def block_table_from_block_ids(block_ids_batch: list[list[int]]) -> torch.Tensor:
    """Return a dense ``[batch, max_blocks]`` int32 block table from physical block IDs."""
    max_blocks = max((len(bids) for bids in block_ids_batch), default=0)
    table = torch.full((len(block_ids_batch), max_blocks), -1, dtype=torch.int32)
    for row, block_ids in enumerate(block_ids_batch):
        if block_ids:
            table[row, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32)
    return table


def slot_mapping_from_block_ids(
    block_ids: list[int],
    token_indices: range | list[int],
    page_size: int,
) -> torch.Tensor:
    """Return physical slot IDs for absolute token indices in one request."""
    slots = []
    for token_index in token_indices:
        page_idx = token_index // page_size
        offset = token_index % page_size
        slots.append(block_ids[page_idx] * page_size + offset)
    return torch.tensor(slots, dtype=torch.int32)


def slot_mapping_from_block_ids_batch(
    block_ids_batch: list[list[int]],
    token_indices_batch: list[range | list[int]],
    page_size: int,
) -> torch.Tensor:
    """Return a padded ``[batch, max_tokens]`` int32 slot mapping from block IDs."""
    mappings = [
        slot_mapping_from_block_ids(block_ids, token_indices, page_size)
        for block_ids, token_indices in zip(block_ids_batch, token_indices_batch, strict=True)
    ]
    max_slots = max((mapping.numel() for mapping in mappings), default=0)
    table = torch.full((len(mappings), max_slots), -1, dtype=torch.int32)
    for row, mapping in enumerate(mappings):
        if mapping.numel() > 0:
            table[row, : mapping.numel()] = mapping
    return table


def slot_mapping_for_decode(page_ids: list[int], tokens_used: int, page_size: int) -> int:
    """Return the single physical slot for a decode step."""
    page_idx = tokens_used // page_size
    offset = tokens_used % page_size
    return page_ids[page_idx] * page_size + offset


def slot_mapping_for_positions(
    page_ids: list[int],
    num_tokens: int,
    page_size: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Return per-position slot mappings, optionally padded with -1."""
    size = num_tokens if max_tokens is None else max_tokens
    mapping = torch.full((size,), -1, dtype=torch.int32)
    for token_index in range(num_tokens):
        page_idx = token_index // page_size
        offset = token_index % page_size
        mapping[token_index] = page_ids[page_idx] * page_size + offset
    return mapping
