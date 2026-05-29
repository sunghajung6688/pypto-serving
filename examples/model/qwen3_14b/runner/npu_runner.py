# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    from python.core._profiling import StageTimer
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        KvAllocation,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
    from python.runtime.worker import Worker as LlmWorker
    from python.runtime.worker import WorkerTensor
except ImportError:
    from python.core._profiling import StageTimer
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        KvAllocation,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
    from python.runtime.worker import Worker as LlmWorker
    from python.runtime.worker import WorkerTensor

_TIMING_ENABLED = True
_LOGITS_BATCH_TILE = 16


@dataclass
class _KernelLayerWeights:
    """Kernel-ready weights for one transformer layer."""

    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class _L2Callable:
    """Assembled non-L3 callable and launch metadata."""

    chip_callable: object
    runtime_name: str
    block_dim: int
    aicpu_thread_num: int
    param_infos: tuple[object, ...]


@dataclass
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors."""

    prefill: _L2Callable
    decode: _L2Callable
    final_rms: _L2Callable | None
    lm_head: _L2Callable | None
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    layers: list[_KernelLayerWeights]
    decode_weights: dict[str, torch.Tensor]
    decode_logits_buffer: torch.Tensor
    prefill_block_table_stride: int | None = None
    decode_block_table_stride: int | None = None
    # L3-wrapped generate artifacts. Populated only when l3_mode=True.
    stacked_weights: dict[str, torch.Tensor] | None = None
    l3_generate_chip_callables: dict[str, object] | None = None
    l3_generate_entry_fn: object | None = None
    l3_generate_sub_worker_fns: dict[str, object] | None = None
    l3_generate_dc: object | None = None  # DistributedConfig
    l3_generate_platform: str | None = None
    l3_generate_runtime_name: str | None = None
    l3_generate_param_infos: object | None = None
    # KV cache quantization NPU kernels. None when disabled or using PyTorch fallback.
    tq_compress: _L2Callable | None = None
    tq_decompress: _L2Callable | None = None
    # TQ fused-dequant kernels for prefill and decode. None when kv-quant disabled.
    prefill_tq: _L2Callable | None = None
    decode_tq: _L2Callable | None = None
    # Per-layer rotation matrices (stacked [num_layers * head_dim, head_dim] BF16).
    rot_matrices: torch.Tensor | None = None


@dataclass
class _PrefillInputs:
    """Host tensors passed to the prefill kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeInputs:
    """Padded host tensors passed to the decode kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _L2ProgramHandle:
    """L2 callable registration state for one runner process."""

    callable_id: int
    runtime_name: str


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        model_id: str,
        compiled: _CompiledKernels,
        platform: str,
        device_id: int,
        save_kernels_dir: str | None,
        l3_trace: bool,
    ) -> None:
        super().__init__()
        self._model_id = model_id
        self._compiled = compiled
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._l3_trace = l3_trace
        self._l2_workers: dict[str, LlmWorker] = {}
        self._l2_programs: dict[int, _L2ProgramHandle] = {}
        self._l2_child_allocs: dict[tuple[str, int], tuple[int, int]] = {}
        self._l2_dirty_kv_models: set[str] = set()

    def run_tq_compress(self, flat_kv, rot_matrix, lo, inv_step, n_levels_m1):
        """TurboQuant compress: L2 norm + rotate + fixed-range quantize.

        Args:
            flat_kv: (N, D) BF16 tensor.
            rot_matrix: (D, D) BF16 rotation matrix.
            lo: float, lower bound of quantization range.
            inv_step: float, 1 / step_size.
            n_levels_m1: int, n_levels - 1.

        Returns:
            (indices, norms) — indices (N, D) uint8, norms (N, 1) float16.
        """
        import torch

        compiled = self._compiled
        if compiled.tq_compress is not None:
            n = flat_kv.shape[0]
            d = flat_kv.shape[1]
            indices_out = torch.zeros(n, d, dtype=torch.uint8, device="cpu")
            norms_out = torch.zeros(n, 1, dtype=torch.float16, device="cpu")

            self._run_l2_program(
                compiled.tq_compress,
                flat_kv.contiguous().cpu(),
                rot_matrix.contiguous().cpu(),
                ctypes.c_float(lo),
                ctypes.c_float(inv_step),
                ctypes.c_int(n_levels_m1),
                indices_out,
                norms_out,
            )
            return indices_out, norms_out

        # Fallback: pure PyTorch
        flat = flat_kv.float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        flat_norm = flat / vec_norms
        rotated = flat_norm @ rot_matrix.float().T
        scaled = (rotated - lo) * inv_step
        indices = torch.clamp(scaled.floor().long(), 0, n_levels_m1).to(torch.uint8)
        return indices, vec_norms.squeeze(-1).half()

    def run_tq_decompress(self, centroid_vals, rot_matrix, norms):
        """TurboQuant decompress: inverse rotate + scale by norms.

        Args:
            centroid_vals: (N, D) BF16 — already looked-up centroids.
            rot_matrix: (D, D) BF16 rotation matrix.
            norms: (N, 1) FP16 or BF16 norms.

        Returns:
            (N, D) BF16 reconstructed tensor.
        """
        import torch

        compiled = self._compiled
        if compiled.tq_decompress is not None:
            n = centroid_vals.shape[0]
            d = centroid_vals.shape[1]
            reconstructed_out = torch.zeros(n, d, dtype=torch.bfloat16, device="cpu")

            self._run_l2_program(
                compiled.tq_decompress,
                centroid_vals.contiguous().cpu(),
                rot_matrix.contiguous().cpu(),
                norms.contiguous().cpu(),
                reconstructed_out,
            )
            return reconstructed_out

        # Fallback: pure PyTorch
        rotated = centroid_vals.float() @ rot_matrix.float()
        return (rotated * norms.float()).to(torch.bfloat16)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the JIT all-layer prefill kernel and return next-token logits."""
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        dw = compiled.decode_weights
        t_prefill_start = time.perf_counter()

        k_cache, v_cache = self.materialize_full_layer_cache(
            model.config.model_id,
        )
        logits_padded = torch.zeros(
            (prefill_inputs.actual_batch, compiled.padded_vocab),
            dtype=torch.float32,
        ).share_memory_()

        self._run_l2_program(
            compiled.prefill,
            prefill_inputs.hidden,
            prefill_inputs.seq_lens,
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_input_rms_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wq"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wk"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wv"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_q_norm_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_k_norm_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_cos),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_sin),
            prefill_inputs.block_table,
            prefill_inputs.slot_mapping,
            k_cache,
            v_cache,
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wo"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_post_rms_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_gate"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_up"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_down"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.final_norm_weight),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.padded_lm_head_weight),
            logits_padded,
        )
        self._l2_dirty_kv_models.add(model.config.model_id)

        if _TIMING_ENABLED:
            print(
                f"[timing] prefill: fused {len(model.layers)} layers, "
                f"{(time.perf_counter() - t_prefill_start) * 1000:.2f} ms",
                flush=True,
            )

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[:, : model.config.vocab_size],
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the JIT all-layer decode kernel and return next-token logits."""
        compiled = self._compiled
        dw = compiled.decode_weights
        model_id = model.config.model_id
        use_tq = (self._kv_cache_manager.is_quantization_enabled(model_id)
                  and hasattr(compiled, "decode_tq") and compiled.decode_tq is not None)

        decode_inputs = self._prepare_decode_inputs(model, batch)
        hidden = decode_inputs.hidden

        k_cache, v_cache = self.materialize_full_layer_cache(
            model.config.model_id,
        )
        refresh_kv_cache = model.config.model_id in self._l2_dirty_kv_models

        if decode_inputs.actual_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"decode batch {decode_inputs.actual_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )
        logits_padded = compiled.decode_logits_buffer[: decode_inputs.actual_batch]
        self._run_l2_program(
            compiled.decode,
            hidden,
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_input_rms_weight"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wq"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wk"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wv"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_q_norm_weight"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_k_norm_weight"]),
            decode_inputs.seq_lens,
            decode_inputs.block_table,
            decode_inputs.slot_mapping,
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.rope_cos),
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.rope_sin),
            self._l2_child_tensor(compiled.decode.runtime_name, k_cache, refresh=refresh_kv_cache),
            self._l2_child_tensor(compiled.decode.runtime_name, v_cache, refresh=refresh_kv_cache),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wo"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_post_rms_weight"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_gate"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_up"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_down"]),
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.final_norm_weight),
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.padded_lm_head_weight),
            logits_padded,
        )
        self._l2_dirty_kv_models.discard(model.config.model_id)
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(
            hidden_states=hidden.float(),
            logits=logits_padded[:, : model.config.vocab_size].to(hidden.device),
        )

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        """Run final RMSNorm and LM head kernels for a hidden-state batch."""
        compiled = self._compiled
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        if compiled.final_rms is None or compiled.lm_head is None:
            raise RuntimeError("standalone final_rms/lm_head kernels are not compiled")

        actual_batch = hidden.shape[0]
        if actual_batch > _LOGITS_BATCH_TILE:
            raise ValueError(
                f"logit batch {actual_batch} exceeds _LOGITS_BATCH_TILE {_LOGITS_BATCH_TILE}"
            )

        x = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        x[:actual_batch] = hidden.to(torch.bfloat16).cpu()
        if not isinstance(compiled.final_rms, _L2Callable) or not isinstance(compiled.lm_head, _L2Callable):
            normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16)
            compiled.final_rms(x, compiled.final_norm_weight, normed, config=None)
            logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32)
            compiled.lm_head(normed, compiled.padded_lm_head_weight, logits_padded, config=None)
            return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

        normed: torch.Tensor | WorkerTensor
        x_arg: torch.Tensor | WorkerTensor = x
        worker: LlmWorker | None = None
        if compiled.final_rms.runtime_name == compiled.lm_head.runtime_name:
            worker = self._worker_for_runtime(compiled.final_rms.runtime_name)
            x_arg = worker.alloc_tensor(x.shape, x.dtype, init=x)
            normed = worker.alloc_tensor(x.shape, x.dtype)
        else:
            normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()

        try:
            self._run_l2_program(
                compiled.final_rms,
                x_arg,
                self._l2_child_tensor(compiled.final_rms.runtime_name, compiled.final_norm_weight),
                normed,
            )

            logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
            self._run_l2_program(
                compiled.lm_head,
                normed,
                self._l2_child_tensor(compiled.lm_head.runtime_name, compiled.padded_lm_head_weight),
                logits_padded,
            )
        finally:
            if isinstance(x_arg, WorkerTensor):
                if worker is None:
                    raise RuntimeError("missing L2 worker for child-memory logits projection")
                worker.free_tensor(x_arg)
            if isinstance(normed, WorkerTensor):
                if worker is None:
                    raise RuntimeError("missing L2 worker for child-memory logits projection")
                worker.free_tensor(normed)
        return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

    def _run_l2_program(self, callable_spec: _L2Callable, *args: Any) -> None:
        """Run a compiled non-L3 program through the LLM Simpler worker."""
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        handle = self._ensure_l2_program(callable_spec)
        orch_args = self._build_l2_orch_args(callable_spec, args)

        cfg = CallConfig()
        cfg.block_dim = callable_spec.block_dim
        cfg.aicpu_thread_num = callable_spec.aicpu_thread_num

        worker = self._l2_workers[handle.runtime_name]
        worker.run(handle.callable_id, orch_args, cfg)

    def _worker_for_runtime(self, runtime_name: str) -> LlmWorker:
        """Return an initialized worker for ``runtime_name``."""
        worker = self._l2_workers.get(runtime_name)
        if worker is not None:
            return worker
        worker = LlmWorker(
            level=2,
            platform=self._platform,
            runtime=runtime_name,
            device_id=self._device_id,
            auto_init=True,
        )
        self._l2_workers[runtime_name] = worker
        return worker

    def _ensure_l2_program(self, callable_spec: _L2Callable) -> _L2ProgramHandle:
        """Register and cache one executor-assembled non-L3 callable."""
        key = id(callable_spec)
        cached = self._l2_programs.get(key)
        if cached is not None:
            return cached

        worker = self._worker_for_runtime(callable_spec.runtime_name)

        handle = _L2ProgramHandle(
            callable_id=worker.register(callable_spec.chip_callable),
            runtime_name=callable_spec.runtime_name,
        )
        self._l2_programs[key] = handle
        return handle

    def _l2_child_tensor(
        self,
        runtime_name: str,
        tensor: torch.Tensor,
        *,
        upload: bool = True,
        refresh: bool = False,
    ) -> WorkerTensor:
        """Return a worker-resident view for a CPU tensor's backing storage."""
        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        if tensor.device.type != "cpu":
            raise ValueError("child-memory tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("child-memory tensor must be contiguous")
        tensor = self._share_cpu_tensor(tensor)
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        tensor_offset = int(tensor.data_ptr()) - storage_ptr
        if tensor_offset < 0 or tensor_offset + int(tensor.nbytes) > storage_nbytes:
            raise ValueError("tensor view is outside its backing storage")

        key = (runtime_name, storage_ptr)
        alloc = self._l2_child_allocs.get(key)
        if alloc is None:
            worker = self._worker_for_runtime(runtime_name)
            dev_ptr = worker.malloc(storage_nbytes)
            if upload:
                worker.copy_to(dev_ptr, storage_ptr, storage_nbytes)
            alloc = (dev_ptr, storage_nbytes)
            self._l2_child_allocs[key] = alloc
        elif upload and refresh:
            worker = self._worker_for_runtime(runtime_name)
            worker.copy_to(alloc[0], storage_ptr, storage_nbytes)

        dev_base, _ = alloc
        shape = tuple(int(dim) for dim in tensor.shape)
        return WorkerTensor(
            data_ptr=dev_base + tensor_offset,
            shape=shape,
            dtype=torch_dtype_to_datatype(tensor.dtype),
        )

    def _sync_kv_page_from_npu(
        self,
        model_id: str,
        alloc: KvAllocation,
    ) -> None:
        """Copy the page containing the latest token's KV from NPU back to CPU.

        This is needed so that CPU-side key_pages/value_pages stay in sync
        with NPU after the decode kernel writes new token KV via slot_mapping.
        """
        pool = self._kv_cache_manager._pool(model_id)
        if pool.key_pages.device.type != "cpu":
            return

        tokens_used = alloc.tokens_used
        if tokens_used <= 0:
            return

        page_idx = (tokens_used - 1) // pool.page_size
        if page_idx >= len(alloc.page_ids):
            return
        physical_page = alloc.page_ids[page_idx]

        # Bytes per page per layer in key_pages: H * S * D * element_size
        page_bytes = pool.num_kv_heads * pool.page_size * pool.head_dim * pool.key_pages.element_size()

        # Get NPU dev_ptr for key_pages and value_pages child allocations.
        runtime_name = self._compiled.decode.runtime_name
        k_cache, v_cache = self._kv_cache_manager.materialize_full_layer_cache(model_id)

        for cache_tensor, pages_tensor in [
            (k_cache, pool.key_pages),
            (v_cache, pool.value_pages),
        ]:
            storage = pages_tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            key = (runtime_name, storage_ptr)
            child_alloc = self._l2_child_allocs.get(key)
            if child_alloc is None:
                continue
            dev_ptr = child_alloc[0]

            # Offset for this page across all layers.
            # key_pages shape: [num_layers, num_pages, H, S, D]
            # Flattened: layers are stacked, each layer has num_pages * H * S * D * el_size bytes.
            layer_stride = pool.key_pages.shape[1] * page_bytes  # bytes per layer
            for layer_idx in range(pool.num_layers):
                offset = layer_idx * layer_stride + physical_page * page_bytes
                worker = self._worker_for_runtime(runtime_name)
                worker.copy_from(
                    storage_ptr + offset,
                    dev_ptr + offset,
                    page_bytes,
                )

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release non-L3 child-memory allocations and L2 workers."""
        for (runtime_name, _), (dev_ptr, _nbytes) in list(self._l2_child_allocs.items()):
            worker = self._l2_workers.get(runtime_name)
            if worker is not None and worker.initialized:
                worker.free(dev_ptr)
        self._l2_child_allocs.clear()
        self._l2_programs.clear()
        self._l2_dirty_kv_models.clear()
        for worker in self._l2_workers.values():
            worker.close()
        self._l2_workers.clear()

    @staticmethod
    def _build_l2_orch_args(callable_spec: _L2Callable, args: tuple[Any, ...]):
        """Build ``ChipStorageTaskArgs`` for a compiled L2 program call."""
        from simpler.task_interface import ChipStorageTaskArgs, ContinuousTensor, scalar_to_uint64  # noqa: PLC0415
        from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

        param_infos = callable_spec.param_infos
        if len(args) != len(param_infos):
            names = [p.name for p in param_infos]
            raise TypeError(
                f"compiled program expects {len(param_infos)} arguments, got {len(args)}. Parameters: {names}"
            )

        # ChipStorageTaskArgs requires all tensors added before scalars.
        # Collect them separately then add tensors first, scalars second.
        tensor_entries = []
        scalar_entries = []
        for info, arg in zip(param_infos, args, strict=True):
            if info.shape is None:
                if not isinstance(arg, ctypes._SimpleCData):
                    raise TypeError(f"scalar parameter {info.name!r} must be passed as a ctypes scalar")
                scalar_entries.append(arg)
                continue
            if isinstance(arg, WorkerTensor):
                tensor_entries.append(arg.to_continuous_tensor())
                continue
            if isinstance(arg, ContinuousTensor):
                tensor_entries.append(arg)
                continue
            if not isinstance(arg, torch.Tensor):
                raise TypeError(f"tensor parameter {info.name!r} expects torch.Tensor, got {type(arg).__name__}")
            if arg.device.type != "cpu":
                raise ValueError(f"tensor parameter {info.name!r} must be on CPU for Simpler L2 dispatch")
            if not arg.is_contiguous():
                raise ValueError(f"tensor parameter {info.name!r} must be contiguous")
            if not arg.is_shared():
                arg.share_memory_()
            tensor_entries.append(make_tensor_arg(arg))

        orch_args = ChipStorageTaskArgs()
        for t in tensor_entries:
            orch_args.add_tensor(t)
        for s in scalar_entries:
            orch_args.add_scalar(scalar_to_uint64(s))
        return orch_args

    # ── L3-wrapped generate: entire prefill + decode loop in one worker.run() ──

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Run the full generate loop inside a single Worker(level=3).

        Dispatches prefill chunks + final_rms + lm_head + decode loop entirely
        within one worker.run() call, using sub_worker for CPU-side sampling
        and embedding lookup between device dispatches.

        Returns (generated_token_ids, final_hidden).
        """
        from simpler.task_interface import CallConfig  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        _verbose = self._l3_trace
        timer = StageTimer(
            enabled=_verbose,
            prefix="L3-breakdown",
            title="run_generate_l3 stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        compiled = self._compiled
        if compiled.l3_generate_chip_callables is None:
            raise RuntimeError("L3 generate artifacts not compiled.")
        if max_new_tokens > model.runtime.max_new_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds compiled L3 limit "
                f"{model.runtime.max_new_tokens}"
            )

        _mark("entry_validate")
        prefill_inputs = self._prepare_prefill_inputs(model, prefill_batch, padded=True)
        _mark("prepare_prefill_inputs")
        actual_batch = prefill_inputs.actual_batch
        if actual_batch != 1:
            raise ValueError(
                "run_generate_l3 currently supports batch_size=1 only; "
                f"got {actual_batch} requests."
            )
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        k_cache_all, v_cache_all = self.materialize_full_layer_cache(
            model.config.model_id,
        )
        _mark("kv_cache_materialize")

        # Build initial decode hidden: last prompt token embedding per batch.
        decode_hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        decode_slot_mapping = torch.zeros((actual_batch,), dtype=torch.int32)
        for b in range(actual_batch):
            seq_len_b = int(prefill_inputs.seq_lens[b].item())
            decode_hidden[b] = prefill_inputs.hidden[b, seq_len_b - 1, :]
            decode_slot_mapping[b] = int(
                prefill_inputs.slot_mapping[b * max_seq + seq_len_b - 1].item()
            )

        # Pre-allocate all shared-memory tensors for the full generate loop.
        prefill_out = torch.zeros_like(prefill_inputs.hidden).share_memory_()
        # decode_out and rms_x share one padded buffer so no CPU copy is needed
        # between them.  l3_generate writes to decode_out (the first actual_batch
        # rows); final_rms reads rms_x (all _LOGITS_BATCH_TILE rows).  The padding
        # rows stay zero throughout, satisfying the zero-pad contract of final_rms.
        _decode_out_storage = torch.zeros(
            (_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16
        ).share_memory_()
        decode_out = _decode_out_storage[:actual_batch, :]   # [actual_batch, hidden_size]
        rms_x = _decode_out_storage                          # [_LOGITS_BATCH_TILE, hidden_size]
        # final_rms / lm_head intermediates.
        rms_gamma = model.final_norm_weight.view(1, hidden_size).float().cpu().share_memory_()
        rms_normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        lm_head_weight = compiled.padded_lm_head_weight.share_memory_()
        logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
        _mark("alloc_io_buffers")
        # Sub-worker communication tensors.
        embed_tokens = model.embed_tokens.to(torch.bfloat16).cpu().share_memory_()
        _mark("embed_tokens_to_shm")
        done_flag = torch.zeros((1,), dtype=torch.int32).share_memory_()
        generated_ids = torch.full((max_new_tokens,), -1, dtype=torch.int64).share_memory_()
        token_count = torch.zeros((1,), dtype=torch.int32).share_memory_()
        # Mutable decode inputs (updated by sub_worker between steps).
        decode_hidden_buf = decode_hidden.clone().share_memory_()
        decode_seq_lens = prefill_inputs.seq_lens.clone().share_memory_()
        decode_slot_mapping_buf = decode_slot_mapping.clone().share_memory_()

        # Ensure all prefill inputs are in shared memory.
        prefill_hidden = prefill_inputs.hidden.share_memory_()
        prefill_seq_lens = prefill_inputs.seq_lens.share_memory_()
        prefill_slot_mapping = prefill_inputs.slot_mapping.share_memory_()
        block_table = prefill_inputs.block_table.share_memory_()
        rope_cos = compiled.rope_cos.share_memory_()
        rope_sin = compiled.rope_sin.share_memory_()
        k_cache_all = k_cache_all.share_memory_()
        v_cache_all = v_cache_all.share_memory_()
        _mark("kv_and_prefill_to_shm")

        # Ensure stacked weights are in shared memory.
        sm_sw = {}
        for k, v in compiled.stacked_weights.items():
            sm_sw[k] = v.share_memory_() if not v.is_shared() else v
        _mark("stacked_weights_to_shm")

        # SSA-name substrings that identify static weight parameters in the
        # tensors_dict built by _build_full_tensors().  Used in
        # generate_orch_fn to pre-upload these tensors once per generate call
        # (child_memory=True) so all decode dispatches skip H2D re-upload.
        _sw_substrings: frozenset[str] = frozenset(sm_sw.keys()) | {
            "rope_cos", "rope_sin", "final_norm_weight", "lm_head_weight",
        }

        # ── Sub-worker callable ──
        # Runs in a forked child process. Reads logits → argmax → embedding lookup
        # → writes decode_hidden_buf / decode_seq_lens / decode_slot_mapping_buf.
        _eos_id = eos_token_id
        _vocab = vocab_size
        _actual_batch = actual_batch
        _page_size = model.runtime.page_size

        # Pre-capture references for the sub_worker closure.
        # These are shared-memory tensors visible in the forked child.
        _embed_tokens = embed_tokens
        _done_flag = done_flag
        _generated_ids = generated_ids
        _token_count = token_count
        _decode_hidden_buf = decode_hidden_buf
        _decode_seq_lens = decode_seq_lens
        _decode_slot_mapping_buf = decode_slot_mapping_buf
        _logits_padded = logits_padded
        _prefill_batch = prefill_batch

        # L3 tracing: enabled by the executor's l3_trace flag (typically wired
        # to --profile-verbose). When disabled, all per-stage timestamp prints
        # are suppressed.
        _l3_trace_enabled = self._l3_trace

        # Shared-memory anchors so forked sub-worker callbacks can print times
        # relative to a common start (set at prefill submit_start in the parent).
        # Slot 0: start anchor (perf_counter seconds).
        # Slot 1: timestamp of the previous printed event (for Δprev).
        _t_anchors = torch.zeros(2, dtype=torch.float64).share_memory_()

        def _fmt_rel(t_now: float) -> str:
            t0 = float(_t_anchors[0].item())
            t_prev = float(_t_anchors[1].item())
            if t0 <= 0.0:
                return "t=+0.000ms (Δ+0.000ms)"
            rel_ms = (t_now - t0) * 1000.0
            d_ms = (t_now - t_prev) * 1000.0 if t_prev > 0.0 else 0.0
            _t_anchors[1] = t_now
            return f"t=+{rel_ms:.2f}ms (Δ+{d_ms:.2f}ms)"

        # Track when each sample_and_prepare finishes so we can split
        # chip-task time vs sub-worker IPC + work time.
        _sample_done_ts = [0.0]  # timestamp of last sample_and_prepare return

        def sample_and_prepare_fn(task_args):
            """Sub-worker: sample token from logits, prepare next decode inputs."""
            _t_enter = time.perf_counter()
            if _done_flag[0].item():
                return  # EOS already hit, no-op.

            # Read logits (written by lm_head chip task into logits_padded).
            logits = _logits_padded[0, :_vocab]
            token_id = int(logits.argmax().item())

            step = int(_token_count[0].item())
            if _l3_trace_enabled:
                _prev_done = _sample_done_ts[0]
                chip_ms = (_t_enter - _prev_done) * 1000.0 if _prev_done > 0.0 else 0.0
                print(
                    f"[L3-step] step={step:02d} sample_enter {_fmt_rel(_t_enter)}"
                    f"  chip_tasks={chip_ms:.1f}ms",
                    flush=True,
                )
            _generated_ids[step] = token_id
            _token_count[0] = step + 1

            if _eos_id is not None and token_id == _eos_id:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            if step + 1 >= max_new_tokens:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            # Embedding lookup.
            _decode_hidden_buf[0, :] = _embed_tokens[token_id]

            # Update seq_lens.
            for b in range(_actual_batch):
                new_seq_len = int(_decode_seq_lens[b].item()) + 1
                _decode_seq_lens[b] = new_seq_len
                # Update slot_mapping for next position.
                alloc = _prefill_batch.kv_allocations[b]
                page_idx = (new_seq_len - 1) // _page_size
                slot_in_page = (new_seq_len - 1) % _page_size
                if page_idx < len(alloc.page_ids):
                    _decode_slot_mapping_buf[b] = alloc.page_ids[page_idx] * _page_size + slot_in_page

            _t_exit = time.perf_counter()
            _sample_done_ts[0] = _t_exit
            if _l3_trace_enabled:
                work_ms = (_t_exit - _t_enter) * 1000.0
                print(
                    f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                    f"  sample_work={work_ms:.1f}ms",
                    flush=True,
                )

        # ── Build the orchestrator function ──

        lg_entry_fn = compiled.l3_generate_entry_fn
        lg_chip_callables = compiled.l3_generate_chip_callables
        lg_dc = compiled.l3_generate_dc
        lg_param_infos = compiled.l3_generate_param_infos

        def _submit_l3_generate(orch, config, tensors_dict, _keep):
            """Submit one l3_generate entry_fn call (dispatches all-layers L2 tasks)."""
            kwargs = {
                "tensors": tensors_dict,
                "callables": lg_callable_ids,
                "sub_ids": sub_ids,
                "_keep": _keep,
            }
            entry_params = inspect.signature(lg_entry_fn).parameters
            if "contexts" in entry_params:
                kwargs["contexts"] = getattr(worker, "chip_contexts", None)
            if "world_size" in entry_params:
                kwargs["world_size"] = 1
            lg_entry_fn(
                orch, None, config,
                **kwargs,
            )

        _has_prefill_tensor = torch.tensor(True, dtype=torch.bool).share_memory_()

        def _build_full_tensors():
            """Build the tensor dict for the single l3_generate dispatch.

            host_orch now owns the full generation loop (prefill step 0 +
            pl.unroll(max_new_tokens) decode steps), so this is called once.
            has_prefill is always True: step 0 inside host_orch runs prefill_all
            then the first decode; subsequent iterations run decode-only.
            """
            td = {}
            for info, val in zip(lg_param_infos, [
                prefill_hidden,           # prefill_hidden
                prefill_seq_lens,         # prefill_seq_lens
                prefill_slot_mapping,     # prefill_slot_mapping
                decode_hidden_buf,        # decode_hidden
                decode_seq_lens,          # decode_seq_lens (mutated by sample_and_prepare sub-worker)
                decode_slot_mapping_buf,  # decode_slot_mapping
                sm_sw["input_rms_weight"],
                sm_sw["wq"],
                sm_sw["wk"],
                sm_sw["wv"],
                sm_sw["q_norm_weight"],
                sm_sw["k_norm_weight"],
                rope_cos,
                rope_sin,
                block_table,
                k_cache_all,
                v_cache_all,
                sm_sw["wo"],
                sm_sw["post_rms_weight"],
                sm_sw["w_gate"],
                sm_sw["w_up"],
                sm_sw["w_down"],
                _has_prefill_tensor,      # has_prefill = True
                prefill_out,              # prefill_out
                decode_out,               # decode_out
                rms_x,                    # rms_x  (shares storage with decode_out)
                rms_gamma,                # final_norm_weight
                rms_normed,               # rms_normed
                lm_head_weight,           # lm_head_weight_t
                logits_padded,            # logits_padded
            ], strict=True):
                if not val.is_shared():
                    val = val.share_memory_()
                td[info.name] = val
            return td

        # KV cache device pointers collected inside generate_orch_fn so the
        # post-run sync-back step can copy updated K/V values back to host.
        # Format: list of (host_ptr: int, dev_ptr: int, nbytes: int).
        _kv_dev_ptrs: list[tuple[int, int, int]] = []

        def generate_orch_fn(orch, _args, _cfg):
            _keep: list = []
            call_config = CallConfig()
            call_config.block_dim = lg_dc.block_dim
            call_config.aicpu_thread_num = lg_dc.aicpu_thread_num

            _t_pf = time.perf_counter()
            _t_anchors[0] = _t_pf
            _t_anchors[1] = _t_pf
            _sample_done_ts[0] = _t_pf  # baseline for step 0 chip_tasks measurement
            if _l3_trace_enabled:
                print(f"[L3-step] host_orch submit_start {_fmt_rel(_t_pf)}", flush=True)

            # Single dispatch: host_orch drives prefill + all decode steps
            # (pl.unroll(max_new_tokens) inside host_orch).
            td = _build_full_tensors()

            from simpler.task_interface import ContinuousTensor as _CT  # noqa: PLC0415
            from simpler_setup.torch_interop import (  # noqa: PLC0415
                torch_dtype_to_datatype as _td2dt,
            )

            # Pre-upload static weight tensors once per generate call.
            # child_memory=True → runtime skips H2D + D2H on every dispatch.
            # ~3 400 ms of init_runtime per step reduced to ~11 ms.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if not any(_sub in _pname for _sub in _sw_substrings):
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            # Pre-upload KV cache (k_cache_all / v_cache_all) once per
            # generate call.  The kernel writes updated K/V values in-place on
            # device; child_memory=True skips H2D and D2H on every decode step,
            # saving ~280 ms H2D + ~360 ms D2H per step (~640 ms × 16 steps).
            # After worker.run() drains (all tasks done), a second worker.run()
            # call copies the final device state back to the host KV cache via
            # orch.copy_from so subsequent generate calls see updated values.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if "k_cache_all" not in _pname and "v_cache_all" not in _pname:
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                _kv_dev_ptrs.append((_t.data_ptr(), _dev_ptr, _nbytes))
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            _submit_l3_generate(orch, call_config, td, _keep)

            if _l3_trace_enabled:
                print(f"[L3-step] all submits done {_fmt_rel(time.perf_counter())}", flush=True)

        # ── Create Worker and execute ──

        lg_sub_fns = dict(compiled.l3_generate_sub_worker_fns or {})
        # Override the placeholder sample_and_prepare sub-worker emitted by the
        # l3_generate compiler with the real closure that reads shared-memory
        # tensors and performs argmax → embedding lookup → slot-map update.
        lg_sub_fns["sample_and_prepare"] = sample_and_prepare_fn

        num_sub = max(lg_dc.num_sub_workers, len(lg_sub_fns))

        _mark("setup_closures_and_buffers")

        worker = Worker(
            level=3,
            device_ids=[self._device_id],
            num_sub_workers=num_sub,
            platform=compiled.l3_generate_platform,
            runtime=compiled.l3_generate_runtime_name,
        )

        _mark("Worker_ctor")

        # Register chip callables and sub-worker callables before worker.init().
        lg_callable_ids: dict[str, int] = {}
        for name, callable_obj in lg_chip_callables.items():
            lg_callable_ids[name] = worker.register(callable_obj)

        sub_ids: dict[str, int] = {}
        for name, fn in lg_sub_fns.items():
            sub_ids[name] = worker.register(fn)

        _mark("worker_register")

        worker.init()
        _mark("worker_init")
        try:
            _t_run_start = time.perf_counter()
            worker.run(generate_orch_fn)
            _mark("worker_run_generate")

            # Sync KV cache back to host.  worker.run() above calls _drain()
            # internally, so all chip tasks (including the last decode step)
            # have completed by the time we reach here.  The child_memory
            # buffers are still live (worker.close() not yet called), so a
            # second worker.run() can copy them back to the host tensors.
            if _kv_dev_ptrs:
                def _kv_sync_orch_fn(orch, _args, _cfg):  # noqa: E306
                    for _host_ptr, _dev_ptr, _nbytes in _kv_dev_ptrs:
                        orch.copy_from(
                            worker_id=0, dst=_host_ptr, src=_dev_ptr, size=_nbytes,
                        )
                worker.run(_kv_sync_orch_fn)
            _mark("worker_run_kv_sync")

            _t_run_end = time.perf_counter()
            print(
                f"[L3-timer] worker.run total wall-clock: "
                f"{(_t_run_end-_t_run_start)*1000:.1f}ms",
                flush=True,
            )
        finally:
            worker.close()
            _mark("worker_close")

        # Update KV allocations.
        final_token_count = int(token_count[0].item())
        for batch_idx, alloc in enumerate(prefill_batch.kv_allocations):
            base_seq = int(prefill_inputs.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, base_seq + final_token_count)

        ids = generated_ids[:final_token_count].tolist()
        ret_val = ids, decode_out[:actual_batch].float()
        _mark("post_process")

        timer.report()
        return ret_val

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
        *,
        padded: bool = False,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        max_blocks = (
            self._compiled.prefill_block_table_stride
            if self._compiled is not None and self._compiled.prefill_block_table_stride is not None
            else self._max_blocks_per_seq(model)
        )
        if padded:
            # L3 generate reuses this block table for both prefill and decode.
            # Size it for the full reserved allocation, not only the standalone
            # prefill kernel's context-tile constant.
            if self._compiled is not None and self._compiled.decode_block_table_stride is not None:
                max_blocks = max(max_blocks, self._compiled.decode_block_table_stride)
            max_blocks = max(max_blocks, self._max_blocks_per_seq(model))
            if batch.kv_allocations:
                max_blocks = max(max_blocks, max(len(alloc.page_ids) for alloc in batch.kv_allocations))

        has_precomputed = batch.slot_mapping is not None and batch.block_table is not None

        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        seq_len_values = [int(batch.seq_lens[idx].item()) for idx in range(actual_batch)]
        total_tokens = sum(seq_len_values)
        if padded:
            hidden = torch.zeros((actual_batch, max_seq, hidden_size), dtype=torch.bfloat16)
            slot_mapping = torch.full((actual_batch * max_seq,), -1, dtype=torch.int32)
        else:
            hidden = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16)
            slot_mapping = torch.empty((total_tokens,), dtype=torch.int32)

        token_offset = 0
        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = seq_len_values[batch_idx]
            if seq_len <= 0:
                raise ValueError("prefill seq_lens must be positive")
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")
            seq_lens[batch_idx] = seq_len
            embeddings = batch.input_embeddings[batch_idx, :seq_len, :].to(torch.bfloat16).cpu()
            if padded:
                hidden[batch_idx, :seq_len, :] = embeddings
            else:
                hidden[token_offset : token_offset + seq_len, :] = embeddings

            if has_precomputed:
                bt_row = batch.block_table[batch_idx]
                valid_blocks = (bt_row >= 0).sum().item()
                block_table[
                    batch_idx * max_blocks : batch_idx * max_blocks + valid_blocks
                ] = bt_row[:valid_blocks].to(torch.int32).cpu()

                sm_row = batch.slot_mapping[batch_idx]
                valid_slots = (sm_row >= 0).sum().item()
                if padded:
                    slot_mapping[
                        batch_idx * max_seq : batch_idx * max_seq + valid_slots
                    ] = sm_row[:valid_slots].to(torch.int32).cpu()
                else:
                    slot_mapping[
                        token_offset : token_offset + seq_len
                    ] = sm_row[:seq_len].to(torch.int32).cpu()
            elif alloc is not None:
                self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
                from python.core.kv_cache import slot_mapping_for_positions as _slot_mapping_for_positions
                slot_row = _slot_mapping_for_positions(
                    alloc.page_ids,
                    seq_len,
                    self._kv_page_sizes[model.config.model_id],
                    max_tokens=max_seq,
                )
                if padded:
                    slot_mapping[batch_idx * max_seq : (batch_idx + 1) * max_seq] = slot_row
                else:
                    slot_mapping[token_offset : token_offset + seq_len] = slot_row[:seq_len]
            token_offset += seq_len

        return _PrefillInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeInputs:
        """Pack active decode requests into fused decode-kernel inputs."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        hidden_size = model.config.hidden_size
        decode_block_table_stride = (
            self._compiled.decode_block_table_stride
            if self._compiled is not None
            else None
        )
        max_blocks = decode_block_table_stride or self._max_blocks_per_seq(model)

        has_precomputed = batch.slot_mapping is not None and batch.block_table is not None

        hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.empty((actual_batch,), dtype=torch.int32)

        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )
            hidden[batch_idx, :] = batch.hidden_states[batch_idx].to(torch.bfloat16).cpu()
            seq_lens[batch_idx] = seq_len

            if has_precomputed:
                bt_row = batch.block_table[batch_idx]
                valid_blocks = (bt_row >= 0).sum().item()
                block_table[
                    batch_idx * max_blocks : batch_idx * max_blocks + valid_blocks
                ] = bt_row[:valid_blocks].to(torch.int32).cpu()
                slot_mapping[batch_idx] = batch.slot_mapping[batch_idx].to(torch.int32).cpu()
            elif alloc is not None:
                self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
                from python.core.kv_cache import slot_mapping_for_decode as _slot_mapping_for_decode
                slot_mapping[batch_idx] = _slot_mapping_for_decode(
                    alloc.page_ids, alloc.tokens_used, self._kv_page_sizes[model.config.model_id]
                )

        return _DecodeInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )

    @staticmethod
    def _write_block_table_row(
        block_table: torch.Tensor,
        batch_idx: int,
        max_blocks: int,
        alloc: KvAllocation,
    ) -> None:
        """Write one request's KV page IDs into a flat block table."""
        row_start = batch_idx * max_blocks
        if alloc.page_ids:
            block_table[row_start : row_start + len(alloc.page_ids)] = torch.tensor(
                alloc.page_ids,
                dtype=torch.int32,
            )

    @staticmethod
    def _validate_batch_size(
        model: RuntimeModel,
        actual_batch: int,
    ) -> int:
        """Validate and return the actual user batch size."""
        if actual_batch <= 0:
            raise ValueError("batch must contain at least one request")
        if actual_batch > model.runtime.max_batch_size:
            max_batch_size = model.runtime.max_batch_size
            raise ValueError(
                f"batch has {actual_batch} requests, but runtime max_batch_size is {max_batch_size}"
            )
        return actual_batch

    @staticmethod
    def _max_blocks_per_seq(model: RuntimeModel) -> int:
        """Return the maximum KV pages one sequence can occupy."""
        return (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size
