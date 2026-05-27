"""INT-N KV cache compressor using per-row absmax quantization.

Supports configurable bit widths (2, 3, 4, 8, etc.):
  Quantize:  q = clamp(floor(x * inv_scale + offset), 0, n_levels-1)
  Dequantize: x_hat = (q - offset) * scale
where inv_scale = (n_levels-1) / (2 * max_abs), offset = (n_levels-1) / 2.
Sub-8-bit values are bit-packed for storage efficiency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class AbsmaxCompressor:
    """Per-row absmax INT-N compressor for one side (keys or values).

    Supports bits in {2, 3, 4, 8}. Sub-8-bit values are bit-packed.
    """

    def __init__(self, head_dim: int, bits: int = 8, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = min(bits, 8)
        self.device = device
        self.n_levels = 2 ** self.bits
        self.half_levels = (self.n_levels - 1) / 2.0

    def _bit_pack(self, indices: torch.Tensor, N: int, D: int) -> tuple[torch.Tensor, int]:
        """Pack UINT8 indices into bit-packed bytes for sub-8-bit."""
        if self.bits >= 8:
            # No packing needed for 8-bit
            return indices.reshape(N, D), 0

        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_flat.device,
        )
        idx_bytes = (
            (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers)
            .sum(-1)
            .to(torch.uint8)
        )
        return idx_bytes, idx_pad

    def _bit_unpack(self, idx_bytes: torch.Tensor, N: int, D: int, idx_pad: int) -> torch.Tensor:
        """Unpack bit-packed bytes back to UINT8 indices."""
        if self.bits >= 8:
            return idx_bytes.reshape(N, D)

        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = (
            (idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask
        ).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]
        return indices.to(torch.uint8)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D) -> dict with indices + per-row scales."""
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        max_abs = flat.abs().amax(dim=-1).clamp(min=1e-8)  # [N]
        inv_scales = ((self.n_levels - 1) / (2.0 * max_abs)).unsqueeze(-1)  # [N, 1]
        scaled = flat * inv_scales + self.half_levels
        indices = torch.clamp(scaled.floor().long(), 0, self.n_levels - 1).to(torch.uint8)

        idx_bytes, idx_pad = self._bit_pack(indices, N, D)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, -1) if self.bits < 8 else idx_bytes.reshape(B, H, S, D),
            "scales": max_abs.half().reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
            "bits": self.bits,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        B, H, S, D = compressed["shape"]
        N = B * H * S
        bits = compressed.get("bits", self.bits)

        if bits < 8:
            n_groups = compressed["idx_bytes"].shape[-1]
            indices = self._bit_unpack(
                compressed["idx_bytes"].reshape(N, n_groups), N, D, compressed["idx_pad"]
            )
        else:
            indices = compressed["idx_bytes"].reshape(N, D)

        q = indices.float()
        n_levels = 2 ** bits
        half_levels = (n_levels - 1) / 2.0
        max_abs = compressed["scales"].reshape(N).float()
        scale_per_level = (2.0 * max_abs / (n_levels - 1)).unsqueeze(-1)  # [N, 1]

        reconstructed = (q - half_levels) * scale_per_level
        return reconstructed.to(torch.bfloat16).reshape(B, H, S, D)

    @torch.no_grad()
    def compress_npu(
        self, states: torch.Tensor, run_kv_quantize_fn
    ) -> dict:
        """Compress using NPU kernel for quantization, Python for scale + bit-pack.

        Args:
            states: (B, H, S, D) BF16 tensor.
            run_kv_quantize_fn: callable(flat_kv, inv_scales, offset) -> indices_uint8
        """
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D)

        max_abs = flat.float().abs().amax(dim=-1).clamp(min=1e-8)  # [N]
        inv_scales = ((self.n_levels - 1) / (2.0 * max_abs)).unsqueeze(-1).contiguous()  # [N, 1]

        indices = run_kv_quantize_fn(
            flat_kv=flat.contiguous(),
            inv_scales=inv_scales,
            offset=self.half_levels,
        )

        idx_bytes, idx_pad = self._bit_pack(indices, N, D)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, -1) if self.bits < 8 else idx_bytes.reshape(B, H, S, D),
            "scales": max_abs.half().reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
            "bits": self.bits,
        }

    @torch.no_grad()
    def decompress_npu(
        self, compressed: dict, run_kv_dequantize_fn
    ) -> torch.Tensor:
        """Decompress using NPU kernel for dequantization.

        Args:
            compressed: dict from compress_npu().
            run_kv_dequantize_fn: callable(indices, scales, offset) -> reconstructed_bf16
        """
        B, H, S, D = compressed["shape"]
        N = B * H * S
        bits = compressed.get("bits", self.bits)
        n_levels = 2 ** bits
        half_levels = (n_levels - 1) / 2.0

        if bits < 8:
            n_groups = compressed["idx_bytes"].shape[-1]
            indices = self._bit_unpack(
                compressed["idx_bytes"].reshape(N, n_groups), N, D, compressed["idx_pad"]
            )
        else:
            indices = compressed["idx_bytes"].reshape(N, D)

        max_abs = compressed["scales"].reshape(N).float()
        scale_per_level = (2.0 * max_abs / (n_levels - 1)).unsqueeze(-1).contiguous()  # [N, 1]

        reconstructed = run_kv_dequantize_fn(
            indices=indices.contiguous(),
            scales=scale_per_level,
            offset=half_levels,
        )
        return reconstructed.reshape(B, H, S, D)


@dataclass
class KvQuantConfig:
    """Configuration for KV cache quantization."""

    enabled: bool = False
    key_bits: int = 4
    value_bits: int = 4
    residual_window: int = 128
    protected_layers: int = 4
    protected_bits: int = 8


class KVCompressor:
    """Per-layer KV cache compressor with configurable INT-N quantization.

    Each layer gets its own AbsmaxCompressor with configurable bit width,
    enabling layer-adaptive precision.
    """

    def __init__(
        self,
        head_dim: int,
        num_layers: int,
        config: KvQuantConfig,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.config = config

        self.key_compressors: list[AbsmaxCompressor] = []
        self.val_compressors: list[AbsmaxCompressor] = []

        print(f"[KV-Quant] Initializing: {num_layers} layers, head_dim={head_dim}, "
              f"key_bits={config.key_bits}, val_bits={config.value_bits}", flush=True)
        t_total = time.perf_counter()
        for layer_idx in range(num_layers):
            is_protected = (
                layer_idx < config.protected_layers
                or layer_idx >= (num_layers - config.protected_layers)
            )
            effective_key_bits = config.protected_bits if is_protected else config.key_bits
            effective_val_bits = config.protected_bits if is_protected else config.value_bits
            effective_key_bits = min(effective_key_bits, 8)
            effective_val_bits = min(effective_val_bits, 8)

            self.key_compressors.append(
                AbsmaxCompressor(head_dim, effective_key_bits, device=device)
            )
            self.val_compressors.append(
                AbsmaxCompressor(head_dim, effective_val_bits, device=device)
            )
            print(f"[KV-Quant]   layer {layer_idx}: key_bits={effective_key_bits}, val_bits={effective_val_bits}", flush=True)
        dt_total = (time.perf_counter() - t_total) * 1000
        print(f"[KV-Quant] Initialization complete: {dt_total:.1f} ms", flush=True)

    @torch.no_grad()
    def compress_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Compress keys/values for one layer (Python fallback).

        Args:
            keys: (S, H, D) -- old tokens to compress
            values: (S, H, D) -- old tokens to compress

        Returns:
            (compressed_k, compressed_v) dicts
        """
        t0 = time.perf_counter()
        keys_4d = keys.permute(1, 0, 2).unsqueeze(0)
        values_4d = values.permute(1, 0, 2).unsqueeze(0)

        compressed_k = self.key_compressors[layer_idx].compress(keys_4d)
        compressed_v = self.val_compressors[layer_idx].compress(values_4d)
        print(f"[KV-Quant] compress_layer {layer_idx} (python): {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return compressed_k, compressed_v

    @torch.no_grad()
    def compress_layer_npu(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        run_kv_quantize_fn,
    ) -> tuple[dict, dict]:
        """Compress keys/values for one layer using NPU kernels.

        Args:
            keys: (S, H, D) -- old tokens to compress
            values: (S, H, D) -- old tokens to compress
            run_kv_quantize_fn: NPU quantize callable

        Returns:
            (compressed_k, compressed_v) dicts
        """
        t0 = time.perf_counter()
        keys_4d = keys.permute(1, 0, 2).unsqueeze(0)
        values_4d = values.permute(1, 0, 2).unsqueeze(0)

        compressed_k = self.key_compressors[layer_idx].compress_npu(keys_4d, run_kv_quantize_fn)
        compressed_v = self.val_compressors[layer_idx].compress_npu(values_4d, run_kv_quantize_fn)
        print(f"[KV-Quant] compress_layer {layer_idx} (npu): {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return compressed_k, compressed_v

    @torch.no_grad()
    def decompress_layer(
        self,
        layer_idx: int,
        compressed_k: dict,
        compressed_v: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress keys/values for one layer (Python fallback).

        Returns:
            keys: (S, H, D), values: (S, H, D)
        """
        t0 = time.perf_counter()
        keys_4d = self.key_compressors[layer_idx].decompress(compressed_k)
        values_4d = self.val_compressors[layer_idx].decompress(compressed_v)

        keys = keys_4d.squeeze(0).permute(1, 0, 2)
        values = values_4d.squeeze(0).permute(1, 0, 2)
        print(f"[KV-Quant] decompress_layer {layer_idx} (python): {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return keys, values

    @torch.no_grad()
    def decompress_layer_npu(
        self,
        layer_idx: int,
        compressed_k: dict,
        compressed_v: dict,
        run_kv_dequantize_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress keys/values for one layer using NPU kernels.

        Returns:
            keys: (S, H, D), values: (S, H, D)
        """
        t0 = time.perf_counter()
        keys_4d = self.key_compressors[layer_idx].decompress_npu(compressed_k, run_kv_dequantize_fn)
        values_4d = self.val_compressors[layer_idx].decompress_npu(compressed_v, run_kv_dequantize_fn)

        keys = keys_4d.squeeze(0).permute(1, 0, 2)
        values = values_4d.squeeze(0).permute(1, 0, 2)
        print(f"[KV-Quant] decompress_layer {layer_idx} (npu): {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return keys, values
