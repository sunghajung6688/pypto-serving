"""TurboQuant KV cache compressor: random rotation + fixed-range quantization.

Implements the TurboQuant algorithm from the paper:
  1. L2 normalize each KV vector, store norms
  2. Rotate: normalized_vector @ Pi  (random orthogonal matrix)
  3. Uniform quantize with fixed range derived from N(0, 1/d):
     floor((rotated - lo) * inv_step), clamp to [0, 2^bits - 1]

After random rotation, each coordinate follows N(0, 1/d). The fixed
quantization range is [-3.5/sqrt(d), +3.5/sqrt(d)], which is a known
constant determined at initialization — no per-row dynamic scale needed.

Sub-8-bit values are bit-packed for storage efficiency.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from ..types import KvQuantConfig
from .lloyd_max import LloydMaxCodebook


def generate_rotation_matrix(d: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).to(device)


class TurboQuantCompressor:
    """Per-vector TurboQuant compressor: rotate + Lloyd-Max quantize.

    After normalizing to unit sphere and applying random rotation,
    each coordinate follows N(0, 1/d). We quantize using Lloyd-Max
    optimal centroids for this Gaussian distribution (bits < 8),
    or uniform quantization for bits >= 8.

    Supports bits in {2, 3, 4, 8}. Sub-8-bit values are bit-packed.
    """

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = min(bits, 8)
        self.device = device
        self.n_levels = 2 ** self.bits

        # Rotation matrix (fixed at init).
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.PiT = self.Pi.T.contiguous()

        # Quantization range for normalized vectors (~N(0, 1/d)).
        sigma = 1.0 / math.sqrt(head_dim)
        self.lo = -3.5 * sigma
        self.hi = 3.5 * sigma

        if self.bits < 8:
            # Lloyd-Max optimal centroids/boundaries for N(0, 1/d).
            codebook = LloydMaxCodebook(head_dim, self.bits)
            self.centroids = codebook.centroids      # (n_levels,) sorted FP32
            self.boundaries = codebook.boundaries    # (n_levels-1,) sorted FP32
        else:
            # 8-bit: uniform quantization (Lloyd-Max has negligible benefit).
            self.centroids = torch.linspace(self.lo, self.hi, self.n_levels, dtype=torch.float32)
            step = (self.hi - self.lo) / (self.n_levels - 1)
            self.boundaries = torch.linspace(self.lo + step / 2, self.hi - step / 2, self.n_levels - 1, dtype=torch.float32)

    def _bit_pack(self, indices: torch.Tensor, N: int, D: int) -> tuple[torch.Tensor, int]:
        """Pack UINT8 indices into bit-packed bytes for sub-8-bit."""
        if self.bits >= 8:
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
        """Compress (B, H, S, D) -> dict with indices + per-vector norms.

        Pipeline: normalize -> rotate -> fixed-range quantize -> bit-pack.
        """
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        # L2 normalize, store norms.
        vec_norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        flat_norm = flat / vec_norms

        # Random rotation.
        rotated = flat_norm @ self.Pi.T

        # Nearest-centroid quantize via boundary search.
        indices = torch.searchsorted(self.boundaries, rotated, right=True)
        indices = torch.clamp(indices, 0, self.n_levels - 1).to(torch.uint8)

        idx_bytes, idx_pad = self._bit_pack(indices, N, D)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, -1) if self.bits < 8 else idx_bytes.reshape(B, H, S, D),
            "norms": vec_norms.squeeze(-1).half().reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
            "bits": self.bits,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor.

        Pipeline: bit-unpack -> centroid lookup -> inverse rotate -> scale by norms.
        """
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

        # Centroid lookup (uniform grid).
        centroid_values = self.centroids.to(indices.device)
        reconstructed = centroid_values[indices.long()]  # (N, D) in rotated space

        # Inverse rotation.
        reconstructed = reconstructed @ self.Pi

        # Scale by norms.
        vec_norms = compressed["norms"].reshape(N, 1).float()
        reconstructed = reconstructed * vec_norms

        return reconstructed.to(torch.bfloat16).reshape(B, H, S, D)

    @torch.no_grad()
    def compress_npu(
        self, states: torch.Tensor, run_tq_compress_fn
    ) -> dict:
        """Compress using NPU tq_compress kernel + CPU-side Lloyd-Max quantize."""
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        # L2 normalize + rotate on CPU (matching compress() pipeline).
        vec_norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        flat_norm = flat / vec_norms
        rot = self.Pi.to(torch.float32)
        rotated = flat_norm @ rot.T

        # Lloyd-Max boundary search.
        indices = torch.searchsorted(self.boundaries, rotated, right=True)
        indices = torch.clamp(indices, 0, self.n_levels - 1).to(torch.uint8)

        idx_bytes, idx_pad = self._bit_pack(indices, N, D)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, -1) if self.bits < 8 else idx_bytes.reshape(B, H, S, D),
            "norms": vec_norms.squeeze(-1).half().reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
            "bits": self.bits,
        }

    @torch.no_grad()
    def decompress_npu(
        self, compressed: dict, run_tq_decompress_fn
    ) -> torch.Tensor:
        """Decompress using NPU tq_decompress kernel with inverse rotation."""
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

        # Centroid lookup on CPU.
        centroid_values = self.centroids.to(indices.device)
        centroid_vals = centroid_values[indices.long()].to(torch.bfloat16)  # (N, D)

        vec_norms = compressed["norms"].reshape(N, 1)

        rot = self.Pi.to(torch.bfloat16).to(centroid_vals.device)

        reconstructed = run_tq_decompress_fn(
            centroid_vals=centroid_vals.contiguous(),
            rot_matrix=rot,
            norms=vec_norms.contiguous(),
        )
        return reconstructed.reshape(B, H, S, D)


class KVCompressor:
    """Per-layer KV cache compressor with TurboQuant (rotation + fixed-range).

    Each layer gets its own TurboQuantCompressor with a unique rotation
    matrix and configurable bit width, enabling layer-adaptive precision.
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

        self.key_compressors: list[TurboQuantCompressor] = []
        self.val_compressors: list[TurboQuantCompressor] = []

        print(f"[TQ] Initializing: {num_layers} layers, head_dim={head_dim}, "
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
                TurboQuantCompressor(head_dim, effective_key_bits,
                                     seed=seed + layer_idx * 1000, device=device)
            )
            self.val_compressors.append(
                TurboQuantCompressor(head_dim, effective_val_bits,
                                     seed=seed + layer_idx * 1000, device=device)
            )
            print(f"[TQ]   layer {layer_idx}: key_bits={effective_key_bits}, "
                  f"val_bits={effective_val_bits}", flush=True)
        dt_total = (time.perf_counter() - t_total) * 1000
        print(f"[TQ] Initialization complete: {dt_total:.1f} ms", flush=True)

    def get_rot_matrices(self, device: str = "cpu") -> torch.Tensor:
        """Stack all per-layer rotation matrices for NPU upload.

        Returns: (num_layers, head_dim, head_dim) BF16 tensor.
        """
        return torch.stack([c.Pi for c in self.key_compressors]).bfloat16().to(device)

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
        print(f"[TQ] compress_layer {layer_idx} (python): "
              f"{(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return compressed_k, compressed_v

    @torch.no_grad()
    def compress_layer_npu(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        run_tq_compress_fn,
    ) -> tuple[dict, dict]:
        """Compress keys/values for one layer using NPU tq_compress kernel.

        Args:
            keys: (S, H, D) -- old tokens to compress
            values: (S, H, D) -- old tokens to compress
            run_tq_compress_fn: NPU compress callable

        Returns:
            (compressed_k, compressed_v) dicts
        """
        t0 = time.perf_counter()
        keys_4d = keys.permute(1, 0, 2).unsqueeze(0)
        values_4d = values.permute(1, 0, 2).unsqueeze(0)

        compressed_k = self.key_compressors[layer_idx].compress_npu(keys_4d, run_tq_compress_fn)
        compressed_v = self.val_compressors[layer_idx].compress_npu(values_4d, run_tq_compress_fn)
        print(f"[TQ] compress_layer {layer_idx} (npu): "
              f"{(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
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
        print(f"[TQ] decompress_layer {layer_idx} (python): "
              f"{(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return keys, values

    @torch.no_grad()
    def decompress_layer_npu(
        self,
        layer_idx: int,
        compressed_k: dict,
        compressed_v: dict,
        run_tq_decompress_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress keys/values for one layer using NPU tq_decompress kernel.

        Returns:
            keys: (S, H, D), values: (S, H, D)
        """
        t0 = time.perf_counter()
        keys_4d = self.key_compressors[layer_idx].decompress_npu(compressed_k, run_tq_decompress_fn)
        values_4d = self.val_compressors[layer_idx].decompress_npu(compressed_v, run_tq_decompress_fn)

        keys = keys_4d.squeeze(0).permute(1, 0, 2)
        values = values_4d.squeeze(0).permute(1, 0, 2)
        print(f"[TQ] decompress_layer {layer_idx} (npu): "
              f"{(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
        return keys, values
