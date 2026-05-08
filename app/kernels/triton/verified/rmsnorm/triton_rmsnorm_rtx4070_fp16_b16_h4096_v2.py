"""
Triton RMSNorm kernel for fp16 on RTX 4070.

Computes: y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight

Conventions:
  * Eps inside the sqrt (PyTorch / HF / Apex convention).
  * Weight is per-feature, shape (H,). No bias.
  * Accumulation in fp32 regardless of input dtype.
  * Persistent single-block pattern when H <= MAX_PERSISTENT_BLOCK (8192);
    the row is loaded once into registers, reduced, scaled, and stored
    without a second HBM read.
  * Two-pass fallback for very large H.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Persistent (single-tile) kernel — used when H <= BLOCK_SIZE.
# -------------------------------------------------------------------------
@triton.jit
def _rmsnorm_persistent_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    x_row_stride,
    y_row_stride,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x_ptrs = x_ptr + row * x_row_stride + cols
    # Load once, cast to fp32 before squaring.
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Sum of squares with fp32 accumulation, denominator is the full row H.
    mean_sq = tl.sum(x * x, axis=0) / H
    rrms = tl.rsqrt(mean_sq + eps)

    # Per-feature weight, broadcasts across rows. No row offset.
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = x * rrms * w

    y_ptrs = y_ptr + row * y_row_stride + cols
    tl.store(y_ptrs, y, mask=mask)


# -------------------------------------------------------------------------
# Two-pass kernel — used when H exceeds the persistent cap.
# -------------------------------------------------------------------------
@triton.jit
def _rmsnorm_twopass_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    x_row_stride,
    y_row_stride,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: accumulate sum of squares in fp32.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    mean_sq = tl.sum(acc, axis=0) / H
    rrms = tl.rsqrt(mean_sq + eps)

    # Pass 2: scale and store.
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(y_ptr + row * y_row_stride + cols, y, mask=mask)


# -------------------------------------------------------------------------
# Python wrapper.
# -------------------------------------------------------------------------
_MAX_PERSISTENT_BLOCK = 8192


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension.

    y = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight
    """
    assert x.is_cuda, "input must be on CUDA"
    assert weight.is_cuda, "weight must be on CUDA"
    assert weight.ndim == 1, "weight must be 1D of shape (H,)"
    assert x.shape[-1] == weight.shape[0], "last dim of x must equal weight size"

    orig_shape = x.shape
    H = orig_shape[-1]
    x2d = x.reshape(-1, H).contiguous()
    n_rows = x2d.shape[0]

    y = torch.empty_like(x2d)

    # Pick BLOCK_SIZE.
    if H <= _MAX_PERSISTENT_BLOCK:
        BLOCK_SIZE = _next_pow2(H)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)

        # num_warps heuristic: bigger blocks use more warps.
        if BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 4096:
            num_warps = 8
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        elif BLOCK_SIZE >= 1024:
            num_warps = 4
        else:
            num_warps = 4

        grid = (n_rows,)
        _rmsnorm_persistent_kernel[grid](
            x2d, y, weight,
            x2d.stride(0), y.stride(0),
            H, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        BLOCK_SIZE = 2048
        grid = (n_rows,)
        _rmsnorm_twopass_kernel[grid](
            x2d, y, weight,
            x2d.stride(0), y.stride(0),
            H, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=1,
        )

    return y.reshape(orig_shape)
