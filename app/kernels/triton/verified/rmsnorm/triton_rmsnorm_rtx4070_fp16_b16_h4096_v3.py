"""
Triton RMSNorm kernel optimized for RTX 4070, fp16, shape (16, 4096).

Math: y = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight
- No mean subtraction (RMSNorm, not LayerNorm)
- No bias
- Eps inside sqrt (PyTorch convention)
- fp32 accumulation; fp16 IO
- Persistent single-pass pattern: one program per row, full row in registers
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,              # *fp16
    y_ptr,              # *fp16
    w_ptr,              # *fp16 weight, shape (H,)
    x_row_stride,       # int
    y_row_stride,       # int
    H,                  # int (run-time)
    eps,                # fp32
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < H

    x_row_ptr = x_ptr + row_idx * x_row_stride
    y_row_ptr = y_ptr + row_idx * y_row_stride

    # Load row, cast to fp32 immediately (other=0.0 contributes 0 to sum-of-squares)
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean of squares in fp32, divide by full H (masked lanes are 0)
    mean_sq = tl.sum(x * x, axis=0) / H
    rrms = tl.rsqrt(mean_sq + eps)

    # Weight is 1D shape (H,) — col_offsets only, no row offset
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    y = x * rrms * w

    tl.store(y_row_ptr + col_offsets, y.to(y_row_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute RMSNorm over the last dim of x using `weight` (shape (H,)).
    Returns a new tensor with the same shape and dtype as x.
    """
    assert x.is_cuda and weight.is_cuda, "inputs must be CUDA"
    assert weight.ndim == 1, "weight must be 1D"
    H = x.shape[-1]
    assert weight.shape[0] == H, "weight dim must match last dim of x"

    # Flatten leading dims to (N_rows, H)
    x_2d = x.reshape(-1, H)
    # Ensure last-dim contiguous so the row stride trick is safe
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()

    N_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    # Persistent single-pass kernel: BLOCK_SIZE = next pow2 >= H
    BLOCK_SIZE = _next_pow2(H)

    if BLOCK_SIZE <= 1024:
        num_warps = 4
    elif BLOCK_SIZE <= 2048:
        num_warps = 8
    elif BLOCK_SIZE <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    grid = (N_rows,)
    _rmsnorm_fwd_kernel[grid](
        x_2d,
        y,
        weight,
        x_2d.stride(0),
        y.stride(0),
        H,
        float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )

    return y.reshape(x.shape)
