"""PyTorch reference for RMSNorm.

This is the *correctness anchor* for any RMSNorm kernel candidate Forge
verifies. Critical: do NOT use `torch.nn.LayerNorm` — LayerNorm subtracts
the mean before dividing, RMSNorm does not. The two are not interchangeable
references.

Math:

    y = x * rsqrt(mean(x², axis=-1) + eps) * weight

The reduction is computed in fp32 even when inputs are fp16/bf16, because
squaring fp16 can overflow / underflow at the values that show up in
real LLM activations. Candidates are expected to do the same; tests may
fail tolerance checks otherwise.
"""

from __future__ import annotations

import torch


def torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm reference. Accumulates the mean-of-squares in fp32.

    Args:
        x: shape `(..., hidden_size)`, any floating dtype.
        weight: shape `(hidden_size,)`, broadcast over leading dims.
        eps: small constant added inside the rsqrt for numerical stability.

    Returns:
        Tensor with the same shape and dtype as `x`.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + eps)
    return (x_norm * weight.to(torch.float32)).to(input_dtype)
