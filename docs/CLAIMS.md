# Allowed and disallowed claims

The point of this document: keep the marketing language and the engineering
language in agreement. NeevPath is supposed to be ambitious *and* credible.
The line between the two is what's measured vs. what's wished for.

## Allowed

> NeevPath can generate skill-guided kernel optimization tasks and promote
> verified kernels after correctness and benchmark checks.

> This RMSNorm candidate is **1.69× faster than the PyTorch baseline for
> shape X on GPU Y**. Op-level evidence; microbenchmark only.

> A verified kernel for this op/GPU/dtype combination is available in the
> registry. The recommendation engine surfaces this as an **optimization
> opportunity** but does not adjust end-to-end latency or cost estimates.

> kernel-skills is consumed as a version-pinned external npm package. It
> provides expert instruction content; it does not run code.

> NeevPath Forge runs every promoted kernel through correctness tests
> (PyTorch reference) and a microbenchmark with explicit warmup, CUDA
> synchronization, and a 1.10× minimum speedup threshold.

## Not allowed

> ❌ NeevPath automatically optimizes every kernel in every model.

We optimize one op per Forge run, and only when a developer or agent
produces a candidate that passes both gates.

> ❌ This model is 1.69× faster end-to-end because an RMSNorm
> microbenchmark is 1.69× faster.

A single op being faster does not imply a model is end-to-end faster.
Op-level evidence stays op-level until a runtime-level or endpoint-level
benchmark proves the integration.

> ❌ Generated kernels are production-ready without runtime-level or
> endpoint-level validation.

Promoted kernels are metadata + source artifacts. Wiring them into a serving
runtime, then benchmarking the full endpoint, is a separate step.

> ❌ NeevPath synthesizes CUDA kernels from scratch.

NeevPath productizes existing optimized kernel paths and verifies new ones
that humans or agents write. It does not do compiler-level synthesis.

> ❌ This benchmark proves the kernel is faster than vendor libraries.

A 1.10–2.0× win against a *PyTorch eager baseline* says nothing about
TensorRT-LLM, cuDNN, FlashAttention-3, or other tuned implementations.
Our threshold is "faster than the obvious reference", not "fastest in
existence".

## When in doubt

If a claim depends on a number, name the measurement that produced it. If
no measurement exists, don't make the claim.
