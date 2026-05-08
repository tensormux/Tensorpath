"""NeevPath Forge — kernel optimization loop.

Forge is the execution layer that retrieves skill bundles from
`@krxgu/kernel-skills`, builds agent-ready prompts, accepts candidate kernels,
verifies correctness, benchmarks performance, and promotes only verified
kernels into the local registry.

Forge does not trust generated code. Generated code stays under
`forge_runs/<run_id>/candidate/` and is never imported by the running app.
Only kernels promoted into `app/kernels/triton/verified/` (and recorded in
`kernel_registry/verified_kernels.json`) are allowed to surface in
recommendation results.
"""
