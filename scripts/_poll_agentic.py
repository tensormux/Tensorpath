"""Internal polling helper for the agentic smoke test. Not committed."""

from __future__ import annotations

import json
import sys
import time
import urllib.request

RUN = sys.argv[1] if len(sys.argv) > 1 else "20260508T045548_rmsnorm_rtx-4070"
URL = f"http://127.0.0.1:8000/api/forge/runs/{RUN}/agentic"
TERMINAL = {"succeeded", "rejected", "aborted", "errored"}

for i in range(60):
    try:
        with urllib.request.urlopen(URL) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f"[{i}] fetch error: {e}")
        time.sleep(5)
        continue
    if not d:
        print(f"[{i}] state=null")
        time.sleep(5)
        continue

    status = d.get("status")
    iter_n = d.get("iteration")
    iter_m = d.get("max_iterations")
    cost = float(d.get("cost_usd", 0))
    v = d.get("last_verify_passed")
    b = d.get("last_benchmark_passed")
    s = d.get("last_speedup")
    msg = (d.get("last_message") or "")[:70]

    print(
        f"[{i:2d}] status={status}  iter={iter_n}/{iter_m}  cost=${cost:.3f}  "
        f"v={v}  b={b}  s={s}  msg={msg!r}"
    )
    if status in TERMINAL:
        print("--- TERMINAL ---")
        if d.get("error"):
            print(f"error: {d['error']}")
        if d.get("promoted_kernel_id"):
            print(f"promoted: {d['promoted_kernel_id']}")
        break
    time.sleep(5)
