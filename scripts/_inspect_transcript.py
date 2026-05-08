"""Pretty-print an agentic transcript JSON. Internal smoke-test helper."""

import json
import sys
import urllib.request

run_id = sys.argv[1] if len(sys.argv) > 1 else "20260508T045548_rmsnorm_rtx-4070"
url = f"http://127.0.0.1:8000/api/forge/runs/{run_id}/agentic/transcript?tail=300"

with urllib.request.urlopen(url) as r:
    data = json.loads(r.read())

print(f"transcript: {data['count']} entries\n")

for e in data["entries"]:
    kind = e.get("kind", "?")
    ts = e.get("at", "")[11:19]
    if kind == "model_turn":
        t = e.get("tokens", {})
        print(
            f"[{ts}] model_turn iter={e['iteration']}  stop={e['stop_reason']}  "
            f"tok in/out/cw/cr={t.get('input_tokens')}/{t.get('output_tokens')}/"
            f"{t.get('cache_write_tokens')}/{t.get('cache_read_tokens')}  "
            f"${e['turn_cost_usd']} cum=${e['cumulative_cost_usd']}"
        )
    elif kind == "tool_call":
        name = e.get("name", "")
        inp = json.dumps(e.get("input", {}))[:90]
        prev = (e.get("content_preview", "") or "").replace("\n", " ")[:120]
        err = " [ERR]" if e.get("is_error") else ""
        print(f"[{ts}] tool {name:22s}{err}  in={inp}")
        print(f"          out={prev}")
    elif kind == "promotion":
        print(f"[{ts}] PROMOTED {e['kernel_id']} speedup={e['speedup']}x")
    else:
        print(f"[{ts}] {kind}: {json.dumps(e)[:140]}")
