"""Benchmark naive vs sliding-window context and export chart-ready results.

Runs a multi-turn conversation against a local Ollama model under both
strategies and writes:

    <out>.csv    per-turn rows (open in Excel / Sheets / pandas)
    <out>.json   same data + config, for programmatic charting
    <out>.html   self-contained chart page (no internet needed)

Run:
    python benchmark_charts.py --turns 8
    python benchmark_charts.py --turns 12 --max-tokens 128 --num-predict 64
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from typing import List

from local_context import QUESTIONS, SAMPLE_DOCUMENT
from local_llm import LocalLLM
from sliding_context import SlidingWindowContext

FIELDS = [
    "turn", "strategy", "prompt_tokens", "prompt_eval_ms", "eval_ms",
    "wall_ms", "window_total_tokens", "compressions",
]


def build_questions(base: List[str], n: int) -> List[str]:
    out = []
    for i in range(n):
        q = base[i % len(base)]
        part = i // len(base) + 1
        out.append(q if part == 1 else f"{q} (follow-up {part})")
    return out


def run_strategy(llm, corpus, questions, managed, max_tokens, summary_tokens,
                 num_predict) -> List[dict]:
    rows: List[dict] = []
    if managed:
        ctx = SlidingWindowContext(max_tokens, summary_tokens)
        ctx.add(f"[Document]\n{corpus}")
    history = [] if managed else [f"[Document]\n{corpus}"]

    for i, q in enumerate(questions, 1):
        if managed:
            ctx.add(f"User: {q}")
            prompt = ctx.render() + "\nAssistant:"
        else:
            prompt = "\n\n".join(history + [f"User: {q}"]) + "\nAssistant:"

        m = llm.generate_full(prompt, max_tokens=num_predict)
        st = ctx.stats() if managed else None
        rows.append({
            "turn": i,
            "strategy": "managed" if managed else "naive",
            "prompt_tokens": m["prompt_tokens"],
            "prompt_eval_ms": round(m["prompt_eval_ms"], 1),
            "eval_ms": round(m["eval_ms"], 1),
            "wall_ms": round(m["wall_ms"], 1),
            "window_total_tokens": st["total_tokens"] if st else m["prompt_tokens"],
            "compressions": st["compressions"] if st else 0,
        })

        answer = m["response"]
        if managed:
            ctx.add(f"Assistant: {answer}")
        else:
            history.append(f"User: {q}")
            history.append(f"Assistant: {answer}")

        print(f"    turn {i:>2}: {rows[-1]['strategy']:<8} "
              f"prompt={m['prompt_tokens']:>4}  read={m['prompt_eval_ms']:>7.0f}ms  "
              f"gen={m['eval_ms']:>7.0f}ms")
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Context benchmark</title>
<style>
 body{font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;background:#0f1117;
      color:#e6e8ee;margin:0;padding:24px;}
 h1{font-size:18px;} h2{font-size:14px;color:#9aa3b8;margin-top:28px;}
 .wrap{display:flex;flex-wrap:wrap;gap:24px;}
 .card{background:#171a23;border:1px solid #2a2f3a;border-radius:10px;padding:14px;}
 table{border-collapse:collapse;font-size:13px;margin-top:10px;}
 th,td{border:1px solid #2a2f3a;padding:6px 12px;text-align:right;}
 th:first-child,td:first-child{text-align:left;}
 .naive{color:#ff6b6b;} .managed{color:#4dd0e1;}
 .legend span{margin-right:16px;font-size:12px;}
</style></head><body>
<h1>Context benchmark &mdash; __MODEL__</h1>
<div class="legend"><span class="naive">&#9632; naive</span>
<span class="managed">&#9632; managed</span></div>
<div class="wrap">
  <div class="card"><h2>Prompt tokens per turn</h2><canvas id="c1" width="520" height="300"></canvas></div>
  <div class="card"><h2>Prompt read time (ms) per turn</h2><canvas id="c2" width="520" height="300"></canvas></div>
  <div class="card"><h2>Managed window size per turn</h2><canvas id="c3" width="520" height="300"></canvas></div>
</div>
<h2>Totals</h2>
<table id="totals"></table>
<script>
const DATA = __DATA__;
const naive = DATA.rows.filter(r=>r.strategy==="naive");
const managed = DATA.rows.filter(r=>r.strategy==="managed");
const labels = naive.map(r=>"t"+r.turn);

function chart(id, series, ylabel){
  const cv=document.getElementById(id), ctx=cv.getContext("2d");
  const W=cv.width,H=cv.height,pad={l:56,r:14,t:16,b:32};
  const iw=W-pad.l-pad.r, ih=H-pad.t-pad.b;
  let max=0; series.forEach(s=>s.data.forEach(v=>{if(v>max)max=v;}));
  max=max*1.15||1;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="#2a2f3a"; ctx.fillStyle="#8b93a7"; ctx.font="11px sans-serif";
  for(let g=0; g<=4; g++){
    const y=pad.t+ih*g/4, val=Math.round(max*(1-g/4));
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
    ctx.fillText(val, 6, y+4);
  }
  ctx.fillText(ylabel, pad.l, 11);
  const n=labels.length;
  const X=i=> pad.l + (n<=1?iw/2:iw*i/(n-1));
  const Y=v=> pad.t + ih*(1 - v/max);
  ctx.fillStyle="#8b93a7";
  labels.forEach((lb,i)=>{ if(n<=12||i%2===0) ctx.fillText(lb, X(i)-8, H-12); });
  series.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath();
    s.data.forEach((v,i)=>{ const x=X(i),y=Y(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    ctx.fillStyle=s.color;
    s.data.forEach((v,i)=>{ ctx.beginPath(); ctx.arc(X(i),Y(v),2.5,0,7); ctx.fill(); });
  });
}
chart("c1",[{color:"#ff6b6b",data:naive.map(r=>r.prompt_tokens)},
            {color:"#4dd0e1",data:managed.map(r=>r.prompt_tokens)}],"tokens");
chart("c2",[{color:"#ff6b6b",data:naive.map(r=>r.prompt_eval_ms)},
            {color:"#4dd0e1",data:managed.map(r=>r.prompt_eval_ms)}],"ms");
chart("c3",[{color:"#4dd0e1",data:managed.map(r=>r.window_total_tokens)}],"tokens");

const sum=a=>a.reduce((x,y)=>x+y,0);
const nTok=sum(naive.map(r=>r.prompt_tokens)), mTok=sum(managed.map(r=>r.prompt_tokens));
const nRead=sum(naive.map(r=>r.prompt_eval_ms)), mRead=sum(managed.map(r=>r.prompt_eval_ms));
document.getElementById("totals").innerHTML =
 "<tr><th>metric</th><th>naive</th><th>managed</th><th>change</th></tr>"+
 `<tr><td>total prompt tokens</td><td>${nTok}</td><td>${mTok}</td><td>${(100*(1-mTok/nTok)).toFixed(1)}%</td></tr>`+
 `<tr><td>total prompt read ms</td><td>${nRead.toFixed(0)}</td><td>${mRead.toFixed(0)}</td><td>${(100*(1-mRead/nRead)).toFixed(1)}%</td></tr>`+
 `<tr><td>final window tokens</td><td>${naive.at(-1).prompt_tokens}</td><td>${managed.at(-1).window_total_tokens}</td><td>&mdash;</td></tr>`+
 `<tr><td>total compressions</td><td>0</td><td>${managed.at(-1).compressions}</td><td>&mdash;</td></tr>`;
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark + chart export")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--summary-tokens", type=int, default=96)
    ap.add_argument("--num-predict", type=int, default=64)
    ap.add_argument("--out", default="benchmark_results")
    args = ap.parse_args(argv)

    questions = build_questions(QUESTIONS, args.turns)
    llm = LocalLLM(model=args.model, host=args.host)
    print(f"backend: {llm.backend}  model: {args.model}  turns: {args.turns}")
    if llm.backend != "ollama":
        print("  ! Ollama not reachable -- results will be from the simulator")
    else:
        print(f"  warm-up: {llm.warmup():.0f} ms")

    print("  NAIVE")
    naive = run_strategy(llm, SAMPLE_DOCUMENT, questions, False,
                         args.max_tokens, args.summary_tokens, args.num_predict)
    print("  MANAGED")
    managed = run_strategy(llm, SAMPLE_DOCUMENT, questions, True,
                           args.max_tokens, args.summary_tokens, args.num_predict)
    rows = naive + managed

    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "model": args.model,
        "config": {
            "turns": args.turns, "max_tokens": args.max_tokens,
            "summary_tokens": args.summary_tokens, "num_predict": args.num_predict,
        },
        "rows": rows,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    html = (HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
            .replace("__MODEL__", args.model))
    with open(args.out + ".html", "w", encoding="utf-8") as f:
        f.write(html)

    n_tok = sum(r["prompt_tokens"] for r in naive)
    m_tok = sum(r["prompt_tokens"] for r in managed)
    n_read = sum(r["prompt_eval_ms"] for r in naive)
    m_read = sum(r["prompt_eval_ms"] for r in managed)
    print("\n================ SUMMARY ================")
    print(f"total prompt tokens : naive={n_tok}  managed={m_tok}  "
          f"saved={100*(1-m_tok/n_tok):.1f}%")
    print(f"total prompt read   : naive={n_read:.0f}ms  managed={m_read:.0f}ms  "
          f"saved={100*(1-m_read/n_read):.1f}%")
    print(f"final window tokens : naive={naive[-1]['prompt_tokens']}  "
          f"managed={managed[-1]['window_total_tokens']}")
    print(f"compressions        : {managed[-1]['compressions']}")
    print(f"\nwrote {args.out}.csv / .json / .html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
