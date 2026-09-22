"""Live web UI: stream tokens to NAIVE and MANAGED panels side by side.

Runs a tiny local web server (standard library only) that:
  * serves a chat page,
  * for every message, runs BOTH context strategies at once and streams the
    model's tokens to the browser as they are produced (NDJSON over HTTP).

Run:
    python web_chat.py
    python web_chat.py --model llama3.1:8b --max-tokens 256 --port 8000

Then open http://localhost:8000  (it opens automatically).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from local_llm import LocalLLM
from sliding_context import SlidingWindowContext

# --------------------------------------------------------------------------- #
# Server-side conversation state (one local user)
# --------------------------------------------------------------------------- #
class ChatState:
    def __init__(self, llm: LocalLLM, max_tokens: int, summary_tokens: int) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary_tokens = summary_tokens
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.naive_history = []
        self.ctx = SlidingWindowContext(self.max_tokens, self.summary_tokens)


STATE: ChatState = None  # type: ignore
NUM_PREDICT = 128


def run_turn(message: str, num_predict: int):
    """Generator of NDJSON event dicts; streams both sides concurrently."""
    q: "queue.Queue[dict]" = queue.Queue()
    answers: dict = {}

    def worker(side: str) -> None:
        try:
            if side == "naive":
                prompt = ("\n\n".join(STATE.naive_history + [f"User: {message}"])
                          + "\nAssistant:")
            else:
                with STATE.lock:
                    STATE.ctx.add(f"User: {message}")
                    prompt = STATE.ctx.render() + "\nAssistant:"

            parts = []
            for ev in STATE.llm.stream_generate(prompt, num_predict):
                if ev["type"] == "token":
                    parts.append(ev["text"])
                    q.put({"side": side, "type": "token", "text": ev["text"]})
                elif ev["type"] == "done":
                    answers[side] = "".join(parts)
                    q.put({"side": side, "type": "done",
                           "prompt_tokens": ev["prompt_tokens"],
                           "eval_tokens": ev["eval_tokens"],
                           "prompt_eval_ms": ev["prompt_eval_ms"],
                           "eval_ms": ev["eval_ms"]})
        except Exception as exc:  # pragma: no cover - defensive
            q.put({"side": side, "type": "error", "message": str(exc)})
        finally:
            q.put({"type": "side_end", "side": side})

    threads = [threading.Thread(target=worker, args=(s,))
               for s in ("naive", "managed")]
    for t in threads:
        t.start()

    ends = 0
    while ends < 2:
        ev = q.get()
        if ev.get("type") == "side_end":
            ends += 1
            continue
        yield ev

    for t in threads:
        t.join()

    with STATE.lock:
        STATE.naive_history.append(f"User: {message}")
        STATE.naive_history.append(f"Assistant: {answers.get('naive', '')}")
        STATE.ctx.add(f"Assistant: {answers.get('managed', '')}")
        st = STATE.ctx.stats()
        naive_tokens = sum(len(h.split()) for h in STATE.naive_history)

    yield {"type": "turn_end", "window": st, "naive_history_tokens": naive_tokens}


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence request logs
        pass

    # ---- helpers ----
    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routes ----
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            message = (body.get("message") or "").strip()
            if not message:
                self._json({"error": "empty message"}, 400)
                return
            try:
                num_predict = int(body.get("num_predict") or NUM_PREDICT)
            except (TypeError, ValueError):
                num_predict = NUM_PREDICT
            num_predict = max(16, min(num_predict, 4096))

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ev in run_turn(message, num_predict):
                    self._chunk((json.dumps(ev) + "\n").encode("utf-8"))
            except Exception as exc:  # pragma: no cover
                self._chunk((json.dumps({"type": "error", "message": str(exc)}) + "\n").encode())
            self._end_chunks()
        elif self.path == "/api/reset":
            STATE.reset()
            self._json({"ok": True})
        else:
            self.send_error(404)


# --------------------------------------------------------------------------- #
# Front-end
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Context: naive vs sliding-window</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --line:#2a2f3a; --txt:#e6e8ee;
          --dim:#8b93a7; --red:#ff6b6b; --cyan:#4dd0e1; --green:#7bd88f; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
         background:var(--bg); color:var(--txt); height:100vh; display:flex;
         flex-direction:column; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .sub { color:var(--dim); font-size:12px; }
  header button { margin-left:auto; background:#242938; color:var(--txt);
           border:1px solid var(--line); border-radius:6px; padding:6px 12px;
           cursor:pointer; font-size:12px; }
  header button:hover { background:#2f3547; }
  main { flex:1; display:flex; gap:1px; background:var(--line); min-height:0; }
  .panel { flex:1; display:flex; flex-direction:column; background:var(--panel);
           min-width:0; }
  .phead { padding:8px 14px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:10px; }
  .phead .name { font-weight:600; font-size:13px; }
  .naive .name { color:var(--red); }
  .managed .name { color:var(--cyan); }
  .phead .note { color:var(--dim); font-size:11px; }
  .stats { margin-left:auto; font-size:11px; color:var(--dim); text-align:right; }
  .stats b { color:var(--txt); font-weight:600; }
  .bar { height:4px; background:#0b0d13; }
  .bar > i { display:block; height:100%; width:0; background:var(--cyan);
             transition:width .2s; }
  .msgs { flex:1; overflow-y:auto; padding:14px; font-size:13.5px;
          line-height:1.55; }
  .msg { margin-bottom:12px; }
  .msg .role { font-size:11px; color:var(--dim); margin-bottom:3px; }
  .msg.user .bubble { background:#232838; border-radius:8px; padding:8px 10px;
          display:inline-block; max-width:90%; }
  .msg.assistant .bubble { white-space:pre-wrap; }
  .caret { display:inline-block; width:7px; height:15px; background:var(--txt);
           vertical-align:-2px; animation:blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  footer { border-top:1px solid var(--line); padding:12px 16px; display:flex;
           gap:10px; }
  footer input { flex:1; background:#0b0d13; border:1px solid var(--line);
           color:var(--txt); border-radius:8px; padding:11px 13px; font-size:14px;
           outline:none; }
  footer input:focus { border-color:var(--cyan); }
  footer .maxout { display:flex; align-items:center; gap:6px; color:var(--dim);
           font-size:12px; white-space:nowrap; }
  footer .maxout input { width:74px; flex:none; padding:8px 8px; }
  footer button { background:var(--cyan); color:#04222a; border:0; border-radius:8px;
           padding:0 20px; font-weight:700; cursor:pointer; font-size:14px; }
  footer button:disabled { opacity:.5; cursor:default; }
</style>
</head>
<body>
<header>
  <h1>Context management: live</h1>
  <span class="sub">same message &rarr; both strategies at once, tokens streamed as generated</span>
  <button id="reset">Reset</button>
</header>

<main>
  <section class="panel naive">
    <div class="phead">
      <span class="name">NAIVE</span>
      <span class="note">sends full history every turn</span>
      <span class="stats" id="stats-naive"></span>
    </div>
    <div class="bar"><i></i></div>
    <div class="msgs" id="msgs-naive"></div>
  </section>

  <section class="panel managed">
    <div class="phead">
      <span class="name">MANAGED</span>
      <span class="note">sliding window + summary</span>
      <span class="stats" id="stats-managed"></span>
    </div>
    <div class="bar"><i id="winbar"></i></div>
    <div class="msgs" id="msgs-managed"></div>
  </section>
</main>

<footer>
  <input id="input" placeholder="Type a message and press Enter..." autocomplete="off"/>
  <label class="maxout">max out
    <input id="maxout" type="number" min="32" step="32" value="512"/>
  </label>
  <button id="send">Send</button>
</footer>

<script>
const els = {
  naive: document.getElementById("msgs-naive"),
  managed: document.getElementById("msgs-managed"),
  statsNaive: document.getElementById("stats-naive"),
  statsManaged: document.getElementById("stats-managed"),
  winbar: document.getElementById("winbar"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  reset: document.getElementById("reset"),
  maxout: document.getElementById("maxout"),
};
const live = { naive: null, managed: null };
const counters = { naive: 0, managed: 0 };
let busy = false;

function addUser(text) {
  for (const side of ["naive", "managed"]) {
    const d = document.createElement("div");
    d.className = "msg user";
    d.innerHTML = '<div class="role">you</div><div class="bubble"></div>';
    d.querySelector(".bubble").textContent = text;
    els[side].appendChild(d);
  }
  scroll();
}

function startAssistant(side) {
  const d = document.createElement("div");
  d.className = "msg assistant";
  d.innerHTML = '<div class="role">assistant</div><div class="bubble"></div>' +
                '<span class="caret"></span>';
  els[side].appendChild(d);
  live[side] = d.querySelector(".bubble");
  counters[side] = 0;
}

function handle(ev) {
  if (ev.type === "token") {
    if (!live[ev.side]) startAssistant(ev.side);
    live[ev.side].textContent += ev.text;
    counters[ev.side] += 1;
    updateStats(ev.side);
    scroll();
  } else if (ev.type === "done") {
    finishAssistant(ev.side, ev);
  } else if (ev.type === "turn_end") {
    const w = ev.window;
    const pct = Math.min(100, 100 * w.total_tokens / w.max_tokens);
    els.winbar.style.width = pct + "%";
    els.statsManaged.innerHTML =
      `window <b>${w.total_tokens}/${w.max_tokens}</b> tok &middot; ` +
      `summary <b>${w.summary_tokens}</b> &middot; compressions <b>${w.compressions}</b>`;
  } else if (ev.type === "error") {
    console.error(ev.message);
  }
}

function finishAssistant(side, ev) {
  const node = live[side] && live[side].parentElement;
  if (node) {
    const caret = node.querySelector(".caret");
    if (caret) caret.remove();
  }
  live[side] = null;
  const s = els[side === "naive" ? "statsNaive" : "statsManaged"];
  const cur = s.dataset.extra || "";
  s.innerHTML = `prompt <b>${ev.prompt_tokens}</b> tok &middot; ` +
                `out <b>${ev.eval_tokens}</b> &middot; ` +
                `read <b>${ev.prompt_eval_ms.toFixed(0)}</b>ms &middot; ` +
                `gen <b>${ev.eval_ms.toFixed(0)}</b>ms` + cur;
}

function updateStats(side) {
  if (side === "naive") {
    els.statsNaive.innerHTML = `streaming&hellip; <b>${counters.naive}</b> tok`;
  } else {
    els.statsManaged.innerHTML = `streaming&hellip; <b>${counters.managed}</b> tok`;
  }
}

function scroll() {
  els.naive.scrollTop = els.naive.scrollHeight;
  els.managed.scrollTop = els.managed.scrollHeight;
}

async function send() {
  const message = els.input.value.trim();
  if (!message || busy) return;
  els.input.value = "";
  busy = true;
  els.send.disabled = true;
  addUser(message);
  startAssistant("naive");
  startAssistant("managed");

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, num_predict: parseInt(els.maxout.value, 10) || 512 }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (line) handle(JSON.parse(line));
      }
    }
  } catch (e) {
    console.error(e);
  } finally {
    busy = false;
    els.send.disabled = false;
    els.input.focus();
  }
}

els.send.onclick = send;
els.input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
els.reset.onclick = async () => {
  await fetch("/api/reset", { method: "POST" });
  els.naive.innerHTML = "";
  els.managed.innerHTML = "";
  els.statsNaive.innerHTML = "";
  els.statsManaged.innerHTML = "";
  els.winbar.style.width = "0";
};
els.input.focus();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    global STATE, NUM_PREDICT
    ap = argparse.ArgumentParser(description="Live web UI for naive vs sliding-window context")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--summary-tokens", type=int, default=96)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    NUM_PREDICT = args.num_predict
    llm = LocalLLM(model=args.model, host=args.host)
    STATE = ChatState(llm, args.max_tokens, args.summary_tokens)

    url = f"http://localhost:{args.port}"
    print(f"backend: {llm.backend}   model: {args.model}")
    print(f"window: max={args.max_tokens} summary={args.summary_tokens} "
          f"num_predict={args.num_predict}")
    print(f"serving at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
