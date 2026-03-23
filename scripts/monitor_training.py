"""Live training monitor — serves a browser dashboard that streams fold metrics.

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --container train-full
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOTAL_FOLDS = 2373
SSH_KEY = str(Path.home() / ".ssh/hetzner_deploy_key")
SSH_HOST = "deploy@46.225.220.203"
SSH_BASE = [
    "ssh", "-i", SSH_KEY,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=999",
    "-o", "ConnectTimeout=10",
    SSH_HOST,
]

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_folds: list[dict] = []
_phase = "starting"
_done = False
_clients: list[list] = []


def _broadcast(event: str, data: dict) -> None:
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _lock:
        for q in _clients:
            q.append(msg)


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------
FOLD_RE = re.compile(
    r"fold_complete\s+accuracy=([\d.]+)\s+fold=(\d+)\s+sharpe=(?:np\.float64\()?([-\d.]+)\)?.*?time_s=([\d.]+)"
)

PHASE_MAP = {
    "computing_features": "features",
    "walk_forward_start": "training",
    "training_stacking": "stacking",
    "training_calibrat": "calibrating",
    "all_models_saved": "saving",
    "training_complete": "complete",
}


def parse_line(line: str) -> None:
    global _phase, _done

    for key, phase in PHASE_MAP.items():
        if key in line:
            _phase = phase
            _broadcast("phase", {"phase": phase})
            if phase == "complete":
                _done = True
                m = re.search(r"mean_sharpe=([-\d.]+)", line)
                _broadcast("complete", {"mean_sharpe": float(m[1]) if m else None})
            return

    m = FOLD_RE.search(line)
    if m:
        entry = {
            "accuracy": float(m[1]),
            "fold": int(m[2]),
            "sharpe": float(m[3]),
            "time_s": float(m[4]),
        }
        with _lock:
            _folds.append(entry)
        _broadcast("fold", entry)


# ---------------------------------------------------------------------------
# Log reader — two modes:
#   remote: greps Docker logs on server via SSH (keeps SSH payloads tiny)
#   local:  tails a local log file (for local training runs)
# ---------------------------------------------------------------------------
def read_logs(container: str, local_log: str | None = None) -> None:
    last_fold_seen = -1
    if local_log:
        _read_local_log(local_log, last_fold_seen)
    else:
        _read_remote_logs(container, last_fold_seen)


def _read_local_log(log_path: str, last_fold_seen: int) -> None:
    """Read training progress from a local log file, polling for new lines."""
    global _done
    path = Path(log_path)
    offset = 0  # byte offset — only read new content each poll

    while not _done:
        try:
            if not path.exists():
                time.sleep(2)
                continue

            with path.open("rb") as f:
                f.seek(offset)
                raw = f.read()
                offset = f.tell()
            # Strip null bytes that appear when multiple processes write to the same file
            new_content = raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

            for line in new_content.splitlines():
                m = FOLD_RE.search(line)
                if m and int(m[2]) <= last_fold_seen:
                    continue
                parse_line(line)
                if m:
                    last_fold_seen = int(m[2])

            if _phase == "complete":
                break
        except Exception:
            pass
        time.sleep(2)


def _read_remote_logs(container: str, last_fold_seen: int) -> None:
    """Read training progress from a remote Docker container via SSH."""
    global _done
    while not _done:
        try:
            cmd = (
                f"docker logs {container} 2>&1 | "
                f"grep -E 'fold_complete|computing_features|walk_forward_start|"
                f"training_stacking|training_calibrat|all_models_saved|training_complete'"
            )
            result = subprocess.run(
                SSH_BASE + [cmd],
                capture_output=True, text=True, timeout=20,
            )
            for line in result.stdout.splitlines():
                m = FOLD_RE.search(line)
                if m and int(m[2]) <= last_fold_seen:
                    continue
                parse_line(line)
                if m:
                    last_fold_seen = int(m[2])

            if _phase == "complete":
                break
        except Exception:
            pass
        time.sleep(3)


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ep2-crypto training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0a0a0a;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  padding: 28px;
  max-width: 960px;
  margin: 0 auto;
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}
header h1 { font-size: 15px; font-weight: 600; color: #888; letter-spacing: 0.04em; }
.badge {
  padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  background: #1a2a1a; color: #4ade80; letter-spacing: 0.05em;
}
.badge.warn { background: #2a2010; color: #fbbf24; }
.badge.idle { background: #1a1a2a; color: #60a5fa; }

/* Progress */
.progress-card {
  background: #111; border: 1px solid #222; border-radius: 12px;
  padding: 20px 24px; margin-bottom: 20px;
}
.progress-row { display: flex; align-items: center; gap: 16px; }
.progress-wrap { flex: 1; background: #1e1e1e; border-radius: 6px; height: 8px; overflow: hidden; }
.progress-bar {
  height: 100%; border-radius: 6px; width: 0%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  transition: width 0.6s ease;
}
.progress-label { font-size: 13px; color: #666; white-space: nowrap; min-width: 110px; text-align: right; }
.eta { font-size: 13px; color: #555; margin-top: 8px; }
.eta span { color: #888; }

/* Big stat cards */
.cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 20px; }
.card {
  background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px;
}
.card-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
.card-value { font-size: 32px; font-weight: 700; line-height: 1; }
.card-sub { font-size: 12px; color: #555; margin-top: 8px; line-height: 1.5; }
.card-sub b { color: #888; }

.green { color: #4ade80; }
.red { color: #f87171; }
.yellow { color: #fbbf24; }
.blue { color: #60a5fa; }
.white { color: #fff; }

/* Verdict bar */
.verdict {
  background: #111; border: 1px solid #222; border-radius: 12px;
  padding: 16px 20px; margin-bottom: 20px;
  display: flex; align-items: center; gap: 14px;
}
.verdict-icon { font-size: 22px; }
.verdict-text { font-size: 14px; color: #aaa; line-height: 1.5; }
.verdict-text strong { color: #fff; }

/* Charts */
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.chart-card {
  background: #111; border: 1px solid #222; border-radius: 12px; padding: 18px;
}
.chart-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
canvas { max-height: 180px; }
</style>
</head>
<body>

<header>
  <h1>⚡ ep2-crypto · walk-forward training</h1>
  <div class="badge idle" id="phase">connecting…</div>
</header>

<!-- Progress -->
<div class="progress-card">
  <div class="progress-row">
    <div class="progress-wrap"><div class="progress-bar" id="pbar"></div></div>
    <div class="progress-label" id="fold-label">waiting…</div>
  </div>
  <div class="eta" id="eta">Computing features on 687,528 bars — folds will start in ~1 min</div>
</div>

<!-- Key stats -->
<div class="cards">
  <div class="card">
    <div class="card-label">Win rate</div>
    <div class="card-value white" id="winrate">—</div>
    <div class="card-sub" id="winrate-sub">Need <b>&gt;52%</b> to cover trading costs</div>
  </div>
  <div class="card">
    <div class="card-label">Consistency</div>
    <div class="card-value white" id="consistency">—</div>
    <div class="card-sub" id="consistency-sub">How stable the edge is across time</div>
  </div>
  <div class="card">
    <div class="card-label">Time left</div>
    <div class="card-value white" id="timeleft">—</div>
    <div class="card-sub" id="speed-sub">measuring speed…</div>
  </div>
</div>

<!-- Verdict -->
<div class="verdict" id="verdict" style="display:none">
  <div class="verdict-icon" id="verdict-icon">🔍</div>
  <div class="verdict-text" id="verdict-text"></div>
</div>

<!-- Charts -->
<div class="charts">
  <div class="chart-card">
    <div class="chart-label">Rolling win rate (50-fold avg)</div>
    <canvas id="winChart"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-label">Profit score per period</div>
    <canvas id="sharpeChart"></canvas>
  </div>
</div>

<script>
const TOTAL = """ + str(TOTAL_FOLDS) + r""";
const folds = [];
let startTime = null;
let historyDone = false;

const chartOpts = {
  animation: false, responsive: true, maintainAspectRatio: true,
  plugins: { legend: { display: false } },
  elements: { point: { radius: 0 } },
  scales: {
    x: { display: false },
    y: { grid: { color: '#1e1e1e' }, ticks: { color: '#444', font: { size: 10 } } }
  }
};

const winChart = new Chart(document.getElementById('winChart').getContext('2d'), {
  type: 'line',
  data: { labels: [], datasets: [
    { data: [], borderColor: '#3b82f6', borderWidth: 2, fill: true,
      backgroundColor: 'rgba(59,130,246,0.05)', tension: 0.4 },
    { data: [], borderColor: '#ffffff22', borderWidth: 1, borderDash: [4,4], fill: false }
  ]},
  options: { ...chartOpts, scales: { ...chartOpts.scales, y: {
    ...chartOpts.scales.y, min: 0.4, max: 0.65,
    ticks: { ...chartOpts.scales.y.ticks, callback: v => (v*100).toFixed(0)+'%' }
  }}}
});

const sharpeChart = new Chart(document.getElementById('sharpeChart').getContext('2d'), {
  type: 'bar',
  data: { labels: [], datasets: [{
    data: [],
    backgroundColor: ctx => ctx.raw >= 0 ? 'rgba(74,222,128,0.6)' : 'rgba(248,113,113,0.6)',
    borderWidth: 0,
  }]},
  options: { ...chartOpts }
});

function rollingAvg(arr, n) {
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i-n+1), i+1);
    return slice.reduce((a,b) => a+b, 0) / slice.length;
  });
}

function avg(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }
function std(arr) {
  const m = avg(arr);
  return Math.sqrt(arr.reduce((a,b) => a+(b-m)**2, 0) / Math.max(arr.length-1,1));
}

function updateVerdict(accs, sharpes) {
  if (accs.length < 20) return;
  const el = document.getElementById('verdict');
  const icon = document.getElementById('verdict-icon');
  const text = document.getElementById('verdict-text');
  el.style.display = 'flex';

  const recentAcc = avg(accs.slice(-100));
  const recentSharpe = avg(sharpes.slice(-100));
  const stability = std(accs.slice(-100));

  let msg, ico;
  if (recentAcc >= 0.54 && stability < 0.06) {
    ico = '✅'; msg = `<strong>Looking good.</strong> The model is winning ${(recentAcc*100).toFixed(1)}% of the time over the last 100 periods, consistently. That's above the ~52% needed to be profitable after costs. Keep going.`;
  } else if (recentAcc >= 0.52) {
    ico = '🟡'; msg = `<strong>Marginal edge.</strong> Win rate is ${(recentAcc*100).toFixed(1)}% — barely above breakeven. It may work but live performance will be tight. Watch if it improves.`;
  } else if (recentAcc < 0.50) {
    ico = '🔴'; msg = `<strong>Below random.</strong> The model is calling ${(recentAcc*100).toFixed(1)}% correctly in recent periods — worse than a coin flip. This could be a hard market period or a model issue.`;
  } else {
    ico = '🔵'; msg = `<strong>Early stage.</strong> ${(recentAcc*100).toFixed(1)}% win rate so far. Need more folds to judge — the model improves as it sees more history.`;
  }
  icon.textContent = ico;
  text.innerHTML = msg;
}

function onFold(fold) {
  if (historyDone && !startTime) startTime = Date.now();
  folds.push(fold);
  const n = folds.length;

  // Progress
  const pct = n / TOTAL * 100;
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('fold-label').textContent = `${n.toLocaleString()} / ${TOTAL.toLocaleString()} folds`;

  // ETA
  if (startTime) {
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = n / elapsed;  // folds/sec
    const remaining = Math.round((TOTAL - n) / rate);
    const mins = Math.floor(remaining / 60), secs = remaining % 60;
    document.getElementById('timeleft').textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    document.getElementById('speed-sub').textContent = `${(rate * 60).toFixed(0)} folds/min`;
    const etaDate = new Date(Date.now() + remaining * 1000);
    document.getElementById('eta').innerHTML =
      `<span>Done around ${etaDate.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})} · ${elapsed < 60 ? Math.round(elapsed)+'s' : Math.floor(elapsed/60)+'m'} elapsed</span>`;
  }

  // Win rate card
  const accs = folds.map(f => f.accuracy);
  const recentAcc = avg(accs.slice(-50));
  const winEl = document.getElementById('winrate');
  winEl.textContent = (recentAcc * 100).toFixed(1) + '%';
  winEl.className = 'card-value ' + (recentAcc >= 0.54 ? 'green' : recentAcc >= 0.52 ? 'yellow' : recentAcc >= 0.50 ? 'white' : 'red');
  document.getElementById('winrate-sub').innerHTML =
    recentAcc >= 0.54 ? '<b>Above target</b> — profitable range' :
    recentAcc >= 0.52 ? 'Need <b>&gt;52%</b> to cover costs — borderline' :
    recentAcc >= 0.50 ? 'Need <b>&gt;52%</b> to cover trading costs' :
    '<b>Below 50%</b> — worse than random right now';

  // Consistency
  const stability = std(accs.slice(-50));
  const consEl = document.getElementById('consistency');
  const isStable = stability < 0.06;
  consEl.textContent = isStable ? 'Stable' : stability < 0.09 ? 'Variable' : 'Volatile';
  consEl.className = 'card-value ' + (isStable ? 'green' : stability < 0.09 ? 'yellow' : 'red');
  document.getElementById('consistency-sub').textContent =
    isStable ? 'Edge is consistent across market periods' :
    stability < 0.09 ? 'Some variance — normal for crypto' :
    'High variance — edge may be unstable';

  // Charts (downsample to last 200 for perf)
  const window = folds.slice(-200);
  const labels = window.map(f => f.fold);
  const rolling = rollingAvg(window.map(f => f.accuracy), 50);
  const baseline = Array(window.length).fill(0.52);

  winChart.data.labels = labels;
  winChart.data.datasets[0].data = rolling;
  winChart.data.datasets[1].data = baseline;
  winChart.update('none');

  // Sharpe as "profit score" — positive = made money, negative = lost
  const sharpeWindow = folds.slice(-60);
  sharpeChart.data.labels = sharpeWindow.map(f => f.fold);
  sharpeChart.data.datasets[0].data = sharpeWindow.map(f => f.sharpe);
  sharpeChart.update('none');

  updateVerdict(accs, folds.map(f => f.sharpe));
}

// SSE with auto-reconnect on close/error
function connectSSE() {
  const es = new EventSource('/stream');
  es.addEventListener('fold', e => onFold(JSON.parse(e.data)));
  es.addEventListener('history_done', e => {
    historyDone = true;
    startTime = Date.now();
    const { folds: n } = JSON.parse(e.data);
    if (n === 0) {
      document.getElementById('eta').textContent = 'Computing features on 687,528 bars — folds will start in ~1 min…';
    }
  });
  es.addEventListener('phase', e => {
    const { phase } = JSON.parse(e.data);
    const el = document.getElementById('phase');
    const labels = {
      starting: 'Starting…', features: 'Computing features…',
      training: 'Training folds', stacking: 'Stacking ensemble',
      calibrating: 'Calibrating', saving: 'Saving models', complete: '✓ Complete'
    };
    el.textContent = labels[phase] || phase;
    el.className = 'badge ' + (phase === 'complete' ? '' : phase === 'training' ? '' : 'idle');
    if (phase === 'training') el.className = 'badge';
  });
  es.addEventListener('complete', e => {
    const { mean_sharpe } = JSON.parse(e.data);
    document.getElementById('phase').textContent = '✓ Training complete';
    document.getElementById('phase').className = 'badge';
    document.getElementById('eta').innerHTML =
      `<span>🎉 Done! Mean profit score across all periods: <b>${mean_sharpe?.toFixed(1)}</b></span>`;
    es.close();
  });
  es.onerror = () => { es.close(); setTimeout(connectSSE, 2000); };
  return es;
}
const es = connectSSE();

</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
            except OSError:
                return

            q: list[str] = []
            with _lock:
                for fold in _folds:
                    q.append(f"event: fold\ndata: {json.dumps(fold)}\n\n")
                q.append(f"event: history_done\ndata: {json.dumps({'folds': len(_folds)})}\n\n")
                q.append(f"event: phase\ndata: {json.dumps({'phase': _phase})}\n\n")
                _clients.append(q)

            try:
                while True:
                    while q:
                        self.wfile.write(q.pop(0).encode())
                        self.wfile.flush()
                    if _done and not q:
                        break
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _lock:
                    if q in _clients:
                        _clients.remove(q)
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="train-full")
    parser.add_argument("--port", type=int, default=7331)
    parser.add_argument(
        "--local",
        metavar="LOG_FILE",
        default=None,
        help="Read from a local log file instead of SSH (e.g. /tmp/local_train.log)",
    )
    args = parser.parse_args()

    reader = threading.Thread(target=read_logs, args=(args.container, args.local), daemon=True)
    reader.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Monitor → {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
