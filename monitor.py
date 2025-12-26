import os
import glob
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from flask import Flask, render_template_string, jsonify, request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

app = Flask(__name__)

# --- CONFIG ---
# Map display names to folder paths
LOG_DIRS = {
    "Battle": "logs/battle",
    "Planning": "logs/planning"
}

HOST = "0.0.0.0"
PORT = 6007
MIN_RELOAD_INTERVAL_S = 1.0
MAX_CACHE_POINTS = 500_000

# HTML Template with Tabs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NitroGen Training Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }

        /* TABS */
        .tabs { display: flex; border-bottom: 1px solid #333; margin-bottom: 20px; }
        .tab { 
            padding: 10px 20px; cursor: pointer; 
            background: #222; border: 1px solid #333; border-bottom: none;
            margin-right: 5px; border-radius: 5px 5px 0 0;
            color: #888; font-weight: bold; transition: 0.2s;
        }
        .tab:hover { background: #333; color: #fff; }
        .tab.active { background: #4db8ff; color: #000; border-color: #4db8ff; }

        .stat-box { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat { background: #222; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #333; }
        .stat h3 { margin: 0 0 5px 0; font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .stat div { font-size: 32px; font-weight: bold; color: #4db8ff; }

        .controls { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 24px; align-items: center; flex-wrap: wrap; border: 1px solid #333; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        input[type=number] { background: #333; border: 1px solid #555; color: #fff; padding: 8px; border-radius: 4px; width: 110px; }
        input[type=range] { width: 150px; accent-color: #4db8ff; }
        label { font-size: 12px; color: #aaa; font-weight: bold; }

        button { background:#4db8ff; color:#000; border:none; padding:10px 20px; border-radius:4px; cursor:pointer; font-weight:bold; transition: background 0.2s; }
        button:hover { background: #3aa8eb; }

        #chart { width: 100%; height: 70vh; background: #000; border-radius: 8px; border: 1px solid #333; }
        .status { font-size: 12px; color: #666; margin-top: 5px; text-align: right; }
    </style>
</head>
<body>
<div class="container">
    
    <div class="tabs" id="tab-container">
        </div>

    <div class="stat-box">
        <div class="stat">
            <h3>Global Step</h3>
            <div id="curr-step">---</div>
        </div>
        <div class="stat">
            <h3>Current Loss</h3>
            <div id="curr-loss" style="color: #ff4d4d">---</div>
        </div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Ignore First N Steps</label>
            <input type="number" id="skip" value="100" min="0" step="100" onchange="resetAndFetch()">
        </div>

        <div class="control-group">
            <label>Max Points (client)</label>
            <input type="number" id="maxPoints" value="5000" min="500" step="500" onchange="resetAndFetch()">
        </div>

        <div class="control-group">
            <label>Smoothing (<span id="smooth-val">0.60</span>)</label>
            <input type="range" id="smooth" min="0" max="0.99" step="0.01" value="0.60" oninput="updateSmoothing()">
        </div>

        <div class="control-group" style="flex-direction: row; align-items: center; gap: 10px;">
            <input type="checkbox" id="refresh" checked>
            <label for="refresh" style="margin:0; cursor:pointer;">Auto-Refresh (1s)</label>
        </div>

        <div style="flex-grow:1; text-align:right;">
            <button onclick="fetchData(true)">Update Now</button>
        </div>
    </div>

    <div id="chart"></div>
    <div class="status" id="status">Initializing...</div>
</div>

<script>
    let currentMode = "Battle"; // Default
    let rawData = { steps: [], values: [] };
    let lastStep = null;
    let inFlight = false;

    // --- Tab Setup ---
    const MODES = {{ modes | tojson }};
    const tabContainer = document.getElementById('tab-container');
    
    MODES.forEach(mode => {
        const btn = document.createElement('div');
        btn.className = `tab ${mode === currentMode ? 'active' : ''}`;
        btn.innerText = mode;
        btn.onclick = () => switchMode(mode, btn);
        tabContainer.appendChild(btn);
    });

    function switchMode(mode, btnEl) {
        if(mode === currentMode) return;
        currentMode = mode;
        
        // Update UI Tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        btnEl.classList.add('active');
        
        // Hard Reset
        rawData = { steps: [], values: [] };
        lastStep = null;
        document.getElementById('curr-step').innerText = "---";
        document.getElementById('curr-loss').innerText = "---";
        
        fetchData(true);
    }

    // Bidirectional exponential smoothing
    function smooth(values, alpha) {
        if (values.length === 0) return [];
        if (alpha === 0) return values;
        const n = values.length;
        let forward = new Float32Array(n);
        let curr = values[0];
        forward[0] = curr;
        for (let i = 1; i < n; i++) {
            curr = curr * alpha + (1 - alpha) * values[i];
            forward[i] = curr;
        }
        let backward = new Float32Array(n);
        curr = forward[n - 1];
        backward[n - 1] = curr;
        for (let i = n - 2; i >= 0; i--) {
            curr = curr * alpha + (1 - alpha) * forward[i];
            backward[i] = curr;
        }
        return Array.from(backward);
    }

    function updateSmoothing() {
        const alpha = parseFloat(document.getElementById('smooth').value);
        document.getElementById('smooth-val').innerText = alpha.toFixed(2);
        renderChart();
    }

    function renderChart() {
        if (rawData.steps.length === 0) {
            Plotly.purge('chart');
            return;
        }

        const alpha = parseFloat(document.getElementById('smooth').value);
        const smoothedVals = smooth(rawData.values, alpha);

        let minVal = Infinity, maxVal = -Infinity;
        for (let v of smoothedVals) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        const range = maxVal - minVal;
        const padding = (range === 0) ? 0.1 : range * 0.05;
        
        const traceRaw = {
            x: rawData.steps, y: rawData.values,
            mode: 'lines', name: 'Raw',
            line: { color: 'rgba(0, 255, 204, 0.15)', width: 1 }, hoverinfo: 'none'
        };

        const traceSmooth = {
            x: rawData.steps, y: smoothedVals,
            mode: 'lines', name: 'Smoothed',
            line: { color: '#00ffcc', width: 2.5 },
            fill: 'tozeroy', fillcolor: 'rgba(0, 255, 204, 0.05)'
        };

        const layout = {
            title: `Training Loss (${currentMode})`,
            paper_bgcolor: '#000', plot_bgcolor: '#000',
            font: { color: '#eee' },
            xaxis: { title: 'Global Step', gridcolor: '#333', zerolinecolor: '#444' },
            yaxis: { title: 'Loss', gridcolor: '#333', zerolinecolor: '#444', range: [Math.max(0, minVal-padding), maxVal+padding] },
            margin: { t: 40, l: 60, r: 20, b: 60 },
            showlegend: false
        };

        Plotly.react('chart', [traceRaw, traceSmooth], layout);
    }

    function resetAndFetch() {
        rawData = { steps: [], values: [] };
        lastStep = null;
        fetchData(true);
    }

    async function fetchData(forceFull = false) {
        if (inFlight) return;
        inFlight = true;

        const skip = parseInt(document.getElementById('skip').value || "0", 10);
        const maxPoints = parseInt(document.getElementById('maxPoints').value || "5000", 10);
        const status = document.getElementById('status');

        try {
            const sinceParam = (forceFull || lastStep === null) ? "" : String(lastStep);
            // Pass currentMode to backend
            const url = `/api/data?mode=${currentMode}&skip=${skip}&max_points=${maxPoints}` + (sinceParam ? `&since=${sinceParam}` : "");
            
            const res = await fetch(url);
            const data = await res.json();

            if (data.error) {
                status.innerText = "Error: " + data.error;
                inFlight = false;
                return;
            }

            if (data.reset) {
                rawData.steps = data.steps || [];
                rawData.values = data.values || [];
            } else {
                if (data.steps && data.steps.length > 0) {
                    rawData.steps.push(...data.steps);
                    rawData.values.push(...data.values);
                }
            }

            if (rawData.steps.length > maxPoints) {
                const start = rawData.steps.length - maxPoints;
                rawData.steps = rawData.steps.slice(start);
                rawData.values = rawData.values.slice(start);
            }

            if (rawData.steps.length > 0) {
                lastStep = rawData.steps[rawData.steps.length - 1];
                const lastLoss = rawData.values[rawData.values.length - 1];
                document.getElementById('curr-step').innerText = lastStep.toLocaleString();
                document.getElementById('curr-loss').innerText = Number(lastLoss).toFixed(4);
                renderChart();
                status.innerText = `Last updated: ${new Date().toLocaleTimeString()} (${rawData.steps.length} pts)`;
            } else {
                status.innerText = "No data points found yet.";
            }

        } catch (e) {
            status.innerText = "Connection Failed";
        } finally {
            inFlight = false;
        }
    }

    // Init
    setInterval(() => {
        if (document.getElementById('refresh').checked) fetchData(false);
    }, 1000);
    // Initial fetch handled by switchMode logic or manual call if needed
    fetchData(true);
</script>
</body>
</html>
"""

def get_latest_log(log_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not files: return None
    return max(files, key=os.path.getctime)

def _safe_int(v: str, default: int) -> int:
    try: return int(v)
    except: return default

def _downsample_stride(steps: List[int], values: List[float], max_points: int) -> Tuple[List[int], List[float]]:
    n = len(steps)
    if max_points <= 0 or n <= max_points: return steps, values
    stride = max(1, n // max_points)
    ds_steps = steps[::stride]
    ds_vals = values[::stride]
    if ds_steps[-1] != steps[-1]:
        ds_steps.append(steps[-1])
        ds_vals.append(values[-1])
    if len(ds_steps) > max_points:
        ds_steps = ds_steps[-max_points:]
        ds_vals = ds_vals[-max_points:]
    return ds_steps, ds_vals

@dataclass
class _ScalarCacheState:
    log_file: Optional[str] = None
    log_mtime_ns: int = 0
    tag: Optional[str] = None
    steps: List[int] = None
    values: List[float] = None
    last_seen_step: int = -1
    last_reload_monotonic: float = 0.0
    ea: Optional[EventAccumulator] = None

class TensorboardScalarCache:
    def __init__(self, log_dir: str):
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._st = _ScalarCacheState(steps=[], values=[])

    def _maybe_switch_log(self) -> bool:
        latest = get_latest_log(self._log_dir)
        if latest is None:
            if self._st.log_file is not None:
                self._reset_state()
                return True
            return False

        try: mtime_ns = os.stat(latest).st_mtime_ns
        except: mtime_ns = 0

        if self._st.log_file != latest:
            self._reset_state()
            self._st.log_file = latest
            self._st.log_mtime_ns = mtime_ns
            self._st.ea = EventAccumulator(latest, size_guidance={"scalars": 0})
            return True

        self._st.log_mtime_ns = mtime_ns
        return False

    def _reset_state(self) -> None:
        self._st = _ScalarCacheState(steps=[], values=[])

    def _maybe_reload_and_append(self) -> None:
        if self._st.log_file is None or self._st.ea is None: return
        now = time.monotonic()
        if (now - self._st.last_reload_monotonic) < MIN_RELOAD_INTERVAL_S: return
        self._st.last_reload_monotonic = now
        
        try:
            self._st.ea.Reload()
        except Exception: return # File access race?

        tags = self._st.ea.Tags().get("scalars", [])
        if not tags: return

        if self._st.tag is None:
            # Prefer loss, fallback to first tag
            self._st.tag = next((t for t in tags if "Loss" in t or "loss" in t), None)
        
        if self._st.tag is None: return

        events = self._st.ea.Scalars(self._st.tag)
        if not events: return

        last = self._st.last_seen_step
        for e in events:
            if e.step <= last: continue
            self._st.steps.append(int(e.step))
            self._st.values.append(float(e.value))
            last = e.step
        
        self._st.last_seen_step = last

        if len(self._st.steps) > MAX_CACHE_POINTS:
            cut = len(self._st.steps) - MAX_CACHE_POINTS
            self._st.steps = self._st.steps[cut:]
            self._st.values = self._st.values[cut:]

    def get_series(self, *, skip: int, since_step: Optional[int], max_points: int) -> Tuple[bool, List[int], List[float], Optional[int], Optional[str]]:
        skip = max(0, int(skip))
        max_points = max(500, int(max_points))

        with self._lock:
            reset = self._maybe_switch_log()
            self._maybe_reload_and_append()

            if self._st.log_file is None or self._st.tag is None:
                return reset, [], [], None, None

            steps = self._st.steps
            vals = self._st.values
            if not steps: return reset, [], [], None, self._st.tag

            import bisect
            i0 = bisect.bisect_left(steps, skip)
            steps2 = steps[i0:]
            vals2 = vals[i0:]

            if since_step is not None:
                if since_step < skip:
                    ds_steps, ds_vals = _downsample_stride(steps2, vals2, max_points)
                    return True, ds_steps, ds_vals, (ds_steps[-1] if ds_steps else None), self._st.tag

                j0 = bisect.bisect_right(steps2, since_step)
                return reset, steps2[j0:], vals2[j0:], steps2[-1], self._st.tag

            ds_steps, ds_vals = _downsample_stride(steps2, vals2, max_points)
            return True if reset else True, ds_steps, ds_vals, (ds_steps[-1] if ds_steps else None), self._st.tag

# --- INIT CACHES ---
_caches = {k: TensorboardScalarCache(v) for k, v in LOG_DIRS.items()}

@app.route("/")
def index():
    # Pass available modes to the template
    return render_template_string(HTML_TEMPLATE, modes=list(LOG_DIRS.keys()))

@app.route("/api/data")
def get_data():
    mode = request.args.get("mode", "Battle") # Default to Battle
    if mode not in _caches:
        return jsonify({"error": f"Unknown mode: {mode}"})
    
    skip_n = _safe_int(request.args.get("skip", "100"), 100)
    max_points = _safe_int(request.args.get("max_points", "5000"), 5000)
    since_raw = request.args.get("since", None)
    
    since_step = None
    if since_raw is not None and since_raw != "":
        s = _safe_int(since_raw, -1)
        if s >= 0: since_step = s

    try:
        reset, steps, values, last_step, tag = _caches[mode].get_series(
            skip=skip_n, since_step=since_step, max_points=max_points
        )

        if tag is None:
            return jsonify({"error": f"Waiting for {mode} logs..."})

        return jsonify({
            "reset": bool(reset),
            "steps": steps,
            "values": values,
            "last_step": last_step,
            "tag": tag,
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    hostname = socket.gethostname()
    try: local_ip = socket.gethostbyname(hostname)
    except: local_ip = "127.0.0.1"

    print(f"\n📊 Monitor running at:")
    print(f"   👉 http://127.0.0.1:{PORT}")
    print(f"   👉 http://{local_ip}:{PORT} (Local Network)\n")
    print(f"   Watching: {list(LOG_DIRS.keys())}")

    app.run(host=HOST, port=PORT, debug=False, threaded=True)