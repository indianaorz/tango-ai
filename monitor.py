import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from flask import Flask, render_template_string, jsonify, request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

app = Flask(__name__)

# --- CONFIG ---
# Each MODE points at a ROOT directory that contains multiple runs (subfolders).
# We use os.path.abspath/normpath later to ensure Windows compatibility.
LOG_DIRS = {
    "Battle": "logs/nitrogen_battle_cache_bellman",
    "Planning": "logs/conditioned_nitrogen_battle_cache_bellman",
    "RL": "logs/rl_battle",
    "Critic": "checkpoints/critic_hpdelta",
    "Critic_RL": "checkpoints/critic_rl",
}

HOST = "0.0.0.0"
PORT = 6007
MIN_RELOAD_INTERVAL_S = 1.0
MAX_CACHE_POINTS = 500_000
AUTO_PIN_SECONDS = 30.0  # keep "auto-selected run" stable for this long unless it disappears

# Which tags to surface in the stat cards per mode (best-effort)
_STAT_TAGS_BY_MODE: Dict[str, List[str]] = {
    "Battle": ["Train/Loss"],
    "Planning": ["Train/Loss"],
    "RL": ["Train/WeightedLoss", "Train/Loss"],
    "Critic": ["Val/RMSE", "Val/MAE", "Train/HuberLoss_avg", "Train/HuberLoss"],
    "Critic_RL": ["Train/HuberLoss", "Val/RMSE", "Val/MAE", "Train/LR"],
}

# Prefer which tags to plot by default per mode.
# MOVED TRAIN TAGS TO TOP so graphs appear immediately during Epoch 1
_PREFER_BY_MODE: Dict[str, List[str]] = {
    "Battle": ["Train/Loss", "Loss", "loss"],
    "Planning": ["Train/Loss", "Loss", "loss"],
    "RL": ["Train/WeightedLoss", "WeightedLoss", "Train/Loss", "Loss", "loss"],
    "Critic": [
        "Train/HuberLoss",
        "Val/RMSE",
        "Train/HuberLoss_avg",
        "Val/MAE",
        "Loss",
    ],
    "Critic_RL": [
        "Train/HuberLoss",      # <--- Priority 1: Shows up immediately
        "Train/HuberLoss_avg",
        "Val/RMSE",             # <--- Priority 3: Shows up after epoch
        "Train/TDLoss",
        "Val/MAE",
        "Loss",
    ],
}

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

        .tabs { display: flex; border-bottom: 1px solid #333; margin-bottom: 20px; flex-wrap: wrap; gap: 6px; }
        .tab {
            padding: 10px 20px; cursor: pointer;
            background: #222; border: 1px solid #333; border-bottom: none;
            border-radius: 6px 6px 0 0;
            color: #888; font-weight: bold; transition: 0.2s;
            user-select: none;
        }
        .tab:hover { background: #333; color: #fff; }
        .tab.active { background: #4db8ff; color: #000; border-color: #4db8ff; }

        .stat-box { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat { background: #222; padding: 15px; border-radius: 8px; flex: 1; min-width: 220px; text-align: center; border: 1px solid #333; }
        .stat h3 { margin: 0 0 5px 0; font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .stat div { font-size: 28px; font-weight: bold; color: #4db8ff; }

        .controls { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 24px; align-items: center; flex-wrap: wrap; border: 1px solid #333; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        input[type=number] { background: #333; border: 1px solid #555; color: #fff; padding: 8px; border-radius: 4px; width: 110px; }
        input[type=range] { width: 150px; accent-color: #4db8ff; }
        select { background: #333; border: 1px solid #555; color: #fff; padding: 8px; border-radius: 4px; min-width: 260px; }
        label { font-size: 12px; color: #aaa; font-weight: bold; }

        button { background:#4db8ff; color:#000; border:none; padding:10px 20px; border-radius:4px; cursor:pointer; font-weight:bold; transition: background 0.2s; }
        button:hover { background: #3aa8eb; }

        #chart { width: 100%; height: 70vh; background: #000; border-radius: 8px; border: 1px solid #333; }
        .status { font-size: 12px; color: #666; margin-top: 5px; text-align: right; }
        .muted { color: #888; font-size: 12px; }
    </style>
</head>
<body>
<div class="container">

    <div class="tabs" id="tab-container"></div>

    <div class="stat-box" id="stat-box">
        <div class="stat">
            <h3>Global Step</h3>
            <div id="curr-step">---</div>
        </div>
        <div class="stat">
            <h3 id="curr-metric-title">Current</h3>
            <div id="curr-loss" style="color: #ff4d4d">---</div>
            <div class="muted" id="curr-metric-tag">---</div>
            <div class="muted" id="curr-run" style="margin-top:6px;">run: ---</div>
        </div>
        <div class="stat" id="extra-stat-1" style="display:none;">
            <h3 id="extra-1-title">---</h3>
            <div id="extra-1-val">---</div>
            <div class="muted" id="extra-1-tag">---</div>
        </div>
        <div class="stat" id="extra-stat-2" style="display:none;">
            <h3 id="extra-2-title">---</h3>
            <div id="extra-2-val">---</div>
            <div class="muted" id="extra-2-tag">---</div>
        </div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Run</label>
            <select id="runSelect" onchange="resetAndFetch()">
                <option value="">(auto)</option>
            </select>
        </div>

        <div class="control-group">
            <label>Metric (Scalar Tag)</label>
            <select id="tagSelect" onchange="resetAndFetch()">
                <option value="">(auto)</option>
            </select>
        </div>

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
            <input type="range" id="smooth" min="0" max="0.9999" step="0.0001" value="0.60" oninput="updateSmoothing()">
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
    const MODES = {{ modes | tojson }};
    let currentMode = (MODES && MODES.length > 0) ? MODES[0] : "Battle";

    let rawData = { steps: [], values: [] };
    let lastStep = null;
    let inFlight = false;
    let forcedFullOnce = false;

    const tabContainer = document.getElementById('tab-container');
    const tagSelect = document.getElementById('tagSelect');
    const runSelect = document.getElementById('runSelect');

    MODES.forEach(mode => {
        const btn = document.createElement('div');
        btn.className = `tab ${mode === currentMode ? 'active' : ''}`;
        btn.innerText = mode;
        btn.onclick = () => switchMode(mode, btn);
        tabContainer.appendChild(btn);
    });

    async function refreshRuns() {
        const prev = runSelect.value || "";
        runSelect.innerHTML = "";
        const optAuto = document.createElement('option');
        optAuto.value = "";
        optAuto.text = "(auto)";
        runSelect.appendChild(optAuto);

        try {
            const res = await fetch(`/api/runs?mode=${encodeURIComponent(currentMode)}`);
            const data = await res.json();
            const runs = data.runs || [];
            runs.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r;
                opt.text = r;
                runSelect.appendChild(opt);
            });
            if (prev && runs.includes(prev)) runSelect.value = prev;
            else runSelect.value = "";
        } catch (e) {
            runSelect.value = "";
        }
    }

    async function refreshTags() {
        const prev = tagSelect.value || "";
        tagSelect.innerHTML = "";
        const optAuto = document.createElement('option');
        optAuto.value = "";
        optAuto.text = "(auto)";
        tagSelect.appendChild(optAuto);

        try {
            const runParam = runSelect.value || "";
            let url = `/api/tags?mode=${encodeURIComponent(currentMode)}`;
            if (runParam) url += `&run=${encodeURIComponent(runParam)}`;
            const res = await fetch(url);
            const data = await res.json();
            const tags = data.tags || [];
            tags.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                opt.text = t;
                tagSelect.appendChild(opt);
            });
            if (prev && tags.includes(prev)) tagSelect.value = prev;
            else tagSelect.value = "";
        } catch (e) {
            tagSelect.value = "";
        }
    }

    function switchMode(mode, btnEl) {
        if (mode === currentMode) return;
        currentMode = mode;

        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        btnEl.classList.add('active');

        rawData = { steps: [], values: [] };
        lastStep = null;
        forcedFullOnce = false;

        document.getElementById('curr-step').innerText = "---";
        document.getElementById('curr-loss').innerText = "---";
        document.getElementById('curr-metric-tag').innerText = "---";
        document.getElementById('curr-run').innerText = "run: ---";

        hideExtraStats();
        refreshRuns().then(() => refreshTags().then(() => fetchData(true)));
    }

    function hideExtraStats() {
        document.getElementById('extra-stat-1').style.display = "none";
        document.getElementById('extra-stat-2').style.display = "none";
    }

    function showExtraStats(extras) {
        hideExtraStats();
        if (!extras || !Array.isArray(extras) || extras.length === 0) return;

        if (extras.length >= 1) {
            const e1 = extras[0];
            document.getElementById('extra-1-title').innerText = "Latest";
            document.getElementById('extra-1-tag').innerText = e1.tag || "---";
            document.getElementById('extra-1-val').innerText = (e1.value != null) ? Number(e1.value).toFixed(4) : "---";
            document.getElementById('extra-stat-1').style.display = "";
        }
        if (extras.length >= 2) {
            const e2 = extras[1];
            document.getElementById('extra-2-title').innerText = "Latest";
            document.getElementById('extra-2-tag').innerText = e2.tag || "---";
            document.getElementById('extra-2-val').innerText = (e2.value != null) ? Number(e2.value).toFixed(4) : "---";
            document.getElementById('extra-stat-2').style.display = "";
        }
    }

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
        document.getElementById('smooth-val').innerText = alpha.toFixed(4);
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

        const tagLabel = (tagSelect.value && tagSelect.value.length) ? tagSelect.value : "(auto)";
        const runLabel = (runSelect.value && runSelect.value.length) ? runSelect.value : "(auto)";
        const layout = {
            title: `Scalar (${currentMode}) — ${runLabel} — ${tagLabel}`,
            paper_bgcolor: '#000', plot_bgcolor: '#000',
            font: { color: '#eee' },
            xaxis: { title: 'Global Step', gridcolor: '#333', zerolinecolor: '#444' },
            yaxis: { title: 'Value', gridcolor: '#333', zerolinecolor: '#444',
                     range: [Math.max(0, minVal-padding), maxVal+padding] },
            margin: { t: 40, l: 60, r: 20, b: 60 },
            showlegend: false
        };

        Plotly.react('chart', [traceRaw, traceSmooth], layout);
    }

    function resetAndFetch() {
        rawData = { steps: [], values: [] };
        lastStep = null;
        forcedFullOnce = false;
        refreshTags().then(() => fetchData(true));
    }

    async function fetchData(forceFull = false) {
        if (inFlight) return;
        inFlight = true;

        const skip = parseInt(document.getElementById('skip').value || "0", 10);
        const maxPoints = parseInt(document.getElementById('maxPoints').value || "5000", 10);
        const status = document.getElementById('status');

        try {
            const sinceParam = (forceFull || lastStep === null) ? "" : String(lastStep);
            const tagParam = tagSelect.value || "";
            const runParam = runSelect.value || "";

            let url = `/api/data?mode=${encodeURIComponent(currentMode)}&skip=${skip}&max_points=${maxPoints}`;
            if (sinceParam) url += `&since=${encodeURIComponent(sinceParam)}`;
            if (tagParam) url += `&tag=${encodeURIComponent(tagParam)}`;
            if (runParam) url += `&run=${encodeURIComponent(runParam)}`;

            const res = await fetch(url);
            const data = await res.json();

            if (data.error) {
                status.innerText = "Error: " + data.error;
                inFlight = false;
                return;
            }

            if (data.run != null) {
                const serverRun = String(data.run);
                document.getElementById('curr-run').innerText = `run: ${serverRun || "(auto)"}`;
                if (!runSelect.value && serverRun) {
                    // auto selection
                }
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

            if (!forceFull && sinceParam && rawData.steps.length === 0 && (!data.steps || data.steps.length === 0) && !forcedFullOnce) {
                forcedFullOnce = true;
                inFlight = false;
                return fetchData(true);
            }

            if (rawData.steps.length > maxPoints) {
                const start = rawData.steps.length - maxPoints;
                rawData.steps = rawData.steps.slice(start);
                rawData.values = rawData.values.slice(start);
            }

            if (data.last_step != null) {
                lastStep = data.last_step;
            } else if (rawData.steps.length > 0) {
                lastStep = rawData.steps[rawData.steps.length - 1];
            }

            if (rawData.steps.length > 0) {
                const shownLastStep = rawData.steps[rawData.steps.length - 1];
                const lastVal = rawData.values[rawData.values.length - 1];

                document.getElementById('curr-step').innerText = shownLastStep.toLocaleString();
                document.getElementById('curr-loss').innerText = Number(lastVal).toFixed(4);

                const shownTag = data.tag || "(auto)";
                document.getElementById('curr-metric-title').innerText = "Current";
                document.getElementById('curr-metric-tag').innerText = shownTag;

                if (data.latest_extras) showExtraStats(data.latest_extras);
                else hideExtraStats();

                renderChart();
                status.innerText = `Last updated: ${new Date().toLocaleTimeString()} (${rawData.steps.length} pts)`;
            } else {
                status.innerText = "No data points found yet.";
                Plotly.purge('chart');
            }

        } catch (e) {
            status.innerText = "Connection Failed";
        } finally {
            inFlight = false;
        }
    }

    setInterval(() => {
        if (document.getElementById('refresh').checked) fetchData(false);
    }, 1000);

    refreshRuns().then(() => refreshTags().then(() => fetchData(true)));
</script>
</body>
</html>
"""


def _safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _list_event_files_recursive(log_root: str) -> List[str]:
    # Robust path handling for Windows
    if not log_root or not os.path.exists(log_root):
        return []
    
    # Use normpath to standardize slashes
    log_root = os.path.normpath(log_root)
    
    out: List[str] = []
    for root, _, files in os.walk(log_root):
        for fn in files:
            if fn.startswith("events.out.tfevents."):
                out.append(os.path.join(root, fn))
    return out


def _list_runs(log_root: str) -> List[str]:
    """
    A "run" is any directory under log_root that contains at least one TB event file.
    Return run_ids as paths relative to log_root.
    """
    log_root = os.path.normpath(os.path.abspath(log_root))
    files = _list_event_files_recursive(log_root)
    if not files:
        return []
    
    run_dirs = set()
    for p in files:
        d = os.path.dirname(p)
        try:
            rel = os.path.relpath(d, log_root)
        except Exception:
            rel = d
        run_dirs.add(rel)
    runs = sorted(run_dirs)
    return runs


def _score_event_file(p: str, prefer_tag_substrings: List[str]) -> Optional[Tuple[int, int, int, float]]:
    """
    Score one TB event file.
    Returns (prefer_hit, last_step, n_tags, ctime) or None if unusable.
    """
    try:
        ea = EventAccumulator(p, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", []) or []
    except Exception:
        return None

    if not tags:
        return None

    non_hparam = [t for t in tags if not t.startswith("hparam/")]
    if not non_hparam:
        return None

    prefer_hit = 0
    picked_tag: Optional[str] = None
    for pat in (prefer_tag_substrings or []):
        for t in non_hparam:
            if pat in t:
                prefer_hit = 1
                picked_tag = t
                break
        if prefer_hit:
            break

    # Fallbacks if preference not found
    if picked_tag is None:
        for t in non_hparam:
            if "WeightedLoss" in t:
                picked_tag = t
                break
        if picked_tag is None:
            for t in non_hparam:
                if "Loss" in t or "loss" in t:
                    picked_tag = t
                    break
        if picked_tag is None:
            # Just take the first valid tag
            picked_tag = non_hparam[0]

    last_step = 0
    try:
        evs = ea.Scalars(picked_tag)
        if evs:
            last_step = int(evs[-1].step)
    except Exception:
        last_step = 0

    try:
        ctime = os.path.getctime(p)
    except Exception:
        ctime = -1.0

    return (prefer_hit, last_step, len(non_hparam), float(ctime))


def _best_event_file_in_dir(run_dir: str, prefer_tag_substrings: List[str], *, max_candidates: int = 25) -> Optional[str]:
    if not run_dir or not os.path.exists(run_dir):
        return None
    files = []
    try:
        for fn in os.listdir(run_dir):
            if fn.startswith("events.out.tfevents."):
                files.append(os.path.join(run_dir, fn))
    except Exception:
        return None

    if not files:
        return None

    # newest first
    def ctime(p: str) -> float:
        try:
            return os.path.getctime(p)
        except Exception:
            return -1.0

    files = sorted(files, key=ctime, reverse=True)
    candidates = files[: max(1, int(max_candidates))]

    best: Optional[Tuple[Tuple[int, int, int, float], str]] = None
    for p in candidates:
        s = _score_event_file(p, prefer_tag_substrings)
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (s, p)

    if best is not None:
        return best[1]
    return candidates[0]


def _downsample_stride(steps: List[int], values: List[float], max_points: int) -> Tuple[List[int], List[float]]:
    n = len(steps)
    if max_points <= 0 or n <= max_points:
        return steps, values
    stride = max(1, n // max_points)
    ds_steps = steps[::stride]
    ds_vals = values[::stride]
    if ds_steps and ds_steps[-1] != steps[-1]:
        ds_steps.append(steps[-1])
        ds_vals.append(values[-1])
    if len(ds_steps) > max_points:
        ds_steps = ds_steps[-max_points:]
        ds_vals = ds_vals[-max_points:]
    return ds_steps, ds_vals


@dataclass
class _CacheState:
    run_id: str = ""
    run_dir_abs: str = ""
    log_file: Optional[str] = None
    log_mtime_ns: int = 0
    last_reload_monotonic: float = 0.0
    ea: Optional[EventAccumulator] = None


class TensorboardScalarCache:
    def __init__(self, log_root: str, *, prefer_tag_substrings: List[str], stat_tags: List[str]):
        # Ensure log_root is absolute and normalized for Windows
        self._log_root = os.path.normpath(os.path.abspath(log_root))
        self._prefer = [str(x) for x in (prefer_tag_substrings or [])]
        self._stat_tags = [str(x) for x in (stat_tags or [])]
        self._lock = threading.Lock()
        self._st = _CacheState()
        self._series: Dict[str, Tuple[List[int], List[float], int]] = {}
        self._known_tags: List[str] = []
        self._last_pin_time = 0.0

    def _abs_run_dir(self, run_id: str) -> str:
        if not run_id:
            return self._log_root
        # Handle mixed slashes if run_id comes from UI
        return os.path.join(self._log_root, os.path.normpath(run_id))

    def _auto_pick_run(self) -> str:
        now = time.monotonic()
        if self._st.run_id and (now - self._last_pin_time) < AUTO_PIN_SECONDS:
            if os.path.exists(self._abs_run_dir(self._st.run_id)):
                return self._st.run_id

        runs = _list_runs(self._log_root)
        if not runs:
            return ""

        best: Optional[Tuple[Tuple[int, int, int, float], str]] = None
        for run_id in runs:
            run_dir = self._abs_run_dir(run_id)
            ev = _best_event_file_in_dir(run_dir, self._prefer)
            if ev is None:
                continue
            s = _score_event_file(ev, self._prefer)
            if s is None:
                continue
            if best is None or s > best[0]:
                best = (s, run_id)

        picked = best[1] if best is not None else runs[-1]
        self._last_pin_time = now
        return picked

    def _pin_run(self, run_id: str) -> bool:
        run_id = str(run_id or "")
        if not run_id:
            run_id = self._auto_pick_run()

        run_dir_abs = self._abs_run_dir(run_id)
        if run_id and not os.path.exists(run_dir_abs):
            self._reset_state()
            run_id = self._auto_pick_run()
            run_dir_abs = self._abs_run_dir(run_id)

        if self._st.run_id != run_id or self._st.run_dir_abs != run_dir_abs:
            self._reset_state()
            self._st.run_id = run_id
            self._st.run_dir_abs = run_dir_abs
            return True
        return False

    def _maybe_switch_event_file(self) -> bool:
        if not self._st.run_dir_abs:
            return False

        latest = _best_event_file_in_dir(self._st.run_dir_abs, self._prefer)
        if latest is None:
            if self._st.log_file is not None:
                self._reset_state()
                return True
            return False

        try:
            mtime_ns = os.stat(latest).st_mtime_ns
        except Exception:
            mtime_ns = 0

        if self._st.log_file != latest:
            self._series = {}
            self._known_tags = []
            self._st.log_file = latest
            self._st.log_mtime_ns = mtime_ns
            self._st.ea = EventAccumulator(latest, size_guidance={"scalars": 0})
            return True

        self._st.log_mtime_ns = mtime_ns
        return False

    def _reset_state(self) -> None:
        self._st = _CacheState()
        self._series = {}
        self._known_tags = []

    def _reload(self) -> None:
        if self._st.log_file is None or self._st.ea is None:
            return
        now = time.monotonic()
        if (now - self._st.last_reload_monotonic) < MIN_RELOAD_INTERVAL_S:
            return
        self._st.last_reload_monotonic = now
        try:
            self._st.ea.Reload()
        except Exception:
            return
        tags = self._st.ea.Tags().get("scalars", [])
        self._known_tags = sorted(tags)

    def list_runs(self) -> List[str]:
        return _list_runs(self._log_root)

    def list_tags(self, *, run_id: str) -> Tuple[str, List[str]]:
        with self._lock:
            self._pin_run(run_id)
            self._maybe_switch_event_file()
            self._reload()
            return self._st.run_id, list(self._known_tags)

    def _choose_default_tag(self, tags: List[str]) -> Optional[str]:
        if not tags:
            return None
        # Try exact preferred matches first
        for pat in self._prefer:
            for t in tags:
                if pat in t:
                    return t
        
        # Fallbacks
        for t in tags:
            if "WeightedLoss" in t:
                return t
        for t in tags:
            if "Loss" in t or "loss" in t:
                return t
        return tags[0]

    def _append_new_events(self, tag: str) -> None:
        if self._st.ea is None:
            return
        if tag not in self._known_tags:
            return
        if tag not in self._series:
            self._series[tag] = ([], [], -1)

        steps, vals, last_seen = self._series[tag]
        try:
            events = self._st.ea.Scalars(tag)
        except Exception:
            return
        if not events:
            return

        # Optimization: only process events strictly after last_seen
        # TB events are usually sorted, but we check to be safe
        for e in events:
            if e.step <= last_seen:
                continue
            steps.append(int(e.step))
            vals.append(float(e.value))
            last_seen = int(e.step)

        if len(steps) > MAX_CACHE_POINTS:
            cut = len(steps) - MAX_CACHE_POINTS
            del steps[:cut]
            del vals[:cut]

        self._series[tag] = (steps, vals, last_seen)

    def get_series(
        self,
        *,
        run_id: str,
        skip: int,
        since_step: Optional[int],
        max_points: int,
        requested_tag: Optional[str],
    ) -> Tuple[
        bool, str, List[int], List[float], Optional[int], Optional[str], Dict[str, Dict[str, float]], List[Dict[str, float]]
    ]:
        skip = max(0, int(skip))
        max_points = max(500, int(max_points))

        with self._lock:
            reset = self._pin_run(run_id)
            reset_file = self._maybe_switch_event_file()
            reset = bool(reset or reset_file)
            self._reload()

            if self._st.log_file is None or self._st.ea is None or not self._known_tags:
                return reset, self._st.run_id, [], [], None, None, {}, []

            tag = requested_tag if (requested_tag and requested_tag in self._known_tags) else self._choose_default_tag(self._known_tags)
            if tag is None:
                return reset, self._st.run_id, [], [], None, None, {}, []

            self._append_new_events(tag)
            steps, vals, _ = self._series.get(tag, ([], [], -1))
            if not steps:
                return reset, self._st.run_id, [], [], None, tag, {}, []

            # Populate Stats
            latest: Dict[str, Dict[str, float]] = {}
            for st_tag in self._stat_tags:
                if st_tag in self._known_tags:
                    self._append_new_events(st_tag)
                    s2, v2, _ls = self._series.get(st_tag, ([], [], -1))
                    if s2:
                        latest[st_tag] = {"step": float(s2[-1]), "value": float(v2[-1])}

            latest_extras: List[Dict[str, float]] = []
            for st_tag in self._stat_tags:
                if st_tag == tag:
                    continue
                v = latest.get(st_tag)
                if v is None:
                    continue
                latest_extras.append({"tag": st_tag, "step": float(v["step"]), "value": float(v["value"])})
                if len(latest_extras) >= 2:
                    break

            import bisect
            i0 = bisect.bisect_left(steps, skip)
            steps2 = steps[i0:]
            vals2 = vals[i0:]
            if not steps2:
                return reset, self._st.run_id, [], [], None, tag, latest, latest_extras

            last_available = steps2[-1]

            if since_step is not None and since_step >= last_available:
                ds_steps, ds_vals = _downsample_stride(steps2, vals2, max_points)
                return True, self._st.run_id, ds_steps, ds_vals, (ds_steps[-1] if ds_steps else None), tag, latest, latest_extras

            if since_step is not None:
                if since_step < skip:
                    ds_steps, ds_vals = _downsample_stride(steps2, vals2, max_points)
                    return True, self._st.run_id, ds_steps, ds_vals, (ds_steps[-1] if ds_steps else None), tag, latest, latest_extras

                j0 = bisect.bisect_right(steps2, since_step)
                tail_steps = steps2[j0:]
                tail_vals = vals2[j0:]
                return reset, self._st.run_id, tail_steps, tail_vals, last_available, tag, latest, latest_extras

            ds_steps, ds_vals = _downsample_stride(steps2, vals2, max_points)
            return True, self._st.run_id, ds_steps, ds_vals, (ds_steps[-1] if ds_steps else None), tag, latest, latest_extras


# --- INIT CACHES (one per MODE) ---
# We configure the caches, which will perform absolute path resolution on init
_caches: Dict[str, TensorboardScalarCache] = {
    mode: TensorboardScalarCache(
        root,
        prefer_tag_substrings=_PREFER_BY_MODE.get(mode, ["Loss", "loss"]),
        stat_tags=_STAT_TAGS_BY_MODE.get(mode, []),
    )
    for mode, root in LOG_DIRS.items()
}


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, modes=list(LOG_DIRS.keys()))


@app.route("/api/runs")
def get_runs():
    mode = request.args.get("mode", "Battle")
    if mode not in _caches:
        return jsonify({"error": f"Unknown mode: {mode}", "runs": []})
    try:
        runs = _caches[mode].list_runs()
        return jsonify({"runs": runs})
    except Exception as e:
        return jsonify({"error": str(e), "runs": []})


@app.route("/api/tags")
def get_tags():
    mode = request.args.get("mode", "Battle")
    run_id = request.args.get("run", "")  # empty => auto
    if mode not in _caches:
        return jsonify({"error": f"Unknown mode: {mode}", "tags": []})
    try:
        selected_run, tags = _caches[mode].list_tags(run_id=run_id)
        return jsonify({"run": selected_run, "tags": tags})
    except Exception as e:
        return jsonify({"error": str(e), "tags": []})


@app.route("/api/data")
def get_data():
    mode = request.args.get("mode", "Battle")
    run_id = request.args.get("run", "")  # empty => auto-pin
    if mode not in _caches:
        return jsonify({"error": f"Unknown mode: {mode}"})

    skip_n = _safe_int(request.args.get("skip", "100"), 100)
    max_points = _safe_int(request.args.get("max_points", "5000"), 5000)
    since_raw = request.args.get("since", None)
    req_tag = request.args.get("tag", None)

    since_step = None
    if since_raw is not None and since_raw != "":
        s = _safe_int(since_raw, -1)
        if s >= 0:
            since_step = s

    try:
        reset, selected_run, steps, values, last_step, tag, latest, latest_extras = _caches[mode].get_series(
            run_id=run_id,
            skip=skip_n,
            since_step=since_step,
            max_points=max_points,
            requested_tag=req_tag,
        )

        if tag is None:
            return jsonify({"error": f"Waiting for {mode} logs...", "run": selected_run})

        return jsonify(
            {
                "reset": bool(reset),
                "run": selected_run,
                "steps": steps,
                "values": values,
                "last_step": last_step,
                "tag": tag,
                "latest": latest,
                "latest_extras": latest_extras,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n📊 Monitor running at:")
    print(f"   👉 http://127.0.0.1:{PORT}")
    print(f"   👉 http://{local_ip}:{PORT} (Local Network)\n")
    print(f"   Watching modes:")
    for k, v in LOG_DIRS.items():
        # Print the resolved absolute path so user can verify
        abs_path = os.path.normpath(os.path.abspath(v))
        exists = "✅" if os.path.exists(abs_path) else "❌ (Folder not found)"
        print(f"   - {k}: {abs_path} {exists}")

    app.run(host=HOST, port=PORT, debug=False, threaded=True)