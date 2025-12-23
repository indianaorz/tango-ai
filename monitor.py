import os
import glob
import json
import socket
from flask import Flask, render_template_string, jsonify, request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

app = Flask(__name__)

# --- CONFIG ---
LOG_DIR = "logs/tango_cached"
HOST = "0.0.0.0" 
PORT = 6007 

# HTML Template with Zero-Phase Smoothing
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
        
        /* Dashboard Header */
        .stat-box { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat { background: #222; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #333; }
        .stat h3 { margin: 0 0 5px 0; font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .stat div { font-size: 32px; font-weight: bold; color: #4db8ff; }
        
        /* Controls */
        .controls { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 30px; align-items: center; flex-wrap: wrap; border: 1px solid #333; }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        input[type=number] { background: #333; border: 1px solid #555; color: #fff; padding: 8px; border-radius: 4px; width: 80px; }
        input[type=range] { width: 150px; accent-color: #4db8ff; }
        label { font-size: 12px; color: #aaa; font-weight: bold; }
        
        button { background:#4db8ff; color:#000; border:none; padding:10px 20px; border-radius:4px; cursor:pointer; font-weight:bold; transition: background 0.2s; }
        button:hover { background: #3aa8eb; }

        /* Chart */
        #chart { width: 100%; height: 70vh; background: #000; border-radius: 8px; border: 1px solid #333; }
        .status { font-size: 12px; color: #666; margin-top: 5px; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
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
                <input type="number" id="skip" value="100" min="0" step="100" onchange="fetchData()">
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
                <button onclick="fetchData()">Update Now</button>
            </div>
        </div>

        <div id="chart"></div>
        <div class="status" id="status">Initializing...</div>
    </div>

    <script>
        let rawData = { steps: [], values: [] };
        
        // --- NEW: Bidirectional Exponential Smoothing (Zero-Phase) ---
        // This runs the filter Forward AND Backward to cancel out the lag.
        function smooth(values, alpha) {
            if (values.length === 0) return [];
            if (alpha === 0) return values;

            const n = values.length;
            
            // 1. Forward Pass
            let forward = new Float32Array(n);
            let curr = values[0];
            forward[0] = curr;
            
            for (let i = 1; i < n; i++) {
                curr = curr * alpha + (1 - alpha) * values[i];
                forward[i] = curr;
            }

            // 2. Backward Pass (Fixes the startup lag)
            let backward = new Float32Array(n);
            curr = forward[n - 1];
            backward[n - 1] = curr;
            
            for (let i = n - 2; i >= 0; i--) {
                curr = curr * alpha + (1 - alpha) * forward[i];
                backward[i] = curr;
            }

            // Float32Array isn't directly serializable by some plotters, convert to regular array
            return Array.from(backward);
        }

        function updateSmoothing() {
            const alpha = parseFloat(document.getElementById('smooth').value);
            document.getElementById('smooth-val').innerText = alpha.toFixed(2);
            renderChart();
        }

        function renderChart() {
            if (rawData.steps.length === 0) return;

            const alpha = parseFloat(document.getElementById('smooth').value);
            const smoothedVals = smooth(rawData.values, alpha);

            // --- SMART ZOOM LOGIC ---
            // Calculate Min/Max of the SMOOTHED line only
            let minVal = Infinity;
            let maxVal = -Infinity;
            for(let v of smoothedVals) {
                if(v < minVal) minVal = v;
                if(v > maxVal) maxVal = v;
            }

            // Add 5% padding top/bottom so lines don't touch edges
            const range = maxVal - minVal;
            const padding = (range === 0) ? 0.1 : range * 0.05; 
            const yMin = Math.max(0, minVal - padding); // Don't go below 0
            const yMax = maxVal + padding;
            // ------------------------

            // 1. Ghost Trace (Raw Data)
            const traceRaw = {
                x: rawData.steps,
                y: rawData.values,
                mode: 'lines',
                name: 'Raw',
                line: { color: 'rgba(0, 255, 204, 0.15)', width: 1 }, 
                hoverinfo: 'none' 
            };

            // 2. Main Trace (Smoothed)
            const traceSmooth = {
                x: rawData.steps,
                y: smoothedVals,
                mode: 'lines',
                name: 'Smoothed',
                line: { color: '#00ffcc', width: 2.5 },
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 255, 204, 0.05)'
            };

            const layout = {
                title: 'Training Loss (Bidirectional Smoothing)',
                paper_bgcolor: '#000',
                plot_bgcolor: '#000',
                font: { color: '#eee' },
                xaxis: { title: 'Global Step', gridcolor: '#333', zerolinecolor: '#444' },
                yaxis: { 
                    title: 'Loss', 
                    gridcolor: '#333', 
                    zerolinecolor: '#444',
                    range: [yMin, yMax], 
                    fixedrange: false    
                },
                margin: { t: 40, l: 60, r: 20, b: 60 },
                showlegend: false
            };

            Plotly.react('chart', [traceRaw, traceSmooth], layout);
        }

        async function fetchData() {
            const skip = document.getElementById('skip').value;
            const status = document.getElementById('status');
            
            try {
                // status.innerText = "Fetching new data...";
                const res = await fetch(`/api/data?skip=${skip}`);
                const data = await res.json();
                
                if (data.error) {
                    status.innerText = "Error: " + data.error;
                    return;
                }

                if (data.steps.length > 0) {
                    rawData = data;
                    
                    const lastStep = data.steps[data.steps.length - 1];
                    const lastLoss = data.values[data.values.length - 1];
                    
                    document.getElementById('curr-step').innerText = lastStep.toLocaleString();
                    document.getElementById('curr-loss').innerText = lastLoss.toFixed(4);
                    
                    renderChart();
                    status.innerText = `Last updated: ${new Date().toLocaleTimeString()} (${data.steps.length} points)`;
                } else {
                    status.innerText = "No data points found yet.";
                }
                
            } catch (e) {
                status.innerText = "Connection Failed";
            }
        }

        // Init
        fetchData();
        setInterval(() => {
            if(document.getElementById('refresh').checked) fetchData();
        }, 1000); 
    </script>
</body>
</html>
"""

def get_latest_log():
    files = glob.glob(os.path.join(LOG_DIR, "events.out.tfevents.*"))
    if not files: return None
    return max(files, key=os.path.getctime)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    skip_n = int(request.args.get('skip', 100))
    log_file = get_latest_log()
    
    if not log_file:
        return jsonify({"error": "No log files found"})

    try:
        ea = EventAccumulator(log_file)
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        loss_tag = next((t for t in tags if 'Loss' in t), None)
        
        if not loss_tag:
            return jsonify({"error": "Waiting for Training to start..."})

        events = ea.Scalars(loss_tag)
        
        steps = []
        values = []
        
        for e in events:
            if e.step < skip_n: continue
            steps.append(e.step)
            values.append(e.value)
            
        return jsonify({"steps": steps, "values": values})
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n📊 Monitor running at:")
    print(f"   👉 http://127.0.0.1:{PORT}")
    print(f"   👉 http://{local_ip}:{PORT} (Local Network)\n")
    
    app.run(host=HOST, port=PORT, debug=False)