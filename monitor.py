import os
import glob
import json
import socket
from flask import Flask, render_template_string, jsonify, request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

app = Flask(__name__)

# --- CONFIG ---
LOG_DIR = "logs/tango_cached"
HOST = "0.0.0.0"  # Allows local network access
PORT = 6007

# HTML Template with Plotly.js for interactive graphs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NitroGen Training Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .controls { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        input { background: #333; border: 1px solid #444; color: #fff; padding: 8px; border-radius: 4px; }
        label { font-size: 14px; color: #aaa; }
        .stat-box { display: flex; gap: 20px; margin-bottom: 10px; }
        .stat { background: #222; padding: 10px 20px; border-radius: 8px; flex: 1; text-align: center; }
        .stat h3 { margin: 0 0 5px 0; font-size: 14px; color: #888; }
        .stat div { font-size: 24px; font-weight: bold; color: #4db8ff; }
        #chart { width: 100%; height: 60vh; background: #000; border-radius: 8px; border: 1px solid #333; }
        .status { font-size: 12px; color: #666; margin-top: 10px; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
        <div class="stat-box">
            <div class="stat">
                <h3>Current Step</h3>
                <div id="curr-step">---</div>
            </div>
            <div class="stat">
                <h3>Current Loss</h3>
                <div id="curr-loss" style="color: #ff4d4d">---</div>
            </div>
        </div>

        <div class="controls">
            <div>
                <label>Ignore First N Steps:</label>
                <input type="number" id="skip" value="100" min="0" step="100">
            </div>
            <div>
                <label>Auto-Refresh:</label>
                <input type="checkbox" id="refresh" checked>
            </div>
            <div style="flex-grow:1; text-align:right;">
                <button onclick="fetchData()" style="background:#4db8ff; color:#000; border:none; padding:8px 16px; border-radius:4px; cursor:pointer;">Update Now</button>
            </div>
        </div>

        <div id="chart"></div>
        <div class="status" id="status">Waiting for data...</div>
    </div>

    <script>
        let plotData = { x: [], y: [] };
        
        async function fetchData() {
            const skip = document.getElementById('skip').value;
            const status = document.getElementById('status');
            
            try {
                status.innerText = "Fetching...";
                const res = await fetch(`/api/data?skip=${skip}`);
                const data = await res.json();
                
                if (data.error) {
                    status.innerText = "Error: " + data.error;
                    return;
                }

                // Update Stats
                if (data.steps.length > 0) {
                    const lastStep = data.steps[data.steps.length - 1];
                    const lastLoss = data.values[data.values.length - 1];
                    document.getElementById('curr-step').innerText = lastStep.toLocaleString();
                    document.getElementById('curr-loss').innerText = lastLoss.toFixed(4);
                }

                // Plot
                const trace = {
                    x: data.steps,
                    y: data.values,
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: '#00ffcc', width: 2 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 255, 204, 0.1)'
                };

                const layout = {
                    title: 'Training Loss',
                    paper_bgcolor: '#000',
                    plot_bgcolor: '#000',
                    font: { color: '#eee' },
                    xaxis: { title: 'Global Step', gridcolor: '#333' },
                    yaxis: { title: 'Loss', gridcolor: '#333' },
                    margin: { t: 40, l: 50, r: 20, b: 40 }
                };

                Plotly.newPlot('chart', [trace], layout);
                status.innerText = `Last updated: ${new Date().toLocaleTimeString()} (${data.steps.length} points)`;
                
            } catch (e) {
                status.innerText = "Connection Failed";
            }
        }

        // Init
        fetchData();
        setInterval(() => {
            if(document.getElementById('refresh').checked) fetchData();
        }, 10000); // Auto-refresh every 10s
    </script>
</body>
</html>
"""

def get_latest_log():
    # Find latest event file
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
        
        # Check available tags
        tags = ea.Tags()['scalars']
        loss_tag = next((t for t in tags if 'Loss' in t), None)
        
        if not loss_tag:
            return jsonify({"error": "No 'Loss' scalar found yet. Training might be initializing."})

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
    # Get local IP for convenience print
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n📊 Monitor running at:")
    print(f"   👉 http://127.0.0.1:{PORT}")
    print(f"   👉 http://{local_ip}:{PORT} (Local Network)\n")
    
    app.run(host=HOST, port=PORT, debug=False)