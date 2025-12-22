import os
import time
import socket
import math
from flask import Flask, render_template_string, send_from_directory, abort

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints"  # Folder to serve
PORT = 6001                     # Port to run on

app = Flask(__name__)

# Dark Mode Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NitroGen Checkpoint Server</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; color: #4db8ff; }
        .file-list { display: flex; flex-direction: column; gap: 10px; }
        .file-card { 
            background: #222; border: 1px solid #333; padding: 15px; border-radius: 8px; 
            display: flex; justify-content: space-between; align-items: center;
            transition: background 0.2s;
        }
        .file-card:hover { background: #2a2a2a; border-color: #555; }
        .file-info { display: flex; flex-direction: column; }
        .filename { font-size: 1.2em; font-weight: bold; color: #fff; }
        .details { font-size: 0.9em; color: #aaa; margin-top: 5px; }
        .download-btn {
            background: #4db8ff; color: #000; text-decoration: none; padding: 10px 20px; 
            border-radius: 5px; font-weight: bold;
        }
        .download-btn:hover { background: #3aa8eb; }
        .latest-tag { background: #4caf50; color: #fff; font-size: 0.7em; padding: 2px 6px; border-radius: 4px; margin-left: 10px; vertical-align: middle; }
    </style>
</head>
<body>
    <h1>💾 Checkpoint Server</h1>
    <p>Serving from: <code>{{ folder }}</code></p>
    
    <div class="file-list">
        {% for file in files %}
        <div class="file-card">
            <div class="file-info">
                <div class="filename">
                    {{ file.name }}
                    {% if loop.index0 == 0 %}<span class="latest-tag">LATEST</span>{% endif %}
                </div>
                <div class="details">Size: {{ file.size }} • Modified: {{ file.date }}</div>
            </div>
            <a href="/download/{{ file.name }}" class="download-btn">⬇ Download</a>
        </div>
        {% endfor %}
        
        {% if not files %}
        <div style="text-align:center; color:#666; padding: 40px;">No checkpoints found in folder.</div>
        {% endif %}
    </div>
</body>
</html>
"""

def convert_size(size_bytes):
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

@app.route('/')
def index():
    if not os.path.exists(CHECKPOINT_DIR):
        return f"Error: Folder '{CHECKPOINT_DIR}' not found.", 404

    # List all files
    files = []
    for f in os.listdir(CHECKPOINT_DIR):
        path = os.path.join(CHECKPOINT_DIR, f)
        if os.path.isfile(path) and f.endswith(".pt"):
            stats = os.stat(path)
            files.append({
                "name": f,
                "size": convert_size(stats.st_size),
                "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime)),
                "mtime": stats.st_mtime
            })
    
    # Sort by newest first
    files.sort(key=lambda x: x["mtime"], reverse=True)
    
    return render_template_string(HTML_TEMPLATE, files=files, folder=os.path.abspath(CHECKPOINT_DIR))

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(CHECKPOINT_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🚀 Server running!")
    print(f"   📂 Serving: {os.path.abspath(CHECKPOINT_DIR)}")
    print(f"   👉 Local Access:  http://127.0.0.1:{PORT}")
    print(f"   👉 Linux Access:  http://{local_ip}:{PORT}")
    print("\n(Press Ctrl+C to stop)\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)