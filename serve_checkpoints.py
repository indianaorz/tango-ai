import os
import time
import socket
import math
from flask import Flask, render_template_string, send_from_directory, abort

# --- CONFIG ---
CHECKPOINT_ROOT = "checkpoints"  # Root folder to serve
PORT = 6001                      # Port to run on

app = Flask(__name__)

# Dark Mode Template with Breadcrumbs and Folders
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NitroGen Checkpoint Server</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; color: #4db8ff; display: flex; align-items: center; gap: 10px; }
        a { color: #4db8ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        
        .breadcrumbs { font-size: 1.1em; color: #888; margin-bottom: 20px; }
        .breadcrumbs a { font-weight: bold; }
        .breadcrumbs span { margin: 0 5px; color: #555; }

        .file-list { display: flex; flex-direction: column; gap: 10px; }
        
        .card { 
            background: #222; border: 1px solid #333; padding: 15px; border-radius: 8px; 
            display: flex; justify-content: space-between; align-items: center;
            transition: background 0.2s;
        }
        .card:hover { background: #2a2a2a; border-color: #555; }
        
        /* Folder Specifics */
        .card.folder { border-left: 4px solid #ffcc00; }
        .card.folder .icon { font-size: 1.5em; margin-right: 15px; }
        
        /* File Specifics */
        .card.file { border-left: 4px solid #4db8ff; }
        .file-info { display: flex; flex-direction: column; }
        .filename { font-size: 1.1em; font-weight: bold; color: #fff; }
        .details { font-size: 0.85em; color: #aaa; margin-top: 5px; }
        
        .download-btn {
            background: #4db8ff; color: #000; text-decoration: none; padding: 8px 16px; 
            border-radius: 4px; font-weight: bold; font-size: 0.9em;
        }
        .download-btn:hover { background: #3aa8eb; text-decoration: none; }
        
        .latest-tag { background: #4caf50; color: #fff; font-size: 0.7em; padding: 2px 6px; border-radius: 4px; margin-left: 10px; vertical-align: middle; }
    </style>
</head>
<body>
    <h1>💾 Checkpoint Server</h1>
    
    <div class="breadcrumbs">
        <a href="/">root</a>
        {% for part in breadcrumbs %}
            <span>/</span> <a href="/browse/{{ part.path }}">{{ part.name }}</a>
        {% endfor %}
    </div>
    
    <div class="file-list">
        {% for folder in folders %}
        <a href="/browse/{{ folder.rel_path }}" class="card folder" style="text-decoration:none; color:inherit;">
            <div style="display:flex; align-items:center;">
                <span class="icon">📁</span>
                <span class="filename">{{ folder.name }}</span>
            </div>
            <div style="color:#666;">DIR</div>
        </a>
        {% endfor %}

        {% for file in files %}
        <div class="card file">
            <div class="file-info">
                <div class="filename">
                    {{ file.name }}
                    {% if loop.index0 == 0 %}<span class="latest-tag">LATEST</span>{% endif %}
                </div>
                <div class="details">Size: {{ file.size }} • Modified: {{ file.date }}</div>
            </div>
            <a href="/download/{{ file.rel_path }}" class="download-btn">⬇ Download</a>
        </div>
        {% endfor %}
        
        {% if not files and not folders %}
        <div style="text-align:center; color:#666; padding: 40px;">Empty folder.</div>
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

def get_breadcrumbs(rel_path):
    if not rel_path: return []
    parts = rel_path.strip('/').split('/')
    crumbs = []
    current = ""
    for p in parts:
        if not p: continue
        current = os.path.join(current, p).replace("\\", "/")
        crumbs.append({"name": p, "path": current})
    return crumbs

@app.route('/')
@app.route('/browse/<path:subpath>')
def browse(subpath=""):
    # Security check: prevent directory traversal
    if ".." in subpath or subpath.startswith("/"):
        abort(403)
        
    abs_path = os.path.join(CHECKPOINT_ROOT, subpath)
    
    if not os.path.exists(abs_path):
        return f"Error: Folder '{subpath}' not found.", 404

    # Separate folders and files
    folders = []
    files = []
    
    try:
        for f in os.listdir(abs_path):
            full_path = os.path.join(abs_path, f)
            rel_path = os.path.join(subpath, f).replace("\\", "/")
            
            if os.path.isdir(full_path):
                folders.append({
                    "name": f,
                    "rel_path": rel_path
                })
            elif os.path.isfile(full_path) and f.endswith(".pt"):
                stats = os.stat(full_path)
                files.append({
                    "name": f,
                    "rel_path": rel_path,
                    "size": convert_size(stats.st_size),
                    "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime)),
                    "mtime": stats.st_mtime
                })
    except PermissionError:
        abort(403)

    # Sort files by newest first
    files.sort(key=lambda x: x["mtime"], reverse=True)
    folders.sort(key=lambda x: x["name"])

    return render_template_string(
        HTML_TEMPLATE, 
        files=files, 
        folders=folders, 
        breadcrumbs=get_breadcrumbs(subpath)
    )

@app.route('/download/<path:filename>')
def download(filename):
    # Security check
    if ".." in filename or filename.startswith("/"):
        abort(403)
    return send_from_directory(CHECKPOINT_ROOT, filename, as_attachment=True)

if __name__ == '__main__':
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    if not os.path.exists(CHECKPOINT_ROOT):
        os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    print(f"\n🚀 Recursive Checkpoint Server Running!")
    print(f"   📂 Root: {os.path.abspath(CHECKPOINT_ROOT)}")
    print(f"   👉 Local: http://127.0.0.1:{PORT}")
    print(f"   👉 Network: http://{local_ip}:{PORT}")
    print("\n(Press Ctrl+C to stop)\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)