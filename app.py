from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import json
import glob

app = Flask(__name__)

TRAINING_DATA_DIR = 'training_data'

@app.route('/')
def index():
    # Get list of folders in training_data
    folders = [name for name in os.listdir(TRAINING_DATA_DIR)
               if os.path.isdir(os.path.join(TRAINING_DATA_DIR, name))]
    return render_template('index.html', folders=folders)

@app.route('/folder/<folder_name>')
def folder_view(folder_name):
    # Check if folder exists
    folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return "Folder not found", 404
    return render_template('viewer.html', folder_name=folder_name)

@app.route('/folder/<folder_name>/frame/<int:frame_index>')
def frame_data(folder_name, frame_index):
    folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return "Folder not found", 404
    
    # Get list of JSON files sorted by timestamp in filenames
    json_files = sorted(glob.glob(os.path.join(folder_path, '*.json')))
    total_frames = len(json_files)
    if frame_index < 0 or frame_index >= total_frames:
        return "Frame index out of range", 404
    
    # Load current frame data
    current_frame_file = json_files[frame_index]
    with open(current_frame_file, 'r') as f:
        current_data = json.load(f)
    
    # Check if the player is the winner by reading the winner.json file
    winner_file = os.path.join(folder_path, 'winner.json')
    is_winner = None
    if os.path.exists(winner_file):
        with open(winner_file, 'r') as f:
            winner_data = json.load(f)
            is_winner = winner_data.get('is_winner', None)  # Adjust key based on winner.json structure
    # Find next reward/punishment
    frames_ahead = None
    reward_punishment = None
    for i in range(frame_index + 1, total_frames):
        next_frame_file = json_files[i]
        with open(next_frame_file, 'r') as f:
            next_data = json.load(f)
        if next_data.get('reward') is not None or next_data.get('punishment') is not None:
            frames_ahead = i - frame_index
            if next_data.get('reward') is not None:
                reward_punishment = {'type': 'reward', 'value': next_data['reward']}
            else:
                reward_punishment = {'type': 'punishment', 'value': next_data['punishment']}
            break
    
    # Prepare response data
    response_data = {
        'current_frame': frame_index,
        'total_frames': total_frames,
        'image_path': current_data['image_path'],
        'input': current_data['input'],
        'reward': current_data.get('reward'),
        'punishment': current_data.get('punishment'),
        'next_reward_punishment': {
            'frames_ahead': frames_ahead,
            'type': reward_punishment['type'] if reward_punishment else None,
            'value': reward_punishment['value'] if reward_punishment else None
        } if frames_ahead is not None else None,
        'winner': is_winner  # Include the winner status from winner.json
    }
    return jsonify(response_data)

@app.route('/training_data/<path:filename>')
def training_data_files(filename):
    return send_from_directory(TRAINING_DATA_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
