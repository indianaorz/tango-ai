from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import json
import glob
import torch  # Import torch to load .pt files

app = Flask(__name__)

TRAINING_DATA_DIR = 'training_data'
TRAINING_CACHE_DIR = 'training_cache'  # Add the cache directory path

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

    # Initialize variables to track next reward and punishment
    next_reward = None
    next_punishment = None
    frames_ahead_reward = None
    frames_ahead_punishment = None

    # Iterate through future frames to find the next reward and punishment
    for i in range(frame_index + 1, total_frames):
        next_frame_file = json_files[i]
        with open(next_frame_file, 'r') as f:
            next_data = json.load(f)

        # Find the next reward if not already found
        if next_reward is None and next_data.get('reward') is not None:
            next_reward = next_data['reward']
            frames_ahead_reward = i - frame_index

        # Find the next punishment if not already found
        if next_punishment is None and next_data.get('punishment') is not None:
            next_punishment = next_data['punishment']
            frames_ahead_punishment = i - frame_index

        # Break early if both are found
        if next_reward is not None and next_punishment is not None:
            break

    # Load the .pt file corresponding to the frame index
    cache_folder_path = os.path.join(TRAINING_CACHE_DIR, folder_name)
    pt_file_name = f'{frame_index:06d}.pt'
    pt_file_path = os.path.join(cache_folder_path, pt_file_name)

    if os.path.exists(pt_file_path):
        sample = torch.load(pt_file_path)
        # Extract net_reward and input tensor
        pt_net_reward = sample.get('net_reward')
        input_tensor = sample.get('input')  # This is a tensor
        # Convert input_tensor to list or string to send in JSON
        input_tensor_list = input_tensor.tolist()
    else:
        pt_net_reward = None
        input_tensor_list = None

    # Prepare response data
    response_data = {
        'current_frame': frame_index,
        'total_frames': total_frames,
        'image_path': current_data['image_path'],
        'input': current_data['input'],
        'reward': current_data.get('reward'),
        'punishment': current_data.get('punishment'),
        'next_reward': {
            'value': next_reward,
            'frames_ahead': frames_ahead_reward
        } if next_reward is not None else None,
        'next_punishment': {
            'value': next_punishment,
            'frames_ahead': frames_ahead_punishment
        } if next_punishment is not None else None,
        'winner': is_winner,  # Include the winner status from winner.json
        'pt_net_reward': pt_net_reward,  # Add net_reward from .pt file
        'pt_input_tensor': input_tensor_list  # Add input tensor from .pt file
    }
    return jsonify(response_data)

@app.route('/training_data/<path:filename>')
def training_data_files(filename):
    return send_from_directory(TRAINING_DATA_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
