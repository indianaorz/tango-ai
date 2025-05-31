from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import json
import glob
import torch  # Import torch to load .pt files
import numpy as np  # Import numpy for type conversion if needed
from PIL import Image  # Import PIL for image loading
from train import GameInputPredictor  # Import the model class
from utils import get_checkpoint_path, get_exponential_sample, get_image_memory, get_threshold, get_root_dir

app = Flask(__name__)

# Define directories
TRAINING_DATA_DIR = os.path.join(get_root_dir(), 'training_data')
TRAINING_CACHE_DIR = os.path.join(get_root_dir(), 'training_cache')  # Ensure this path is correct

# Global configuration
USE_MODEL = False  # Set to False to disable model loading and inference
checkpoint_path = os.path.join(get_root_dir(), 'checkpoints')
IMAGE_MEMORY = get_image_memory()  # Adjust as needed or make dynamic

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, image_memory=1):
    """Load the AI model if USE_MODEL is True."""
    global model
    if USE_MODEL:
        model = GameInputPredictor(image_memory=image_memory).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("Model loading skipped because USE_MODEL is set to False.")

@app.route('/')
def index():
    """
    Home route that lists all available folders in the training data directory.
    """
    # Get list of folders in training_data
    folders = [name for name in os.listdir(TRAINING_DATA_DIR)
               if os.path.isdir(os.path.join(TRAINING_DATA_DIR, name))]
    return render_template('index.html', folders=folders)


@app.route('/folder/<folder_name>')
def folder_view(folder_name):
    """
    Route to view a specific folder's contents.
    Now also passes a sorted list of all folders and the current folder's index.
    """
    # Get list of folders sorted alphabetically
    folders = sorted([name for name in os.listdir(TRAINING_DATA_DIR)
                      if os.path.isdir(os.path.join(TRAINING_DATA_DIR, name))])
    
    if folder_name not in folders:
        return "Folder not found", 404

    current_index = folders.index(folder_name)
    
    return render_template('viewer.html', 
                           folder_name=folder_name, 
                           image_memory=IMAGE_MEMORY,
                           folders=folders,
                           current_index=current_index)

@app.route('/folder/<folder_name>/frame/<int:frame_index>')
def frame_data(folder_name, frame_index):
    """
    Route to fetch data for a specific frame within a folder.
    Includes new data points: player_health, enemy_health, player_position, enemy_position, inside_window.
    """
    folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return "Folder not found", 404

    # Get list of JSON files sorted by timestamp in filenames
    json_files = sorted(glob.glob(os.path.join(folder_path, '*.json')))
    json_files = [f for f in json_files if os.path.basename(f) != 'winner.json']
    total_frames = len(json_files)
    if frame_index < 0 or frame_index >= total_frames:
        return "Frame index out of range", 404

    # Calculate the start index based on image_memory
    frame_indices = get_exponential_sample(list(range(total_frames)), frame_index, IMAGE_MEMORY)
    while len(frame_indices) < IMAGE_MEMORY:
        frame_indices.insert(0, 0)

    # Load frame data for the sequence
    frames_data = []
    for idx in frame_indices:
        frame_file = json_files[idx]
        with open(frame_file, 'r') as f:
            frame_data = json.load(f)
            # Modify image_path to be relative to folder_name
            image_filename = os.path.basename(frame_data.get('image_path', ''))
            frames_data.append({
                'image_path': image_filename,  # Only the filename
                'input': frame_data.get('input', ''),
                'reward': frame_data.get('reward', None),
                'punishment': frame_data.get('punishment', None),
            })

    # Load current frame data
    current_frame_file = json_files[frame_index]
    with open(current_frame_file, 'r') as f:
        current_data = json.load(f)
        # Modify image_path to be relative to folder_name
        current_image_filename = os.path.basename(current_data.get('image_path', ''))

    # Check if the player is the winner
    winner_file = os.path.join(folder_path, 'winner.json')
    is_winner = None
    if os.path.exists(winner_file):
        with open(winner_file, 'r') as f:
            try:
                winner_data = json.load(f)
                is_winner = winner_data.get('is_winner', None)
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {winner_file}, skipping this folder.")
                return "Invalid winner.json format", 400

    # Initialize variables to track next reward and punishment
    next_reward = None
    next_punishment = None
    frames_ahead_reward = None
    frames_ahead_punishment = None

    # Iterate through future frames to find the next non-zero reward and punishment
    for i in range(frame_index + 1, total_frames):
        next_frame_file = json_files[i]
        with open(next_frame_file, 'r') as f:
            try:
                next_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {next_frame_file}, skipping this frame.")
                continue  # Skip this frame due to invalid JSON

        # Find the next reward if not already found
        if next_reward is None and next_data.get('reward') not in [None, 0]:
            next_reward = next_data['reward']
            frames_ahead_reward = i - frame_index

        # Find the next punishment if not already found
        if next_punishment is None and next_data.get('punishment') not in [None, 0]:
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
        try:
            sample = torch.load(pt_file_path, map_location='cpu')  # Ensure compatibility
            # Extract net_reward and input tensor
            pt_net_reward = sample.get('net_reward')
            input_tensor = sample.get('input')  # This is a tensor

            # Convert to serializable types
            pt_net_reward = float(pt_net_reward) if pt_net_reward is not None else None
            input_tensor_list = input_tensor.tolist() if input_tensor is not None else None
        except Exception as e:
            print(f"Error loading .pt file {pt_file_path}: {e}")
            pt_net_reward = None
            input_tensor_list = None
    else:
        pt_net_reward = None
        input_tensor_list = None

    # Prepare response data
    response_data = {
        'current_frame': frame_index,
        'total_frames': total_frames,
        'frames': frames_data,  # List of frames based on image_memory
        'input': current_data.get('input', ''),
        'model_prediction': None,  # Placeholder, updated in frontend
        'player_charge': current_data.get('player_charge', 0),
        'enemy_charge': current_data.get('enemy_charge', 0),
        # Conditionally include 'reward' if not zero
        # 'punishment' if not zero
        # Using dictionary unpacking to include fields only if they meet criteria
        **({'reward': current_data['reward']} if current_data.get('reward') not in [None, 0] else {}),
        **({'punishment': current_data['punishment']} if current_data.get('punishment') not in [None, 0] else {}),
        'next_reward': {
            'value': next_reward,
            'frames_ahead': frames_ahead_reward
        } if next_reward is not None else None,
        'next_punishment': {
            'value': next_punishment,
            'frames_ahead': frames_ahead_punishment
        } if next_punishment is not None else None,
        'winner': is_winner,  # Include the winner status from winner.json
        'player_health': current_data.get('player_health', 0),
        'enemy_health': current_data.get('enemy_health', 0),
        'player_position': current_data.get('player_position', None),
        'enemy_position': current_data.get('enemy_position', None),
        'inside_window': current_data.get('inside_window', False),
        'pt_net_reward': pt_net_reward,  # Add net_reward from .pt file
        'pt_input_tensor': input_tensor_list,  # Add input tensor from .pt file
        'future_net_reward': pt_net_reward  # Alias for clarity
    }

    return jsonify(response_data)

@app.route('/folder/<folder_name>/frame/<int:frame_index>/inference')
def frame_inference(folder_name, frame_index):
    """
    Route to perform inference on a specific frame using the AI model.
    Returns the model's prediction.
    """
    if not USE_MODEL:
        return jsonify({'error': 'Inference is disabled because USE_MODEL is set to False'}), 403

    folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return "Folder not found", 404

    # Get list of JSON files sorted by timestamp in filenames
    json_files = sorted(glob.glob(os.path.join(folder_path, '*.json')))
    json_files = [f for f in json_files if os.path.basename(f) != 'winner.json']
    total_frames = len(json_files)
    if frame_index < 0 or frame_index >= total_frames:
        return "Frame index out of range", 404

    # Calculate the start index based on image_memory
    frame_indices = get_exponential_sample(list(range(total_frames)), frame_index, IMAGE_MEMORY)
    while len(frame_indices) < IMAGE_MEMORY:
        frame_indices.insert(0, 0)

    # Load frame data for the sequence
    frames_data = []
    for idx in frame_indices:
        frame_file = json_files[idx]
        with open(frame_file, 'r') as f:
            frame_data = json.load(f)
            # Modify image_path to be relative to folder_name
            image_filename = os.path.basename(frame_data.get('image_path', ''))
            frames_data.append({
                'image_path': image_filename,  # Only the filename
                'input': frame_data.get('input', ''),
                'reward': frame_data.get('reward', None),
                'punishment': frame_data.get('punishment', None),
            })

    # Load and preprocess images
    image_paths = [os.path.join(folder_path, frame['image_path']) for frame in frames_data]

    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Image file {path} not found.")
            return jsonify({'error': f"Image file {path} not found"}), 404
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return jsonify({'error': f"Error loading image {path}"}), 500

    # Run inference using the sequence of images
    predicted_input_str = None
    if USE_MODEL and model is not None:
        try:
            from torchvision import transforms  # Ensure torchvision is imported
            # Preprocess and stack frames
            transform = transforms.Compose([
                transforms.Resize((160, 240)),
                transforms.ToTensor()
            ])
            preprocessed_frames = [transform(img) for img in images]
            stacked_frames = torch.stack(preprocessed_frames, dim=1).unsqueeze(0).to(device)  # Shape: (1, 3, depth, H, W)

            with torch.no_grad():
                outputs = model(stacked_frames)
                probs = torch.sigmoid(outputs)
                probs = probs.cpu().numpy()[0]
                predicted_inputs = (probs >= get_threshold()).astype(int)  # Adjust threshold if needed
                predicted_input_str = ''.join(map(str, predicted_inputs))
        except Exception as e:
            print(f"Error during inference: {e}")
            return jsonify({'error': 'Inference failed'}), 500

    # Prepare response data
    response_data = {
        'model_prediction': predicted_input_str,
    }

    return jsonify(response_data)

@app.route('/training_data/<path:filename>')
def training_data_files(filename):
    """
    Route to serve static files (images) from the training_data directory.
    """
    return send_from_directory(TRAINING_DATA_DIR, filename)

@app.route('/folder/<folder_name>/swap_player', methods=['POST'])
def swap_player(folder_name):
    """
    Route to swap player data (e.g., swap rewards and punishments).
    """
    print(f"Swapping player for folder {folder_name}")
    folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404

    # Process all data files in the folder
    # Swap 'reward' and 'punishment' in each json file
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    json_files = [f for f in json_files if os.path.basename(f) != 'winner.json']

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Swap 'reward' and 'punishment'
            reward = data.get('reward')
            punishment = data.get('punishment')
            data['reward'] = punishment
            data['punishment'] = reward

            #swap healths and positions too
            player_health = data.get('player_health')
            enemy_health = data.get('enemy_health')
            player_position = data.get('player_position')
            enemy_position = data.get('enemy_position')
            data['player_health'] = enemy_health
            data['enemy_health'] = player_health
            data['player_position'] = enemy_position
            data['enemy_position'] = player_position


            # Save back to file
            with open(json_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            return jsonify({'error': f"Error processing {json_file}: {e}"}), 500

    # Switch 'winner' status in 'winner.json' if it exists
    winner_file = os.path.join(folder_path, 'winner.json')
    if os.path.exists(winner_file):
        try:
            with open(winner_file, 'r') as f:
                winner_data = json.load(f)

            is_winner = winner_data.get('is_winner')
            if is_winner is not None:
                # Switch winner status
                winner_data['is_winner'] = not is_winner

                # Save back to file
                with open(winner_file, 'w') as f:
                    json.dump(winner_data, f)
        except Exception as e:
            print(f"Error processing {winner_file}: {e}")
            return jsonify({'error': f"Error processing {winner_file}: {e}"}), 500

    return jsonify({'message': 'Player swapped successfully'}), 200

@app.route('/routes')
def list_routes():
    """
    Optional route to list all registered routes for debugging purposes.
    """
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        line = urllib.parse.unquote(f"{rule.endpoint}: {rule.rule} [{methods}]")
        output.append(line)
    return "<br>".join(output)

if __name__ == '__main__':
    """
    Entry point of the Flask application.
    Loads the AI model if enabled and starts the server.
    """
    # Load the model only if USE_MODEL is True
    if USE_MODEL:
        checkpoint_path_full = get_checkpoint_path(checkpoint_path, IMAGE_MEMORY)  # Adjust the path as needed
        if checkpoint_path_full and os.path.exists(checkpoint_path_full):
            load_model(checkpoint_path_full, image_memory=IMAGE_MEMORY)
        else:
            print("No checkpoint found. Model not loaded.")
    else:
        print("Model loading skipped because USE_MODEL is set to False.")
    app.run(debug=True)
