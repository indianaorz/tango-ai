# extract_datasets.py
import os
import h5py
import numpy as np
import glob
import argparse
import json
from utils import get_root_dir  # Ensure utils module is accessible

def get_cache_directories(cache_type):
    """
    Retrieves the planning and/or battle cache directories based on the root directory and cache_type.
    
    :param cache_type: 'planning', 'battle', or 'both'
    :return: List of cache directories to process
    """
    root_dir = get_root_dir()
    training_cache_dir = os.path.join(root_dir, 'training_cache')
    planning_cache_dir = os.path.join(training_cache_dir, 'planning')
    battle_cache_dir = os.path.join(training_cache_dir, 'battle')
    
    directories = []
    if cache_type in ['planning', 'both']:
        directories.append(('Planning', planning_cache_dir))
    if cache_type in ['battle', 'both']:
        directories.append(('Battle', battle_cache_dir))
    return directories

def find_h5_files(cache_dir):
    """
    Finds all .h5 files within the specified cache directory.
    
    :param cache_dir: Path to the cache directory
    :return: List of .h5 file paths
    """
    return sorted(glob.glob(os.path.join(cache_dir, '*.h5')))

def binary_array_to_string(binary_array):
    """
    Converts a numpy array of binary values (0s and 1s) to a binary string.
    
    :param binary_array: Numpy array of 0s and 1s
    :return: String representation, e.g., "0001"
    """
    return ''.join(str(int(bit)) for bit in binary_array)

def dataset_to_serializable(data):
    """
    Converts numpy data types to Python native types for JSON serialization.
    
    :param data: Numpy data
    :return: Serializable data
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def extract_datasets_from_h5(h5_path, datasets):
    """
    Extracts specified datasets from an HDF5 file and converts them to serializable formats.
    
    :param h5_path: Path to the HDF5 file
    :param datasets: List of dataset names to extract
    :return: Dictionary with dataset names as keys and extracted data as values
    """
    extracted_data = {}
    try:
        with h5py.File(h5_path, 'r') as h5f:
            for dataset in datasets:
                if dataset not in h5f:
                    print(f"  - Dataset '{dataset}' not found in {os.path.basename(h5_path)}. Skipping this dataset.")
                    continue
                data = h5f[dataset][:]
                if dataset == 'inputs':
                    # Convert binary arrays to strings
                    data = [binary_array_to_string(arr) for arr in data]
                elif dataset == 'images':
                    # Convert image arrays to lists or handle differently if needed
                    data = data.tolist()
                elif dataset in ['player_grids', 'enemy_grids']:
                    # Ensure grids are serialized properly
                    data = data.tolist()
                elif dataset in ['inside_windows']:
                    # Convert floats to bool if necessary
                    data = [bool(x) for x in data]
                # Convert all data to serializable format
                data = dataset_to_serializable(data)
                extracted_data[dataset] = data
    except Exception as e:
        print(f"  - Error processing {os.path.basename(h5_path)}: {e}")
    return extracted_data

def save_to_json(output_path, data):
    """
    Saves the extracted data to a JSON file.
    
    :param output_path: Path to the output JSON file
    :param data: Data to save
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving to JSON file: {e}")

def save_to_csv(output_path, data, datasets):
    """
    Saves the extracted data to a CSV file. Note that this is simplistic and may not handle nested structures well.
    
    :param output_path: Path to the output CSV file
    :param data: Data to save
    :param datasets: List of dataset names
    """
    import csv
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(datasets)
            # Determine the number of rows based on the first dataset
            num_rows = len(data[datasets[0]]) if datasets else 0
            for i in range(num_rows):
                row = []
                for dataset in datasets:
                    value = data.get(dataset, [])
                    if isinstance(value, list) and i < len(value):
                        row.append(value[i])
                    else:
                        row.append('')
                writer.writerow(row)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving to CSV file: {e}")

def parse_arguments():
    """
    Parses command-line arguments.
    
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract specified datasets from HDF5 cache files.")
    parser.add_argument(
        '-d', '--datasets',
        nargs='+',
        default=['inputs'],
        help="List of datasets to extract (e.g., inputs images player_healths). Defaults to ['inputs']."
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default=None,
        help="Path to save the extracted data as a JSON or CSV file. If not provided, data will be printed to the console."
    )
    parser.add_argument(
        '-c', '--cache-type',
        type=str,
        choices=['planning', 'battle', 'both'],
        default='both',
        help="Specify which cache directory to process: 'planning', 'battle', or 'both'. Defaults to 'both'."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    datasets = args.datasets
    output_file = args.output_file
    cache_type = args.cache_type
    
    # Validate datasets
    available_datasets = {'images', 'inputs', 'player_healths', 'enemy_healths', 
                          'player_grids', 'enemy_grids', 'inside_windows', 'net_rewards'}
    invalid_datasets = set(datasets) - available_datasets
    if invalid_datasets:
        print(f"Error: Invalid dataset names specified: {', '.join(invalid_datasets)}")
        print(f"Available datasets are: {', '.join(available_datasets)}")
        return
    
    cache_directories = get_cache_directories(cache_type)
    
    # Check if cache directories exist
    for cache_label, cache_dir in cache_directories:
        if not os.path.exists(cache_dir):
            print(f"Warning: {cache_label} cache directory does not exist: {cache_dir}")
    
    # Find all .h5 files in specified cache directories
    all_h5_files = []
    for cache_label, cache_dir in cache_directories:
        if os.path.exists(cache_dir):
            h5_files = find_h5_files(cache_dir)
            if h5_files:
                all_h5_files.extend([(cache_label, f) for f in h5_files])
            else:
                print(f"No HDF5 (.h5) files found in {cache_label} cache directory: {cache_dir}")
    
    if not all_h5_files:
        print("No HDF5 (.h5) files found in the specified cache directories.")
        return
    
    extracted_results = {}
    
    for cache_label, h5_file in all_h5_files:
        print(f"\n=== {cache_label} Cache File: {os.path.basename(h5_file)} ===")
        extracted_data = extract_datasets_from_h5(h5_file, datasets)
        if not extracted_data:
            print("No data extracted from this file.")
            continue
        extracted_results[os.path.basename(h5_file)] = extracted_data
        # Print to console if no output file is specified
        if not output_file:
            for dataset, data in extracted_data.items():
                print(f"\n-- Dataset: {dataset} --")
                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        print(f"  {dataset.capitalize()} {idx}: {item}")
                else:
                    print(f"  {dataset.capitalize()}: {data}")
    
    # Save to output file if specified
    if output_file:
        file_ext = os.path.splitext(output_file)[1].lower()
        if file_ext == '.json':
            save_to_json(output_file, extracted_results)
        elif file_ext == '.csv':
            # For CSV, flatten the structure
            # This simplistic approach assumes all datasets have the same length per file
            # For more complex structures, consider using a different format or multiple CSVs
            # Here, we'll create separate CSVs per HDF5 file
            for h5_filename, data in extracted_results.items():
                csv_filename = f"{os.path.splitext(h5_filename)[0]}_extracted.csv"
                csv_path = os.path.join(os.path.dirname(output_file), csv_filename)
                save_to_csv(csv_path, data, datasets)
        else:
            print("Unsupported output file format. Please use .json or .csv extensions.")

if __name__ == '__main__':
    main()
