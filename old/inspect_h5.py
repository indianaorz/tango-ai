import h5py
import os

def inspect_hdf5_file(file_path):
    print(f"\nInspecting HDF5 File: {file_path}")
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return
    try:
        with h5py.File(file_path, 'r') as hf:
            print("Datasets in the file:")
            # for key in hf.keys():
            #     print(f" - {key}")
            # Specifically check for 'health'
            if 'health' in hf:
                print("'health' dataset is present.")
                print(f"'health' shape: {hf['health'].shape}")
                print(f"'health' dtype: {hf['health'].dtype}")
            else:
                print("'health' dataset is missing.")
            #print the reward value
            if 'reward' in hf:
                print(f"'reward' value: {hf['reward'][()]}")
            #print target list and cross_target
            if 'target_list' in hf:
                print(f"'target_list' value: {hf['target_list'][()]}")
            if 'cross_target' in hf:
                print(f"'cross_target' value: {hf['cross_target'][()]}")
    except Exception as e:
        print(f"Error opening {file_path}: {e}")

# Replace these paths with your actual HDF5 file paths
folder = '../TANGO/data/planning_data/'

#populate files from the folder
hdf5_files = []
for file in os.listdir(folder):
    if file.endswith(".h5"):
        hdf5_files.append(os.path.join(folder, file))

for file_path in hdf5_files:
    inspect_hdf5_file(file_path)
