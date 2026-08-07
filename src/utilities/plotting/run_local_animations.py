import json
import numpy as np
import os
import sys
from datetime import datetime

# Allow running this file directly (python src/utilities/plotting/run_local_animations.py)
# without relying on PYTHONPATH being set by direnv.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utilities.plotting.animate_model import animate_model

# Map-style playback options:
# - Center the mesh on the origin (useful for small surface patches saved in body-centric coordinates)
# - Keep the mesh geometry fixed so only the colormap changes frame-to-frame
CENTER_MESH_AT_ORIGIN = True
ROTATE_MESH = False
COLOUR_MAP = 'RdBu_r'  # Diverging colormap for temperature (red=hot, blue=cold)

def get_output_folders(base_dir):
    """Retrieve all folders in the base directory sorted by modification time."""
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    # Sort by modification time, newest first
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    return folders

def get_user_confirmation(default_folder, available_folders):
    """Ask the user if they want to use the default most recent folder or choose another."""
    print(f"The most recent folder is: {default_folder}")
    use_default = input("Do you want to use this folder? (y/n): ").strip().lower()

    if use_default == 'y':
        return default_folder

    else:
        print("\nAvailable folders:")
        for i, folder in enumerate(available_folders):
            print(f"{i}: {folder}")
        
        # Get the user's choice
        while True:
            try:
                choice = int(input("Select a folder by number: "))
                if 0 <= choice < len(available_folders):
                    return available_folders[choice]
                else:
                    print(f"Invalid selection. Please select a number between 0 and {len(available_folders) - 1}.")
            except ValueError:
                print("Please enter a valid number.")

def run_saved_animation(json_file, npz_file):
    """
    Loads saved animation parameters from JSON and NPZ files and runs the animation.
    
    Parameters:
    - json_file (str): Path to the JSON file containing non-NumPy parameters.
    - npz_file (str): Path to the NPZ file containing NumPy arrays.
    """
    # Load data from files
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    npz_data = np.load(npz_file, allow_pickle=True)
    
    # Extract base arguments
    args = [
        json_data['args'][0],  # path_to_shape_model_file
        npz_data['plotted_variable_array'],  # plotted_variable_array
        npz_data['rotation_axis'],  # rotation_axis
        npz_data['sunlight_direction'],  # sunlight_direction
    ]
    
    # Extract required positional arguments
    kwargs = json_data['kwargs']
    # Positional scalar parameters including emissivity
    args.extend([
        kwargs.pop('timesteps_per_day'),
        kwargs.pop('solar_distance_au'),
        kwargs.pop('rotation_period_hr'),
        kwargs.pop('emissivity', 0.5)  # Default emissivity for old files
    ])
    
    # Extract remaining required positional arguments from kwargs
    plot_title = kwargs.pop('plot_title', 'Temperature distribution')
    axis_label = kwargs.pop('axis_label', 'Temperature (K)')
    animation_frames = kwargs.pop('animation_frames', 200)
    save_animation = kwargs.pop('save_animation', False)
    save_animation_name = kwargs.pop('save_animation_name', 'temperature_animation.gif')
    background_colour = kwargs.pop('background_colour', 'black')
    
    args.extend([plot_title, axis_label, animation_frames, save_animation, save_animation_name, background_colour])

    # Remove placeholder entries from kwargs
    kwargs.pop('rotation_axis', None)
    kwargs.pop('sunlight_direction', None)
    
    # Print debugging information
    print("\nLoaded parameters:")
    print(f"Positional args count: {len(args)}")
    print(f"Keyword args: {list(kwargs.keys())}\n")
    
    # Call the animate_model function; pass output_dir so dome data loads from same folder
    output_dir = os.path.dirname(json_file)
    kwargs['output_dir'] = output_dir

    # Keep the model fixed and recentered (map view)
    kwargs.setdefault('center_mesh_at_origin', CENTER_MESH_AT_ORIGIN)
    kwargs.setdefault('rotate_mesh', ROTATE_MESH)
    kwargs['colour_map'] = COLOUR_MAP

    # Remove placeholder entries from kwargs
    kwargs.pop('rotation_axis', None)
    kwargs.pop('sunlight_direction', None)
    kwargs.pop('shape_model', None)

    print("Running animation...")
    animate_model(*args, animation_debug_mode=True, **kwargs)
    print("Animation complete.")

# Function to plot histogram of temperature distribution (for debugging)
def plot_temperature_histogram(plotted_variable_array):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(plotted_variable_array.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Plotted Variable (Temperature)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main():
    # Determine base directory for saved animations relative to project root
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    base_dir = os.path.join(project_root, 'data', 'output', 'remote_outputs')

    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Exiting.")
        return
    
    # Get all available folders
    available_folders = get_output_folders(base_dir)
    
    if not available_folders:
        print(f"No folders found in {base_dir}. Exiting.")
        return
    
    # The most recent folder
    default_folder = available_folders[0]

    # Ask the user for confirmation or selection
    selected_folder = get_user_confirmation(default_folder, available_folders)

    # Set the file paths
    json_file = os.path.join(base_dir, selected_folder, 'animation_params.json')
    npz_file = os.path.join(base_dir, selected_folder, 'animation_params.npz')

    # Check if both files exist
    if not os.path.exists(json_file) or not os.path.exists(npz_file):
        print(f"Required files not found in {selected_folder}. Exiting.")
        return
    
    # Optional: Plot histogram of temperature distribution for debugging
    npz_data = np.load(npz_file, allow_pickle=True)
    plotted_variable_array = npz_data['plotted_variable_array']
    if plotted_variable_array is not None:
        plot_temperature_histogram(plotted_variable_array)
    else:
        print("WARNING: plotted_variable_array is None (solver detected invalid temperatures)")
    
    # Run the saved animation
    run_saved_animation(json_file, npz_file)

if __name__ == "__main__":
    main()
