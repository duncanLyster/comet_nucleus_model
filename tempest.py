'''
This model simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
generalised for use with asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation can be found at: https://github.com/duncanLyster/TEMPEST

Started: 15 Feb 2024

Author: Duncan Lyster
'''

import os
import sys
import math
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from numba.typed import List
from stl import mesh as stl_mesh_module
from scipy.interpolate import interp1d
import h5py
from joblib import Parallel, delayed

# Ensure src directory is in the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

# Imports from src modules
from src.model.calculate_phase_curve import calculate_phase_curve
from src.model.insolation import calculate_insolation
from src.model.simulation import Simulation, ThermalData
from src.model.facet import Facet
from src.model.solvers import TemperatureSolverFactory
from src.model.view_factors import (
    calculate_and_cache_visible_facets,
    calculate_all_view_factors,
    calculate_thermal_view_factors,
    calculate_view_factors
)
from src.utilities.locations import Locations
from src.utilities.config import Config
from src.utilities.utils import (
    calculate_black_body_temp,
    conditional_print,
    calculate_rotation_matrix,
    conditional_tqdm,
    rays_triangles_intersection
)
from src.utilities.plotting.plotting import check_remote_and_animate, plot_static_variable


def read_shape_model(filename, timesteps_per_day, n_layers, max_days, calculate_energy_terms, config=None):
    ''' 
    This function reads in the shape model of the body from a .stl file and return an array of facets, each with its own area, position, and normal vector.

    Ensure that the .stl file is saved in ASCII format, and that the file is in the same directory as this script. Additionally, ensure that the model dimensions are in meters and that the normal vectors are pointing outwards from the body. An easy way to convert the file is to open it in Blender and export it as an ASCII .stl file.

    This function will give an error if the file is not in the correct format, or if the file is not found.
    '''
    
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    
    try:
        # Try reading as ASCII first
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        shape_model = []
        for i in range(len(lines)):
            if lines[i].strip().startswith('facet normal'):
                normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
                vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
                vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
                vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
                facet = Facet(normal, [vertex1, vertex2, vertex3], timesteps_per_day, max_days, n_layers, calculate_energy_terms)
                shape_model.append(facet)
    except UnicodeDecodeError:
        # If ASCII reading fails, try binary
        shape_mesh = stl_mesh_module.Mesh.from_file(filename)
        shape_model = []
        for i in range(len(shape_mesh.vectors)):
            normal = shape_mesh.normals[i]
            vertices = shape_mesh.vectors[i]
            facet = Facet(normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms)
            shape_model.append(facet)

    # Apply shape model rotations if specified
    if config is not None:
        shape_model = apply_shape_model_rotation(shape_model, config)

    return shape_model

def apply_shape_model_rotation(shape_model, config):
    """
    Apply shape model rotations to align the STL file coordinates.
    
    This rotates the shape model itself (separate from orbital dynamics).
    Useful for correcting alignment issues or testing different latitudes.
    
    Rotations are applied around the shape model's center of mass to avoid orbital motion.
    Rotations are applied in order: X rotation first, then Z rotation.
    """
    rotation_x = getattr(config, 'shape_model_rotation_x_degrees', 0)
    rotation_z = getattr(config, 'shape_model_rotation_z_degrees', 0)
    
    if rotation_x == 0 and rotation_z == 0:
        return shape_model  # No rotation needed
    
    # Calculate the center of mass of the shape model
    all_vertices = np.array([v for facet in shape_model for v in facet.vertices])
    center_of_mass = np.mean(all_vertices, axis=0)
    
    # Build rotation matrix: first X rotation, then Z rotation
    rotation_matrix = np.eye(3)
    
    if rotation_x != 0:
        rot_x = calculate_rotation_matrix(np.array([1, 0, 0]), np.radians(rotation_x))
        rotation_matrix = np.dot(rot_x, rotation_matrix)
    
    if rotation_z != 0:
        rot_z = calculate_rotation_matrix(np.array([0, 0, 1]), np.radians(rotation_z))
        rotation_matrix = np.dot(rot_z, rotation_matrix)
    
    # Apply rotation to all facets around the center of mass
    for facet in shape_model:
        # Translate vertices to origin, rotate, then translate back
        facet.vertices = np.array([
            np.dot(rotation_matrix, v - center_of_mass) + center_of_mass 
            for v in facet.vertices
        ])
        # Rotate normal
        facet.normal = np.dot(rotation_matrix, facet.normal)
        facet.normal = facet.normal / np.linalg.norm(facet.normal)  # Renormalize
    
    return shape_model

def save_shape_model(shape_model, filename, config):
    """
    Save the shape model to an ASCII STL file.
    """
    with open(filename, 'w') as f:
        f.write("solid model\n")
        for facet in shape_model:
            f.write(f"facet normal {' '.join(map(str, facet.normal))}\n")
            f.write("  outer loop\n")
            for vertex in facet.vertices:
                f.write(f"    vertex {' '.join(map(str, vertex))}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid model\n")
    
    conditional_print(config.silent_mode, f"Shape model saved to {filename}")

def export_results(shape_model_name, config, temperature_array):
    ''' 
    This function exports the final results of the model to be used in an instrument simulator. It creates a folder within /output with the shape model, model parameters, a plot of the temperature distribution, and final timestep temperatures.
    '''

    folder_name = f"{shape_model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}" # Create new folder name
    os.makedirs(f"output/{folder_name}") # Create folder for results
    shape_mesh = stl_mesh_module.Mesh.from_file(config.path_to_shape_model_file) # Load shape model
    os.system(f"cp {config.path_to_shape_model_file} output/{folder_name}") # Copy shape model .stl file to folder
    os.system(f"cp {config.path_to_setup_file} output/{folder_name}") # Copy model parameters .json file to folder
    np.savetxt(f"output/{folder_name}/temperatures.csv", temperature_array, delimiter=',') # Save the final timestep temperatures to .csv file

    # Plot the temperature distribution for the final timestep and save it to the folder
    temp_output_file_path = f"output/{folder_name}/"

def parse_args():
    parser = argparse.ArgumentParser(description='TEMPEST: Thermal Model for Planetary Bodies')
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file'
    )
    return parser.parse_args()

def check_environment(config):
    """Check if the execution environment matches config settings"""
    from multiprocessing import cpu_count
    
    available_cores = cpu_count()
    requested_cores = config.config_data.get('n_jobs', 4)  # Get original requested cores
    
    if config.remote and available_cores < 8:
        print(f"\nWARNING: Remote execution requested but only {available_cores} CPU cores available")
        print(f"This might significantly impact performance")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit("Execution cancelled by user")
    
    # Always warn about core count mismatch
    if requested_cores > available_cores:
        print(f"\nWARNING: Requested {requested_cores} cores but only {available_cores} available")
        print(f"Setting n_jobs to {available_cores}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    full_run_start_time = time.time()

    # Load user configuration with specified config path
    config = Config(config_path=args.config)
    
    # Check environment
    check_environment(config)

    # Load setup parameters from JSON file
    simulation = Simulation(config)

    # ============================================================================
    # NOTE: Adaptive timestep calculation is used by default
    # If you need to override timesteps_per_day for testing, uncomment below:
    # ============================================================================
    # original_timesteps = simulation.timesteps_per_day
    # simulation.timesteps_per_day = 5000  # Override timesteps (WARNING: Can be very slow!)
    # simulation.delta_t = simulation.rotation_period_s / simulation.timesteps_per_day
    # conditional_print(config.silent_mode, f"WARNING: Overriding timesteps_per_day from {original_timesteps} to {simulation.timesteps_per_day}")
    # ============================================================================

    # Setup simulation
    try:
        shape_model = read_shape_model(
            config.path_to_shape_model_file,
            simulation.timesteps_per_day,
            simulation.n_layers,
            simulation.max_days,
            config.calculate_energy_terms,
            config
        )
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        sys.exit(1)
    thermal_data = ThermalData(len(shape_model), simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)

    # Compute dimensionless thermal parameter (Theta) for comparison with LUT
    _sigma = 5.670374419e-8
    _solar_flux = getattr(simulation, 'solar_luminosity', 3.828e26) / (4 * np.pi * simulation.solar_distance_m**2)
    _tss = ((1 - simulation.albedo) * _solar_flux / (simulation.emissivity * _sigma)) ** 0.25
    _theta = (simulation.thermal_inertia * np.sqrt(simulation.angular_velocity)) / (simulation.emissivity * _sigma * _tss**3)

    conditional_print(config.silent_mode,  f"\nDerived model parameters:")
    conditional_print(config.silent_mode,  f"Number of timesteps per day: {simulation.timesteps_per_day}")
    conditional_print(config.silent_mode,  f"Layer thickness: {simulation.layer_thickness} m")
    conditional_print(config.silent_mode,  f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5")
    conditional_print(config.silent_mode,  f"Thermal parameter (Theta): {_theta:.4f}  [LUT grid: 0.0316]")
    conditional_print(config.silent_mode,  f"Subsolar temperature: {_tss:.1f} K")
    conditional_print(config.silent_mode,  f"Skin depth: {simulation.skin_depth} m")
    conditional_print(config.silent_mode,  f"\n Number of facets: {len(shape_model)}")

    # Setup the model
    positions = np.array([facet.position for facet in shape_model])
    normals = np.array([facet.normal for facet in shape_model])
    vertices = np.array([facet.vertices for facet in shape_model])
    
    # Time the visible facets loading/conversion
    t_vis_start = time.time()
    visible_indices = calculate_and_cache_visible_facets(config.silent_mode, shape_model, positions, normals, vertices, config)
    t_vis_end = time.time()
    conditional_print(config.silent_mode, f"Visible facets load time: {t_vis_end - t_vis_start:.2f} seconds")

    thermal_data.set_visible_facets(visible_indices)

    for i, facet in enumerate(shape_model):
        facet.visible_facets = visible_indices[i]   
        
    if config.include_self_heating or config.n_scatters > 0:
        calculate_view_factors_start = time.time()
        
        # Calculate regular view factors for scattering and self-heating
        all_view_factors = calculate_all_view_factors(shape_model, thermal_data, config, config.vf_rays)
        thermal_data.set_secondary_radiation_view_factors(all_view_factors)
        
        # Calculate thermal view factors if self-heating is enabled
        if config.include_self_heating:
            thermal_view_factors = calculate_thermal_view_factors(
                shape_model,
                thermal_data,
                config
            )
            thermal_data.set_thermal_view_factors(thermal_view_factors)
        
        calculate_view_factors_end = time.time()
        conditional_print(config.silent_mode, f"Time taken to calculate view factors: {calculate_view_factors_end - calculate_view_factors_start:.2f} seconds")
        
        # Convert thermal_view_factors to Numba lists (always, for consistency with solver)
        numba_view_factors = List()
        for view_factors in thermal_data.thermal_view_factors:
            # Ensure array is 1D and handle empty arrays
            arr = np.array(view_factors, dtype=np.float64)
            if arr.size == 0:
                numba_view_factors.append(np.array([], dtype=np.float64))
            else:
                numba_view_factors.append(arr.flatten() if arr.ndim > 1 else arr)
        thermal_data.thermal_view_factors = numba_view_factors

    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)

    if config.plot_insolation_curve and not config.silent_mode:
        fig_insolation = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode,  f"Preparing insolation curve plot.\n")
        
        if config.plotted_facet_index >= len(shape_model):
            conditional_print(config.silent_mode,  f"Facet index {config.plotted_facet_index} out of range. Please select a facet index between 0 and {len(shape_model) - 1}.")
        else:
            # Get the insolation data for the facet
            insolation_data = thermal_data.insolation[config.plotted_facet_index]
            
            roll_amount = 216
            # Roll the array to center the peak
            centered_insolation = np.roll(insolation_data, roll_amount)
            
            # Create x-axis in degrees
            degrees = np.linspace(0, 360, len(insolation_data), endpoint=False)
            
            # Create DataFrame for easy CSV export
            df = pd.DataFrame({
                'Rotation (degrees)': degrees,
                'Insolation (W/m^2)': centered_insolation
            })

            # Export to CSV
            output_dir = 'insolation_data'
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            output_csv_path = os.path.join(output_dir, f'facet_{config.plotted_facet_index}.csv')
            df.to_csv(output_csv_path, index=False)
            conditional_print(config.silent_mode,  f"Insolation data exported to {output_csv_path}")
            
            # Plot the centered insolation curve
            plt.plot(degrees, centered_insolation)
            plt.xlabel('Rotation of the body (degrees)')
            plt.ylabel('Insolation (W/m^2)')
            plt.title(f'Insolation curve for facet {config.plotted_facet_index} over one full rotation of the body')
            plt.xlim(0, 360)
            plt.xticks(np.arange(0, 361, 45))  # Set x-axis ticks every 45 degrees
            
            conditional_print(config.silent_mode,  f"Facet {config.plotted_facet_index} insolation curve plotted with peak centered.")
            plt.show()

            sys.exit()

    conditional_print(config.silent_mode, f"Insolation calculation complete. Preparing for thermal simulation...\n")

    if config.animate_shadowing:
        conditional_print(config.silent_mode,  f"Preparing shadowing visualisation.\n")

        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                          thermal_data.insolation, 
                          simulation.rotation_axis, 
                          simulation.sunlight_direction, 
                          simulation.timesteps_per_day, 
                          simulation.solar_distance_au,
                          simulation.rotation_period_hours,
                          config.emissivity,
                          colour_map='binary_r', 
                          plot_title='Shadowing on the body', 
                          axis_label='Insolation (W/m^2)', 
                          animation_frames=200, 
                          save_animation=False, 
                          save_animation_name='shadowing_animation.gif', 
                          background_colour = 'black',
                          shape_model=shape_model)

    conditional_print(config.silent_mode,  f"Calculating initial temperatures.\n")

    initial_temperatures_start = time.time()
    solver = TemperatureSolverFactory.create(config.temp_solver)
    thermal_data = solver.initialize_temperatures(thermal_data, simulation, config)
    initial_temperatures_end = time.time()

    conditional_print(config.silent_mode,  f"Time taken to calculate initial temperatures: {initial_temperatures_end - initial_temperatures_start:.2f} seconds")

    if config.plot_initial_temp_histogram and not config.silent_mode:
        fig_histogram = plt.figure()
        initial_temperatures = [thermal_data.temperatures[i, 0, 0] for i in range(len(shape_model))]
        plt.hist(initial_temperatures, bins=20)
        plt.xlabel('Initial temperature (K)')
        plt.ylabel('Number of facets')
        plt.title('Initial temperature distribution of all facets')
        fig_histogram.show()

    numba_visible_facets = List()
    conditional_print(config.silent_mode, f"Converting visible facets to numba format...")
    for facets in thermal_data.visible_facets:
        numba_visible_facets.append(np.array(facets, dtype=np.int64))
    thermal_data.visible_facets = numba_visible_facets
    conditional_print(config.silent_mode, f"Visible facets converted.\n")

    if config.animate_secondary_radiation_view_factors:
        selected_facet = 1454  # Change this to the index of the facet you're interested in
        
        # Get the indices and view factors of contributing facets
        contributing_indices = thermal_data.visible_facets[selected_facet]
        contributing_view_factors = thermal_data.secondary_radiation_view_factors[selected_facet]
        
        # Create an array of zeros for all facets
        contribution_data = np.zeros(len(shape_model))
        
        # Set the view factors for the contributing facets
        contribution_data[contributing_indices] = 1
        contribution_data[selected_facet] = 0.5

        # Print contributing facets and their view factors
        conditional_print(config.silent_mode,  f"\nContributing facets for facet {selected_facet}:")
        for index, view_factors in zip(contributing_indices, contributing_view_factors):
            conditional_print(config.silent_mode,  f"Facet {index}: view factor = {view_factors:.6f}")
        conditional_print(config.silent_mode,  f"Total number of contributing facets: {len(contributing_indices)}")
        
        conditional_print(config.silent_mode,  f"Preparing visualization of contributing facets for facet {selected_facet}.")
        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                      contribution_data[:, np.newaxis], 
                    simulation.rotation_axis, 
                    simulation.sunlight_direction, 
                    1,               
                    simulation.solar_distance_au,              
                    simulation.rotation_period_hours,
                    config.emissivity,
                    colour_map='viridis', 
                    plot_title=f'Contributing Facets for Facet {selected_facet}', 
                    axis_label='View Factors Value', 
                    animation_frames=1, 
                    save_animation=False, 
                    save_animation_name=f'contributing_facets_{selected_facet}.png', 
                    background_colour='black')
        
    if config.animate_secondary_contributions:
        # Calculate the sum of secondary radiation view factors for each facet
        secondary_radiation_sum = np.array([np.sum(view_factors) for view_factors in thermal_data.secondary_radiation_view_factors])

        conditional_print(config.silent_mode,  "Preparing secondary radiation visualization.")
        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                    secondary_radiation_sum[:, np.newaxis], 
                    simulation.rotation_axis, 
                    simulation.sunlight_direction, 
                    1,               
                    simulation.solar_distance_au,              
                    simulation.rotation_period_hours,
                    config.emissivity,
                    colour_map='viridis', 
                    plot_title='Secondary Radiation Contribution', 
                    axis_label='Sum of View Factors', 
                    animation_frames=1, 
                    save_animation=False, 
                    save_animation_name='secondary_radiation.png', 
                    background_colour='black')

    conditional_print(config.silent_mode,  f"Running main simulation loop.\n")
    conditional_print(config.silent_mode,  f"Convergence target: {simulation.convergence_target} K with {config.convergence_method} convergence method.\n")

    # OPTIMIZATION: Clean up shape_model object before solver (not needed anymore, frees ~1-2 GB)
    # Solver only needs thermal_data (insolation, view factors, visible facets), not the facet objects
    conditional_print(config.silent_mode, "Freeing shape model from memory for solver phase...")
    shape_model = None
    gc.collect()
    conditional_print(config.silent_mode, "Memory freed.\n")

    # Run solver
    solver_start_time = time.time()
    result = solver.solve(thermal_data, shape_model, simulation, config)
    solver_end_time = time.time()
    solver_execution_time = solver_end_time - solver_start_time
    full_run_end_time = time.time()

    if result["final_timestep_temperatures"] is not None:
        conditional_print(config.silent_mode, f"Convergence target achieved after {result['days_to_convergence']} days.")
        conditional_print(config.silent_mode, f"Final temperature error: {result['mean_temperature_error']} K")
        conditional_print(config.silent_mode, f"Max temperature error: {result['max_temperature_error']} K")
    else:
        conditional_print(config.silent_mode, f"Model did not converge after {result['days_to_convergence']} days.")
        conditional_print(config.silent_mode, f"Final temperature error: {result['mean_temperature_error']} K")

    conditional_print(config.silent_mode, f"Solver execution time: {solver_execution_time} seconds")
    conditional_print(config.silent_mode, f"Full run time: {full_run_end_time - full_run_start_time} seconds")

    if config.plot_insolation_curve and not config.silent_mode:
        fig_temperature = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode, f"Preparing temperature curve plot.\n")
        
        if config.plotted_facet_index >= thermal_data.temperatures.shape[0]:
            conditional_print(config.silent_mode, f"Facet index {config.plotted_facet_index} out of range. Please select a facet index between 0 and {thermal_data.temperatures.shape[0] - 1}.")
        else:
            # Get the temp data for the facet
            temperature_data = result["final_day_temperatures"][config.plotted_facet_index]
            
            # Calculate black body temperatures
            insolation_data = thermal_data.insolation[config.plotted_facet_index]
            black_body_temps = np.array([calculate_black_body_temp(ins, simulation.emissivity, simulation.albedo) for ins in insolation_data])
            
            # Find the index of the maximum black body temperature
            max_index = np.argmax(black_body_temps)
            
            # Calculate the roll amount to center the peak
            roll_amount = len(black_body_temps) // 2 - max_index
            
            # Roll the arrays to center the peak
            centered_temperature = np.roll(temperature_data, roll_amount)
            centered_black_body = np.roll(black_body_temps, roll_amount)
            
            # Create x-axis in degrees
            degrees = np.linspace(0, 360, len(temperature_data), endpoint=False)
            
            # Create DataFrame for easy CSV export
            df = pd.DataFrame({
                'Rotation (degrees)': degrees,
                'Temperature (K)': centered_temperature,
                'Black Body Temperature (K)': centered_black_body
            })

            # Export to CSV
            output_dir = 'temperature_data'
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            output_csv_path = os.path.join(output_dir, f'facet_{config.plotted_facet_index}_with_black_body.csv')
            df.to_csv(output_csv_path, index=False)
            conditional_print(config.silent_mode, f"Temperature data exported to {output_csv_path}")
            
            # Plot the centered temperature curves
            plt.plot(degrees, centered_temperature, label='Model Temperature')
            plt.plot(degrees, centered_black_body, label='Black Body Temperature', linestyle='--')
            plt.xlabel('Rotation of the body (degrees)')
            plt.ylabel('Temperature (K)')
            plt.title(f'Temperature curves for facet {config.plotted_facet_index} over one full rotation of the body')
            plt.xlim(0, 360)
            plt.xticks(np.arange(0, 361, 90))  # Set x-axis ticks every 90 degrees
            plt.legend()
            
            conditional_print(config.silent_mode, f"Facet {config.plotted_facet_index} temperature curves plotted with peak centered.")
            plt.show()

            sys.exit()

    if config.plot_temp_curve and len(config.plot_temp_curve) > 0 and not config.silent_mode:
        fig_temp_curve = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode, f"Preparing temperature curve plot for {len(config.plot_temp_curve)} facet(s).\n")
        
        # Validate all facet indices
        invalid_indices = [idx for idx in config.plot_temp_curve if idx < 0 or idx >= thermal_data.temperatures.shape[0]]
        if invalid_indices:
            conditional_print(config.silent_mode, 
                f"ERROR: Invalid facet indices {invalid_indices}. "
                f"Valid facet indices are between 0 and {thermal_data.temperatures.shape[0] - 1}.")
            sys.exit(1)
        
        # Create x-axis in degrees
        degrees = np.linspace(0, 360, simulation.timesteps_per_day, endpoint=False)
        
        # Plot temperature curves for each facet
        for facet_idx in config.plot_temp_curve:
            temperature_data = result["final_day_temperatures"][facet_idx]
            plt.plot(degrees, temperature_data, label=f'Facet {facet_idx}')
        
        plt.xlabel('Rotation of the body (degrees)')
        plt.ylabel('Temperature (K)')
        if len(config.plot_temp_curve) == 1:
            plt.title(f'Diurnal temperature curve for facet {config.plot_temp_curve[0]}')
        else:
            plt.title(f'Diurnal temperature curves for facets {config.plot_temp_curve}')
        plt.xlim(0, 360)
        plt.xticks(np.arange(0, 361, 45))
        plt.grid(True, alpha=0.3)
        if len(config.plot_temp_curve) > 1:
            plt.legend()
        
        # Save data if requested
        if config.save_temp_curve_data:
            output_dir = 'temperature_curves'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Create filename with parameters for identification
            param_str = f"TI{simulation.thermal_inertia}_alb{simulation.albedo}_heat{config.subsurface_heating_flux}"
            
            for facet_idx in config.plot_temp_curve:
                temperature_data = result["final_day_temperatures"][facet_idx]
                
                # Create DataFrame with rotation angle and temperature
                df = pd.DataFrame({
                    'Rotation (degrees)': degrees,
                    'Temperature (K)': temperature_data
                })
                
                # Add metadata as comment in CSV header
                output_csv_path = os.path.join(output_dir, f'facet_{facet_idx}_{param_str}_{timestamp}.csv')
                df.to_csv(output_csv_path, index=False)
                
                conditional_print(config.silent_mode, 
                    f"Temperature curve data saved to {output_csv_path}")
        
        conditional_print(config.silent_mode, f"Temperature curve(s) plotted for facet(s) {config.plot_temp_curve}.")
        plt.show()

    if config.plot_final_day_all_layers_temp_distribution and not config.silent_mode:
        fig_final_all_layers_temp_dist = plt.figure()
        plt.plot(result["final_day_temperatures_all_layers"][config.plotted_facet_index])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all layers in facet')
        fig_final_all_layers_temp_dist.show()

    if config.plot_final_day_all_layers_temp_distribution and not config.silent_mode:
        fig_all_days_all_layers_temp_dist = plt.figure()
        plt.plot(thermal_data.temperatures[config.plotted_facet_index, :, :])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature distribution for all layers in facet for the full run')
        fig_all_days_all_layers_temp_dist.show()

    if config.plot_energy_terms and not config.silent_mode:
        fig_energy_terms = plt.figure()
        plt.plot(shape_model[config.plotted_facet_index].unphysical_energy_loss[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Unphysical energy loss')
        plt.plot(shape_model[config.plotted_facet_index].insolation_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Insolation energy')
        plt.plot(shape_model[config.plotted_facet_index].re_emitted_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Re-emitted energy')
        plt.plot(-shape_model[config.plotted_facet_index].surface_energy_change[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Surface energy change')
        plt.plot(shape_model[config.plotted_facet_index].conducted_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Conducted energy')
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Energy (J)')
        plt.title('Energy terms for facet for the final day')
        fig_energy_terms.show()

    if config.animate_final_day_temp_distribution:
        conditional_print(config.silent_mode,  f"Preparing temperature animation.\n")

        plot_array = result["final_day_temperatures"]
        check_remote_and_animate(
            config.remote, config.path_to_shape_model_file, 
            plot_array, 
            simulation.rotation_axis, 
            simulation.sunlight_direction, 
            simulation.timesteps_per_day,
            simulation.solar_distance_au,              
            simulation.rotation_period_hours,              
            config.emissivity,
            apply_kernel_based_roughness=False,
            dome_radius_factor=config.kernel_dome_radius_factor,
            colour_map='coolwarm', 
            plot_title='Temperature distribution', 
            axis_label='Temperature (K)', 
            animation_frames=200, 
            save_animation=False, 
            save_animation_name='temperature_animation.gif', 
            background_colour='black',
            shape_model=shape_model
            )
        conditional_print(config.silent_mode, "Animation window closed, continuing...")

    if config.plot_final_day_peak_temp:
        conditional_print(config.silent_mode, f"Preparing peak temperature plot.\n")
        
        # Calculate peak temperature for each facet
        final_day_temps = result["final_day_temperatures"]
        peak_temperatures = np.max(final_day_temps, axis=1)
        
        plot_static_variable(
            config.remote, config.path_to_shape_model_file,
            peak_temperatures,
            simulation.rotation_axis,
            simulation.sunlight_direction,
            simulation.solar_distance_au,
            simulation.rotation_period_hours,
            config.emissivity,
            apply_kernel_based_roughness=False,
            dome_radius_factor=config.kernel_dome_radius_factor,
            colour_map='coolwarm',
            plot_title='Peak Temperature Distribution',
            axis_label='Temperature (K)',
            background_colour='black')
        conditional_print(config.silent_mode, "Peak temperature plot window closed, continuing...")

    if config.plot_final_day_comparison and not config.silent_mode:
        conditional_print(config.silent_mode,  f"Saving final day temperatures to CSV file.\n")
        np.savetxt("final_day_temperatures.csv", np.column_stack((np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), result["final_day_temperatures"][config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

        thermprojrs_data = np.loadtxt("thermprojrs_data.csv", delimiter=',', skiprows=1)

        fig_model_comparison = plt.figure()
        plt.plot(thermprojrs_data[:, 0], thermprojrs_data[:, 1], label='Thermprojrs')
        plt.plot(np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), result["final_day_temperatures"][config.plotted_facet_index], label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for facet')
        plt.legend()
        fig_model_comparison.show()

        x_original = np.linspace(0, 2 * np.pi, simulation.timesteps_per_day)
        x_new = np.linspace(0, 2 * np.pi, thermprojrs_data.shape[config.plotted_facet_index])

        interp_func = interp1d(x_new, thermprojrs_data[:, 1], kind='linear')
        thermprojrs_interpolated = interp_func(x_original)

        plt.plot(x_original, result["final_day_temperatures"][config.plotted_facet_index] - thermprojrs_interpolated, label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature difference (K)')
        plt.title('Temperature difference between this model and Thermprojrs for facet')
        plt.legend()
        plt.show()

        np.savetxt("final_day.csv", np.column_stack((x_original, result["final_day_temperatures"][config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

    if config.calculate_visible_phase_curve:
        phase_angles, brightness_values = calculate_phase_curve(
            shape_model,
            simulation,
            thermal_data,
            config,
            phase_curve_type='visible',
            observer_distance=1e9,
            normalized=True,
            plot=True
        )

    # Save the visible phase curve data to a CSV file
    if config.save_visible_phase_curve_data:
        locations = Locations()
        locations.ensure_directories_exist()  # Ensure output directories exist
        # Create name using shape model name and time
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(locations.phase_curve_data, f'{filename}_visible_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        df = pd.DataFrame({
            'Phase Angle (degrees)': phase_angles,
            'Brightness Value': brightness_values
        })
        df.to_csv(output_csv_path, index=False)

        # Create and save the phase curve plot
        plt.figure()  # Create a new figure
        plt.plot(phase_angles, brightness_values, label='Brightness vs Phase Angle')  # Plot the data
        plt.xlabel('Phase Angle (degrees)')
        plt.ylabel('Brightness Value')
        plt.title('Visible Phase Curve')
        plt.legend()
        
        # Save the plot
        output_image_path = output_csv_path.replace('.csv', '.png')
        plt.savefig(output_image_path)  # Save the figure as a .png file
        plt.close()  # Close the figure after saving to avoid displaying it in non-interactive environments

        conditional_print(config.silent_mode,  f"Visible phase curve data exported to {output_csv_path}")

    if config.calculate_thermal_phase_curve:
        phase_angles, brightness_values = calculate_phase_curve(
            shape_model,
            simulation,
            thermal_data,
            config,
            phase_curve_type='thermal',
            observer_distance=1e8,
            normalized=False,
            plot=config.show_thermal_phase_curve
        )

    # Save the thermal phase curve data to a CSV file
    if config.save_thermal_phase_curve_data:
        locations = Locations()
        locations.ensure_directories_exist()  # Ensure output directories exist
        # Create name using shape model name and time
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(locations.phase_curve_data, f'{filename}_thermal_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        df = pd.DataFrame({
            'Phase Angle (degrees)': phase_angles,
            'Brightness Value': brightness_values
        })
        df.to_csv(output_csv_path, index=False)
        
        # Create and save the thermal phase curve plot
        plt.figure()  # Create a new figure
        plt.plot(phase_angles, brightness_values, label='Thermal Brightness vs Phase Angle')
        plt.xlabel('Phase Angle (degrees)')
        plt.ylabel('Thermal Brightness Value')
        plt.title('Thermal Phase Curve')
        plt.legend()
        
        # Save the plot
        output_image_path = output_csv_path.replace('.csv', '.png')
        plt.savefig(output_image_path)  # Save the figure as a .png file
        plt.close()  # Close the figure after saving to avoid displaying it in non-interactive environments

        # Notify user
        conditional_print(config.silent_mode, f"Thermal phase curve data exported to {output_csv_path}")

    # ============================================================================
    # SAVE OUTPUTS FOR TEMPEST_RAD (Unconditional)
    # ============================================================================
    conditional_print(config.silent_mode, "Saving temperature arrays for TEMPEST_RAD...")
    
    # Each model run gets its own timestamped folder so runs never overwrite
    # each other.  They all live under output/runs/ to keep output/ itself
    # readable (analysis products, figures and scratch dirs sit alongside).

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = f"output/runs/run_{timestamp}"
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Save temperatures (Facets x Timesteps)
    # thermal_data.temperatures is (N_facets, N_timesteps)
    temp_csv_path = os.path.join(run_output_dir, "temperatures.csv")
    temp_npy_path = os.path.join(run_output_dir, "temperatures.npy")
    
    np.savetxt(temp_csv_path, thermal_data.temperatures, delimiter=',')
    np.save(temp_npy_path, thermal_data.temperatures)
    
    conditional_print(config.silent_mode, f"Saved temperatures to {run_output_dir}")
    # Also copy config there for easy reference by TEMPEST_RAD
    os.system(f"cp {args.config} {os.path.join(run_output_dir, 'config.yaml')}")

    conditional_print(config.silent_mode,  f"Model run complete.\n")

    # Export insolation data for all facets (if enabled)
    if config.save_insolation_data:
        # Note: thermal_data.insolation contains absorbed insolation (insolation * (1-albedo))
        # Convert to actual insolation by dividing by (1-albedo)
        conditional_print(config.silent_mode, "Exporting insolation data to CSV...")
        degrees = np.linspace(0, 360, simulation.timesteps_per_day, endpoint=False)
        os.makedirs('insolation_data', exist_ok=True)
        
        # Convert absorbed insolation to actual insolation
        # thermal_data.insolation = actual_insolation * (1 - albedo)
        # Therefore: actual_insolation = thermal_data.insolation / (1 - albedo)
        one_minus_albedo = 1.0 - simulation.albedo
        if abs(one_minus_albedo) < 1e-10:
            conditional_print(config.silent_mode, "WARNING: Albedo is 1.0, cannot convert to actual insolation. Saving absorbed insolation as-is.")
            actual_insolation = thermal_data.insolation.copy()
        else:
            actual_insolation = thermal_data.insolation / one_minus_albedo
        
        # Create a combined DataFrame with all facets
        insolation_dict = {'rotation_deg': degrees}
        n_facets = thermal_data.temperatures.shape[0]
        
        # Add each facet's insolation as a column
        for idx in range(n_facets):
            insolation_dict[f'facet_{idx}_insolation_Wm2'] = actual_insolation[idx]
        
        # Save combined CSV with all facets
        df_combined = pd.DataFrame(insolation_dict)
        combined_csv_path = 'insolation_data/all_facets_insolation.csv'
        df_combined.to_csv(combined_csv_path, index=False)
        conditional_print(config.silent_mode, f"Combined insolation data saved to {combined_csv_path}")
        
        # Also save individual CSV files for each facet
        for idx in range(n_facets):
            try:
                df_ins = pd.DataFrame({
                    'rotation_deg': degrees, 
                    'insolation_Wm2': actual_insolation[idx]
                })
                df_ins.to_csv(f'insolation_data/facet_{idx}.csv', index=False)
            except Exception as e:
                conditional_print(config.silent_mode, f"  Error exporting insolation for facet {idx}: {e}")
        
        conditional_print(config.silent_mode, f"Insolation data exported for {n_facets} facets")

# Call the main program with interrupt handling
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRun interrupted by user (Ctrl-C). Exiting.")
        sys.exit(1)
