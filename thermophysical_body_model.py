''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
generalised for use with asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation to be found (one day) at: https://github.com/duncanLyster/comet_nucleus_model

OPEN QUESTIONS: 
Do we consider partial shadow? 
Do we treat facets as points or full 2D polygons?

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary local heat sources e.g. jets

IMPORTANT CONSIDERATIONS: 
Generalising the model so it can be used e.g for asteroids, Enceladus fractures, adjacent emitting bodies (e.g. binaries, Saturn) 

Started: 15 Feb 2024

Author: Duncan Lyster
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from visualise_shape_model import visualise_shape_model
from animate_temperature_distribution import animate_temperature_distribution
from matplotlib import colormaps

# Define global variables
# Material properties (currently placeholders)
emmisivity = 0.5                                    # Dimensionless
albedo = 0.5                                        # Dimensionless
thermal_conductivity = 1.0                          # W/mK 
density = 500.0                                     # kg/m^3
specific_heat_capacity = 1000.0                     # J/kgK
beaming_factor = 1.0                                # Dimensionless

# Model setup parameters
layer_thickness = 0.1                               # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 1                                        # Number of layers in the conduction model
solar_distance_au = 1.0                             # AU
solar_distance = solar_distance_au * 1.496e11       # m
solar_luminosity = 3.828e26                         # W
sunlight_direction = np.array([0, -1, 0])           # Unit vector pointing from the sun to the 
timesteps_per_day = 40                              # Number of time steps per day
delta_t = 86400 / timesteps_per_day                 # s (1 day in seconds)
rotation_period = 100000                            # s
max_days = 5                                        # Maximum number of days to run the model for NOTE - this is not intended to be the final model run time as this will be determined by convergence. Just a safety limit.
rotation_axis = np.array([0.3, -0.5, 1])            # Unit vector pointing along the rotation axis
body_orientation = np.array([0, 0, 1])              # Unit vector pointing along the body's orientation
convergence_target = 5                              # K

# Define any necessary functions
def read_shape_model(filename):
    ''' 
    This function reads in the shape model of the body from a .stl file and return an array of facets, each with its own area, position, and normal vector.

    Ensure that the .stl file is saved in ASCII format, and that the file is in the same directory as this script. Additionally, ensure that the model dimensions are in meters and that the normal vectors are pointing outwards from the body. An easy way to convert the file is to open it in Blender and export it as an ASCII .stl file.

    This function will give an error if the file is not in the correct format, or if the file is not found.
    '''
    
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    
    # Attempt to read the file and check for ASCII STL format
    try:
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            if not first_line.startswith('solid'):
                raise ValueError("The file is not in ASCII STL format.")
    except UnicodeDecodeError:
        raise ValueError("The file is not in ASCII STL format or is binary.")

    # Reopen the file for parsing after format check
    with open(filename, 'r') as file:
        lines = file.readlines()

    facets = []
    for i in range(len(lines)):
        if lines[i].strip().startswith('facet normal'):
            normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
            vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
            vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
            vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
            facets.append({'normal': normal, 'vertices': [vertex1, vertex2, vertex3]})
    
    # Process facets to calculate area and centroid
    for i,  facet in enumerate(facets):
        v1, v2, v3 = facet['vertices']
        area = calculate_area(v1, v2, v3)
        centroid = (v1 + v2 + v3) / 3
        # facet['normal'] = normal[i]
        facet['area'] = area
        facet['position'] = centroid
        #initialise insolation and secondary radiation arrays
        facet['insolation'] = np.zeros(timesteps_per_day) # Insolation curve doesn't change day to day
        facet['secondary_radiation'] = np.zeros(len(facets))
        #initialise temperature arrays
        facet['temperature'] = np.zeros((timesteps_per_day * (max_days + 1), n_layers))

    print(f"Read {len(facets)} facets from the shape model.\n")
    
    return facets

def calculate_area(v1, v2, v3):
    '''Calculate the area of the triangle formed by vertices v1, v2, and v3.'''
    u = v2 - v1
    v = v3 - v1
    return np.linalg.norm(np.cross(u, v)) / 2

def calculate_insolation(shape_model):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''

    # Calculate rotation matrix for the body's rotation
    def rotation_matrix(axis, theta):
        '''Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.'''
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # Calculate the zenith angle (angle between the sun and the normal vector of the facet) for each facet at every timestep for one full rotation of the body 
    for facet in shape_model:
        for t in range(timesteps_per_day):
            # Normal vector of the facet at time t=0
            normal = facet['normal']

            new_normal = np.dot(rotation_matrix(rotation_axis, (2 * np.pi * delta_t / rotation_period) * t), normal)

            # Calculate zenith angle
            zenith_angle = np.arccos(np.dot(sunlight_direction, new_normal) / (np.linalg.norm(sunlight_direction) * np.linalg.norm(new_normal)))

            #  Elimate angles where the sun is below the horizon
            if zenith_angle > np.pi / 2:
                insolation = 0

            else:
                # Calculate illumination factor
                # NOTE PLACEHOLDER: This will be a bit tricky to calculate - need to consider whether other facets fall on the vector between the sun and the facet
                illumination_factor = 1

                # Calculate insolation converting AU to m
                insolation = solar_luminosity * (1 - albedo) * illumination_factor * np.cos(zenith_angle) / (4 * np.pi * solar_distance**2) 
                
            # Write the insolation value to the insolation array for this facet at time t
            facet['insolation'][t] = insolation

    print(f"Calculated insolation for each facet.\n")

    # Plot the insolation curve for a single facet with number of days on the x-axis
    plt.plot(shape_model[0]['insolation'])
    plt.xlabel('Number of timesteps')
    plt.ylabel('Insolation (W/m^2)')
    plt.title('Insolation curve for a single facet for one full rotation of the body')
    plt.show()

    return shape_model

def calculate_initial_temperatures(shape_model):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet. It writes the initial temperatures to the data cube.

    Additionally, it plots a histogram of the initial temperatures for all facets.
    '''

    # Calculate initial temperature for each facet
    for facet in shape_model:
        # Calculate the initial temperature based on the integrated insolation curve
        # Integrate the insolation curve to get the total energy received by the facet over one full rotation
        total_energy = np.trapz(facet['insolation'], dx=delta_t)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law and set the initial temperature of all layers to the same value
        for layer in range(n_layers):
            facet['temperature'][0][layer] = (total_energy / (emmisivity * facet['area'] * 5.67e-8))**(1/4)

    print(f"Calculated initial temperatures for each facet.\n")

    # Plot a histogram of the initial temperatures for all facets
    initial_temperatures = [facet['temperature'][0][0] for facet in shape_model]
    plt.hist(initial_temperatures, bins=20)
    plt.xlabel('Initial temperature (K)')
    plt.ylabel('Number of facets')
    plt.title('Initial temperature distribution of all facets')
    plt.show()

    return shape_model

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Get the shape model and setup data storage arrays
    filename = "67P_low_res.stl"
    shape_model = read_shape_model(filename)

    # Visualise the shape model
    visualise_shape_model(filename, rotation_axis, rotation_period, solar_distance_au, sunlight_direction)

    # Calculate insolation array for each facet
    shape_model = calculate_insolation(shape_model)

    # Calulate initial temperature array
    shape_model = calculate_initial_temperatures(shape_model)

    # Calculate secondary radiation array
        # Ray tracing to work out which facets are visible from each facet
        # Calculate the geometric coefficient of secondary radiation from each facet
        # Write the index and coefficient to the data cube
    
    convergence_factor = 10 # Set to a value greater than 1 to start the iteration
    day = 0 

    # Proceed to iterate the model until it converges
    while day < max_days and convergence_factor > 1:
        for time_step in range(timesteps_per_day):
            for facet in shape_model:
                current_step = int(time_step + (day * timesteps_per_day))
                # Calculate insolation term, bearing in mind that the insolation curve is constant for each facet and repeats every rotation period
                insolation_term = facet['insolation'][time_step] * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate re-emitted radiation term
                re_emitted_radiation_term = emmisivity * beaming_factor * 5.67e-8 * (facet['temperature'][current_step][0]**4) * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate secondary radiation term (identify facets above horizon first, then check if they face, same process for shadows but maybe segment facet into shadow/light with a calculated line?)
                # Calculate conducted heat term
                # Calculate sublimation energy loss term
                # Calculate new surface temperature
                # Calculate new temperatures for all sub-surface layers
                # Save the new temperatures to the data cube

                # Calculate the new temperature of the surface layer (currently very simplified)
                facet['temperature'][current_step + 1][0] = facet['temperature'][current_step][0] + insolation_term - re_emitted_radiation_term 

        # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
        day += 1

        temperature_error = 0
        for facet in shape_model:
                temperature_error += abs(facet['temperature'][day * timesteps_per_day][0] - facet['temperature'][(day - 1) * timesteps_per_day][0])

        convergence_factor = (temperature_error / (len(shape_model))) / convergence_target

        print(f"Day {day} temperature error: {temperature_error / (len(shape_model))} K\n")

    day -= 1    
    
    # Post-loop check to display appropriate message
    if convergence_factor <= 1:
        print(f"Convergence target achieved after {day} days.\n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")

        # Create an array of temperatures at each timestep in final day for each facet
        # Initialise the array
        final_day_temperatures = np.zeros((len(shape_model), timesteps_per_day))

        # Print length of shape model
        print(f"Length of shape model: {len(shape_model)}\n")

        # Fill the array
        for i, facet in enumerate(shape_model):
            for t in range(timesteps_per_day):
                final_day_temperatures[i][t] = facet['temperature'][day * timesteps_per_day + t][0]

        print(f"Final day temperatures array: {final_day_temperatures}\n")
        
        # Visualise the results - animation of final day's temperature distribution
        animate_temperature_distribution(filename, final_day_temperatures, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day, delta_t)

        # Save a sample of the final day's temperature distribution to a file
        np.savetxt('test_data/final_day_temperatures.csv', final_day_temperatures, delimiter=',')
    
    else:
        print(f"Maximum days reached without achieving convergence. \n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")

    # Save the final day temperatures to a file that can be used with ephemeris to produce instrument simulations

# Call the main program to start execution
if __name__ == "__main__":
    main()
