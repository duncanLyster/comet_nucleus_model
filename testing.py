import concurrent.futures
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product, combinations
from thermophysical_body_model import (
    thermophysical_body_model,
    Simulation,
    read_shape_model,
    calculate_visible_facets,
    calculate_insolation,
    calculate_initial_temperatures,
    calculate_secondary_radiation_coefficients
)
from sklearn.linear_model import LinearRegression

def run_model_with_parameters(combination, shape_model_snapshot, path_to_setup_file, param_names):
    simulation = Simulation(path_to_setup_file)  # Create a new simulation instance for thread safety

    # Update simulation parameters for the current combination
    for param_name, value in zip(param_names, combination):
        setattr(simulation, param_name, value)
    
    # Run the thermophysical model and calculate statistics
    start_time = time.time()
    final_timestep_temperatures = thermophysical_body_model(shape_model_snapshot, simulation)
    execution_time = time.time() - start_time

    # Check if temperatures were calculated
    if final_timestep_temperatures is None:
        mean_temperature = None
        temperature_iqr = None
    else:
        mean_temperature = np.mean(final_timestep_temperatures)
        temperature_iqr = np.percentile(final_timestep_temperatures, 75) - np.percentile(final_timestep_temperatures, 25)

    return {
        'parameters': dict(zip(param_names, combination)),
        'mean_temperature': mean_temperature,
        'temperature_iqr': temperature_iqr,
        'execution_time': execution_time
    }

def generate_heatmaps(df, parameters, value_column, output_folder):
    for param_pair in combinations(parameters, 2):  # Get all combinations of parameter pairs
        pivot_table = df.pivot_table(index=param_pair[0], columns=param_pair[1], values=value_column, aggfunc="mean")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': value_column})
        plt.title(f'{value_column} by {param_pair[0]} and {param_pair[1]}')
        plt.xlabel(param_pair[1])
        plt.ylabel(param_pair[0])
        plt.savefig(f"{output_folder}heatmap_{param_pair[0]}_vs_{param_pair[1]}.png")  # Save each heatmap as a PNG file
        plt.close()  # Close the figure to avoid displaying it inline if running in a notebook

def main():
    # Define a function to calculate comet temperature given input parameters
    def calculate_comet_temperature(input_parameters):
        # Assign input parameters
        emissivity, albedo, thermal_conductivity, density, specific_heat_capacity, beaming_factor, \
        solar_distance_au, solar_luminosity, = input_parameters

        # Load the final day temperatures
        final_day_temperatures = np.loadtxt('outputs/final_day_temperatures.csv', delimiter=',')

        # Return the mean temperature of the comet
        return np.mean(final_day_temperatures)

    # Define a function to perform linear regression
    def perform_linear_regression(X, y, parameter_name):
        # Reshape X and y
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict y values
        y_pred = model.predict(X)

        # Plot the results
        plt.scatter(X, y, color='blue', label='Actual data')
        plt.plot(X, y_pred, color='red', label='Linear regression')
        plt.xlabel(parameter_name)
        plt.ylabel('Comet Temperature (K)')
        plt.title(f'Linear Regression: {parameter_name} vs Comet Temperature')
        plt.legend()
        plt.show()

    # Model setup and initialization
    shape_model_name = "5km_ico_sphere_80_facets.stl"
    path_to_shape_model_file = f"shape_models/{shape_model_name}"
    path_to_setup_file = "model_setups/generic_model_parameters.json"
    simulation = Simulation(path_to_setup_file)
    shape_model = read_shape_model(path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days)

    # Model component calculations
    shape_model = calculate_visible_facets(shape_model)
    shape_model = calculate_insolation(shape_model, simulation)
    shape_model = calculate_initial_temperatures(shape_model, simulation.n_layers, simulation.emissivity)
    shape_model = calculate_secondary_radiation_coefficients(shape_model)

    # Parameters for analysis with their ranges
    parameter_ranges = {
        'emissivity': np.linspace(0.4, 0.6, 3),
        'albedo': np.linspace(0.4, 0.6, 3),
        # Add other parameters here
    }

    param_names = list(parameter_ranges.keys())
    param_values = [parameter_ranges[name] for name in param_names]
    all_combinations = list(product(*param_values))

    # Include param_names with each combination
    param_combinations_with_shape_model = [
        (combination, shape_model, path_to_setup_file, param_names) for combination in all_combinations
    ]

    # Read input parameters from CSV file
    input_parameters_df = pd.read_csv('Datasheet_comet.csv')
    input_parameters_df = input_parameters_df.dropna()

    # Loop through each row in the dataframe
    for index, row in input_parameters_df.iterrows():
        # Convert row to list (input parameters)
        input_parameters = row.tolist()

        # Calculate comet temperature
        comet_temperature = calculate_comet_temperature(input_parameters)

        # Append comet temperature to the dataframe
        input_parameters_df.at[index, 'Comet Temperature'] = comet_temperature

    # Perform linear regression for each input parameter against comet temperature
    for column in input_parameters_df.columns[1:]:
        X = input_parameters_df[column].values
        y = input_parameters_df['Comet Temperature'].values
        perform_linear_regression(X, y, column)

    print("Starting parameter exploration...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_model_with_parameters, *params) for params in param_combinations_with_shape_model}
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing models"):
            results.append(future.result())

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)

    # Flatten the 'parameters' dictionary into separate columns for easier analysis
    parameters_df = pd.json_normalize(results_df['parameters'])
    results_df = pd.concat([results_df.drop(columns=['parameters']), parameters_df], axis=1)

    output_folder = "runner_outputs/"

    # Call the function for generating and saving heatmaps
    generate_heatmaps(results_df, param_names, 'mean_temperature', output_folder)

    print("Heatmaps generated and saved.")

    csv_file_path = "runner_outputs/thermophysical_model_results.csv"

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_file_path, index=False)

    print("Results saved to 'thermophysical_model_results.csv'.")

if __name__ == '__main__':
    main()
