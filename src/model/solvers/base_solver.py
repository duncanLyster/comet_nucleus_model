# Base class for temperature solvers

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utilities.utils import conditional_print
from src.utilities.utils import conditional_tqdm

class TemperatureSolver:
    def __init__(self, name):
        self.name = name
        self.required_parameters = []

    def initialize_temperatures(self, thermal_data, simulation, config):
        """Initialize temperature arrays based on average insolation"""
        # Stefan-Boltzmann constant
        sigma = 5.67e-8

        def process_facet(insolation, emissivity, sigma):
            # Calculate the initial temperature based on average power in
            power_in = np.mean(insolation)
            # Calculate the temperature of the facet using the Stefan-Boltzmann law
            # Guard: power_in must be non-negative to avoid NaN from fractional power of negative
            if power_in <= 0:
                power_in = 1e-20  # Small positive to avoid 0/0 and give ~0 K
            temp = (power_in / (emissivity * sigma))**(1/4)
            # Floor at 50 K to prevent numerical instability in implicit solver linearization
            return max(temp, 50.0)

        # Parallel processing of facets
        conditional_print(config.silent_mode, f"Calculating initial temperatures for {thermal_data.temperatures.shape[0]} facets...")
        results = Parallel(n_jobs=config.n_jobs)(
            delayed(process_facet)(thermal_data.insolation[i], simulation.emissivity, sigma) 
            for i in range(thermal_data.temperatures.shape[0])
        )

        conditional_print(config.silent_mode, f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

        # Self-heating-aware equilibrium: iterate T_i^4 = <ins_i>/(eps*sigma) + sum_j F_ij T_j^4.
        # Without this, concave geometry (crater kernels, contact-binary necks) is initialised
        # far below its equilibrium and high-TI runs relax so slowly that the day-to-day
        # convergence test can pass while the surface is still cold.
        tvfs = getattr(thermal_data, 'thermal_view_factors', None)
        if config.include_self_heating and tvfs is not None and len(tvfs) == len(results) \
                and any(len(v) > 0 for v in tvfs):
            drive_t4 = np.array([max(r, 50.0) ** 4.0 for r in results])   # <ins>/(eps*sigma)
            mean_t4 = drive_t4.copy()
            vis = thermal_data.visible_facets
            for _ in range(30):
                irr = np.array([np.sum(mean_t4[np.asarray(vis[i])] * np.asarray(tvfs[i]))
                                if len(tvfs[i]) else 0.0 for i in range(len(results))])
                mean_t4 = drive_t4 + irr
            results = [max(t4 ** 0.25, 50.0) for t4 in mean_t4]
            conditional_print(config.silent_mode,
                              "Initial temperatures include self-heating equilibrium.")

        # Update both surface and layer temperatures with initial values
        for i, temperature in conditional_tqdm(enumerate(results), config.silent_mode, total=len(results), desc='Saving temps'):
            thermal_data.temperatures[i, :] = temperature
            thermal_data.layer_temperatures[i, 0, :] = temperature
            thermal_data.layer_temperatures[i, 1, :] = temperature

        conditional_print(config.silent_mode, "Initial temperatures saved for all facets.")
        return thermal_data

    def solve(self, thermal_data, shape_model, simulation, config):
        """Run the temperature solving algorithm"""
        raise NotImplementedError("Subclasses must implement solve")

    def get_required_parameters(self):
        return self.required_parameters 