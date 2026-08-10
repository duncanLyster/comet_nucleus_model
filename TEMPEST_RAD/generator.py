"""
Generate High-Fidelity SPECTRAL Roughness Lookup Tables (LUTs) for TEMPEST.

This script simulates the thermal behavior of a rough surface (modeled as a spherical crater)
across a wide range of conditions to generate a radiometric correction table.

Changes in V2:
- SPECTRAL OUTPUT: Calculates radiance ratios using Planck function instead of T^4.
- High-Resolution internal simulation (720 steps) for stability.
- Uses Dimensionless Thermal Parameter (Theta) for grid.

Dimensions:
1. Thermal Parameter (Theta): Capture thermal memory.
2. Opening Angle: Capture crater shape (90 = Hemisphere).
3. Wavelength (Lambda): Capture spectral non-linearity.
4. Latitude: Capture diurnal history (Equatorial vs Polar).
5. Sun Phase (Time): The rotation angle of the body (0-360).
6. Emission Angle (View): Observer angle (0-90).
7. Azimuth Angle (View): Relative angle between Sun and Observer (0-180).

Output:
A 7D tensor containing the Radiance Factor:
    R(lambda) = Radiance_Rough(lambda) / Radiance_Smooth(lambda)
"""

import os
import sys
import time
import traceback
import numpy as np
import h5py
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

# Ensure root directory is in the Python path to allow 'from src...' imports
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.model.facet import Facet
from src.model.simulation import Simulation, ThermalData
from src.model.solvers import TemperatureSolverFactory
from src.utilities.config import Config
from src.utilities.locations import Locations
from src.utilities.utils import calculate_rotation_matrix, rays_triangles_intersection

# Import the core modules for the Standard Simulation
from src.model.insolation import calculate_insolation
from src.model.view_factors import calculate_all_view_factors, calculate_thermal_view_factors

# ============================================================================
# CONFIGURATION: LUT Generation Parameters
# ============================================================================
# Edit this section to customize your LUT generation.
# All parameters are defined here for easy modification.

# --- PARALLELISM ---
N_JOBS = 32             # Number of parallel workers.
                        # -1 = use ALL available CPU cores (auto-detect).
                        # Set to a specific number to limit, e.g. 32 on a shared server.

# --- OUTPUT ---
OUTPUT_DIR = "lut_output"       # Directory for per-theta HDF5 files
OUTPUT_PREFIX = "roughness_lut_spectral_v2"  # Filename prefix
# Each theta value is saved as: {OUTPUT_DIR}/{OUTPUT_PREFIX}_theta_{value}.h5

# --- LUT GRID DIMENSIONS ---
# These define the dimensionality and resolution of your lookup table.
# Higher resolution = more accurate, but slower generation and larger files.

# Thermal Parameter (Theta = TI * sqrt(omega) / (emissivity * sigma * Tss^3))
# Controls how quickly surface responds to solar heating
# At P=10h: Theta=0.04→TI=10 (dust), Theta=0.4→TI=100 (regolith), Theta=8→TI=2000 (rock)
THETA_VALUES = np.logspace(-1.5, 1, 15)  # 15 points from ~0.03 to 10 (TI: 10-2500)
# For testing, use: THETA_VALUES = np.array([0.5])  # TI≈100 at P=10h

# Crater Opening Angles (degrees)
# 90 = hemisphere, smaller = bowl-shaped
OPENING_ANGLES = [90.0]

# Wavelengths for spectral radiance calculations (microns)
# 12 bands spanning Wien peak to Rayleigh-Jeans tail
WAVELENGTHS_MICRONS = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0, 200.0]

# Latitudes (degrees) - Surface position on rotating body
# Captures diurnal variation from equator (0°) to pole (90°)
LATITUDE_VALUES = np.arange(0.0, 92.5, 2.5)  # 37 points, 2.5° spacing
# For testing, use: LATITUDE_VALUES = np.array([0.0, 30.0, 60.0, 85.0])

# Emission Angles (degrees) - Observer viewing angle
# 0° = looking straight down, 90° = grazing view
EMISSION_ANGLES = np.linspace(0, 90, 37)  # 37 points, 2.5° spacing (matches latitude)

# Azimuth Angles (degrees) - Angle between sun and observer
# 0° = opposition (sun behind observer), 180° = full phase
AZIMUTH_ANGLES = np.linspace(0, 180, 36)  # 37 points, ~4.9° spacing

# --- TEMPORAL RESOLUTION ---
LUT_TIMESTEPS = 180    # Output resolution: 360°/180 = 2° per step
SIM_TIMESTEPS = 720    # Internal simulation: 4x finer for stability

# --- CRATER GEOMETRY ---
# Resolution of the spherical crater mesh
CRATER_SUBFACETS = 2000   # Facet count in crater
                          # Note: Actual count may differ slightly due to geodesic tessellation.
                          # Geometry is computed ONCE in main() and cached/shared with all workers.
                          # Timing on 32 cores: ~20 hours total for all 15 thetas.
                          # (10000 facets would take ~250 hours due to O(N^2) visibility kernel)

# View factor ray tracing resolution
VIEW_FACTOR_RAYS = 20000  # Rays per facet for view factor calculation
                          # Higher = better energy conservation (aim for <5% error).
                          # Computed ONCE in main() and cached to disk.
                          # (Scaled down proportionally with CRATER_SUBFACETS)

# --- PHYSICAL PARAMETERS (Standard Reference Conditions) ---
# These define the "canonical" surface for the LUT
# Actual simulations scale these using the thermal parameter (Theta)

REFERENCE_ALBEDO = 0.12        # Bond albedo (typical for asteroids)
REFERENCE_EMISSIVITY = 0.95    # Infrared emissivity
ROTATION_PERIOD_HOURS = 10.0   # Body rotation period
SOLAR_DISTANCE_AU = 1.0        # Distance from sun (1 AU)
SOLAR_LUMINOSITY = 3.828e26    # Solar luminosity (W)

# Subsurface thermal model
N_LAYERS = 40           # Number of subsurface layers
MAX_DAYS = 80           # Maximum days to reach thermal equilibrium
MIN_DAYS = 5            # Minimum days before checking convergence
CONVERGENCE_TARGET = 0.2  # Convergence threshold (K for mean temperature change);
                          # tight because high-Theta cavity runs relax slowly and a
                          # loose test can pass while the crater is still warming

# ============================================================================
# End of User Configuration
# ============================================================================

# --- Environment overrides --------------------------------------------------
# joblib's loky workers RE-IMPORT this module, so monkey-patched module globals
# in a launcher script are silently lost in the workers (the parent still sees
# the patched values, the workers see the defaults, and the returned grids no
# longer match the allocated output arrays).  Env vars survive the fork/spawn,
# so scripted runs must use these instead of (or in addition to) patching.
SIM_TIMESTEPS    = int(os.environ.get('LUT_SIM_TIMESTEPS', SIM_TIMESTEPS))
LUT_TIMESTEPS    = int(os.environ.get('LUT_LUT_TIMESTEPS', LUT_TIMESTEPS))
CRATER_SUBFACETS = int(os.environ.get('LUT_CRATER_SUBFACETS', CRATER_SUBFACETS))
VIEW_FACTOR_RAYS = int(os.environ.get('LUT_VIEW_FACTOR_RAYS', VIEW_FACTOR_RAYS))
N_JOBS           = int(os.environ.get('LUT_N_JOBS', N_JOBS))
OUTPUT_DIR       = os.environ.get('LUT_OUTPUT_DIR', OUTPUT_DIR)
OUTPUT_PREFIX    = os.environ.get('LUT_OUTPUT_PREFIX', OUTPUT_PREFIX)
if 'LUT_THETA_VALUES' in os.environ:
    THETA_VALUES = np.array([float(x) for x in os.environ['LUT_THETA_VALUES'].split(',')])
# crater opening angle (90 = hemisphere; >90 = beyond-hemispherical / deeper cavity)
if 'LUT_OPENING_ANGLES' in os.environ:
    OPENING_ANGLES = [float(x) for x in os.environ['LUT_OPENING_ANGLES'].split(',')]
# restrict the latitude grid — for single-latitude convergence/benchmark runs
if 'LUT_LATITUDE_VALUES' in os.environ:
    LATITUDE_VALUES = np.array([float(x) for x in os.environ['LUT_LATITUDE_VALUES'].split(',')])
# inter-facet scattering iterations inside the crater (1 = single bounce; higher =
# converges the multiple-scattering absorption/beaming enhancement)
N_SCATTERS_ENV = int(os.environ.get('LUT_N_SCATTERS', '1'))

def planck_function(wavelength_um, temp_k):
    """
    Calculate spectral radiance B(lambda, T).
    Result unit: W / (m^2 sr um)
    
    Constants:
    c1 = 2 * h * c^2 = 1.191042e8 W um^4 / (m^2 sr)
    c2 = h * c / k_B = 1.4387752e4 um K
    """
    c1 = 1.191042e8  
    c2 = 1.4387752e4 
    
    # Avoid overflow/div-by-zero for T=0 or very small
    t_safe = np.maximum(temp_k, 1e-5)
    
    # Calculate
    val = c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))
    return val


@njit(cache=True)
def compute_visibility_mask(d_local, normals, centers, triangles):
    """Which subfacets are visible from local-space view direction d_local.

    Uses Möller–Trumbore ray-triangle intersection with early exit.
    Zero heap allocations: runs entirely in numba-compiled scalar code.

    Args:
        d_local:    (3,) view direction in crater-local frame
        normals:    (N, 3) subfacet normals in local frame
        centers:    (N, 3) subfacet centers in local frame
        triangles:  (N, 3, 3) subfacet triangle vertices in local frame

    Returns:
        visible: (N,) boolean array, True = visible from this direction
    """
    N = normals.shape[0]
    visible = np.zeros(N, dtype=np.bool_)
    dx, dy, dz = d_local[0], d_local[1], d_local[2]

    for i in range(N):
        # Skip back-facing subfacets
        if normals[i, 0] * dx + normals[i, 1] * dy + normals[i, 2] * dz <= 0.0:
            continue

        ox, oy, oz = centers[i, 0], centers[i, 1], centers[i, 2]
        blocked = False

        for j in range(N):
            if i == j:
                continue

            # Möller–Trumbore ray-triangle intersection
            e1x = triangles[j, 1, 0] - triangles[j, 0, 0]
            e1y = triangles[j, 1, 1] - triangles[j, 0, 1]
            e1z = triangles[j, 1, 2] - triangles[j, 0, 2]
            e2x = triangles[j, 2, 0] - triangles[j, 0, 0]
            e2y = triangles[j, 2, 1] - triangles[j, 0, 1]
            e2z = triangles[j, 2, 2] - triangles[j, 0, 2]

            hx = dy * e2z - dz * e2y
            hy = dz * e2x - dx * e2z
            hz = dx * e2y - dy * e2x

            a = e1x * hx + e1y * hy + e1z * hz
            if a > -1e-10 and a < 1e-10:
                continue

            f = 1.0 / a
            sx = ox - triangles[j, 0, 0]
            sy = oy - triangles[j, 0, 1]
            sz = oz - triangles[j, 0, 2]
            u = f * (sx * hx + sy * hy + sz * hz)
            if u < 0.0 or u > 1.0:
                continue

            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x
            v = f * (dx * qx + dy * qy + dz * qz)
            if v < 0.0 or u + v > 1.0:
                continue

            t = f * (e2x * qx + e2y * qy + e2z * qz)
            if t > 1e-6:
                blocked = True
                break

        if not blocked:
            visible[i] = True

    return visible


class ReferenceConfig(Config):
    """
    A minimal Config subclass that does not require a file.
    Used to generate the generic LUT under standard reference conditions.
    All values now pulled from configuration constants above.
    """
    def __init__(self):
        # Initialize Locations (required by base class, but we won't use paths)
        self.locations = Locations()
        
        # Build config_data from configuration constants
        self.config_data = {
            # Physical parameters from config
            'solar_distance_au': SOLAR_DISTANCE_AU,
            'solar_luminosity': SOLAR_LUMINOSITY,
            'albedo': REFERENCE_ALBEDO,
            'emissivity': REFERENCE_EMISSIVITY,
            'beaming_factor': 1.0,
            'subsurface_heating_flux': 0.0,
            'sunlight_direction': [1.0, 0.0, 0.0],  # Sun along +X axis
            
            # Solver Settings from config
            'temp_solver': 'tempest_implicit',
            'convergence_method': 'mean',
            'convergence_target': CONVERGENCE_TARGET,
            'max_days': MAX_DAYS,
            'min_days': MIN_DAYS,
            'n_layers': N_LAYERS,
            'include_shadowing': True,
            'n_scatters': N_SCATTERS_ENV,
            'include_self_heating': True,
            # thermal VFs are cheap (EPF multiply) and their cache was poisoned by
            # the missing set_secondary_radiation_view_factors call above
            'force_recalculate_thermal_view_factors': True,
            'n_jobs': N_JOBS,  # Parallel workers for view factor calculation etc.
            'vf_rays': 10000,  # Not used (overridden in simulate_crater_diurnal_cycle)
            'intra_facet_scatters': 2,
            'silent_mode': True,  # Suppress verbose solver output in parallel workers
            
            # Kernel/Roughness specific from config
            'apply_kernel_based_roughness': True,
            'roughness_kernel': 'spherical_section',
            'kernel_subfacets_count': CRATER_SUBFACETS,
            'kernel_profile_angle_degrees': 90.0,  # Will be set per-case
            
            # Disable Plotting/Saving
            'plot_temp_curve': [],
            'save_temp_curve_data': False,
            'save_visible_phase_curve_data': False,
            'save_thermal_phase_curve_data': False,
            'animate_subfacets': False,
            'animate_final_day_temp_distribution': False,
            'animate_shadowing': False,
            
            # Shape model settings (dummy)
            'shape_model_file': 'cube_without_roughness.obj',
            'ra_degrees': 0.0,
            'dec_degrees': 90.0,
            'rotation_period_hours': ROTATION_PERIOD_HOURS,
            
            # Physical props (will be overwritten by generator)
            'density': 1000.0,
            'specific_heat_capacity': 1000.0,
            'thermal_inertia': 100.0 
        }
        
        # Also set attributes for Config compatibility if needed by other utils
        self.load_settings()
        
        # Override path to avoid errors
        self.path_to_shape_model_file = None


def rotate_to_lat(vector, lat_degrees):
    """
    Rotate a vector (initially up Z, Lat 90) to a new latitude.
    Rotation is around Y-axis by (90 - lat) degrees.
    """
    angle_rad = np.radians(90.0 - lat_degrees)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return np.dot(Ry, vector)

def simulate_crater_diurnal_cycle(theta, opening_angle, latitude, config, n_timesteps,
                                  precomputed, return_thermal_data=False):
    """
    Run a full diurnal simulation for a single crater configuration.
    Uses pre-computed geometry (mesh, view factors) to avoid redundant setup.
    Only latitude-dependent calculations (insolation, thermal solve) run per case.
    """
    # 1. Setup Simulation Object
    simulation = Simulation(config)
    
    # Only independent inputs are set here; Simulation derives delta_t,
    # angular_velocity, skin_depth etc. on read, so they stay consistent.
    simulation.timesteps_per_day = n_timesteps
    simulation.rotation_period_hours = ROTATION_PERIOD_HOURS
    simulation.density = 2000.0 # Standard Value
    simulation.specific_heat_capacity = 1000.0 # Standard Value
    omega = simulation.angular_velocity

    # Calculate Thermal Inertia from Dimensionless Theta
    # P = I * sqrt(omega) / (epsilon * sigma * Tss^3)
    # Tss = ((1-A) * S / (epsilon * sigma))^0.25

    
    solar_flux = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
    boltzmann = 5.670374419e-8
    epsilon = simulation.emissivity
    albedo = simulation.albedo
    
    # Sub-solar temperature
    val = ((1 - albedo) * solar_flux) / (epsilon * boltzmann)
    tss = val ** 0.25
    
    # Avoid division by zero
    if omega <= 0: omega = 1e-9
    
    thermal_inertia = (theta * epsilon * boltzmann * (tss**3)) / np.sqrt(omega)
    
    # Update Simulation with derived thermal inertia. Conductivity, skin depth,
    # layer thickness and diffusivity all follow from this automatically.
    simulation.thermal_inertia = thermal_inertia

    # 2. Build shape model from pre-computed geometry (mesh + view factors computed once in main)
    n_facets = precomputed['n_facets']
    rotation_to_equator = precomputed['rotation_to_equator']
    
    # Sun direction with declination (latitude proxy for solar illumination angle)
    # Keeps crater at equator and tilts sun to vary illumination.
    sun_declination = latitude
    lat_rad = np.radians(sun_declination)
    simulation.sunlight_direction = np.array([np.cos(lat_rad), 0.0, np.sin(lat_rad)])
    
    shape_model = []
    for i in range(n_facets):
        new_f = Facet(
            precomputed['rotated_normals'][i].copy(),
            precomputed['rotated_vertices'][i].copy(),
            simulation.timesteps_per_day,
            simulation.max_days,
            simulation.n_layers,
            config.calculate_energy_terms
        )
        shape_model.append(new_f)
    
    # 3. Setup Thermal Data
    thermal_data = ThermalData(n_facets, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)
    
    # 4. Set visible facets (all-see-all for concave crater)
    all_indices = np.arange(n_facets)
    for i, f in enumerate(shape_model):
        f.visible_facets = np.concatenate([all_indices[:i], all_indices[i+1:]])
    visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
    thermal_data.set_visible_facets(visible_facets_list)
    
    # 5. Use pre-loaded view factors from precomputed dict (no disk I/O)
    thermal_data.set_secondary_radiation_view_factors(precomputed['all_view_factors'])
    
    from numba.typed import List
    numba_vfs = List()
    for arr in precomputed['thermal_view_factors']:
        numba_vfs.append(arr)
    thermal_data.thermal_view_factors = numba_vfs

    # 6. Calculate Insolation (with Shadowing)
    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)
    

    
    # 7. Run Solver (Standard Implicit)
    from src.model.solvers import TemperatureSolverFactory
    
    solver = TemperatureSolverFactory.create(config.temp_solver)
    thermal_data = solver.initialize_temperatures(thermal_data, simulation, config)
    
    res = solver.solve(thermal_data, shape_model, simulation, config)
    
    if res['final_day_temperatures'] is None:
        raise ValueError(f"Solver failed for Theta={theta}, lat={latitude}")
        
    rough_temps = res['final_day_temperatures'] # (N_facets, T)
    
    # 8. Solve Smooth Reference
    # Create flat facet at SAME LATITUDE as rough crater for proper comparison
    # Normal should point at same angle as crater opening (latitude from equator)
    lat_rad = np.radians(latitude)
    flat_n = np.array([1.0, 0.0, 0.0])  # Normal pointing toward sun at equator
    flat_n /= np.linalg.norm(flat_n)
    
    # Vertices perpendicular to normal
    # Use two perpendicular vectors in the plane perpendicular to flat_n
    # Scaled to give area = 1.0 to match crater opening area
    scale = 1.0 / np.sqrt(2.0)
    
    # Create perpendicular basis vectors
    if abs(flat_n[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])
    
    u = np.cross(flat_n, up)
    u /= np.linalg.norm(u)
    v = np.cross(flat_n, u)
    
    flat_v = [
        -scale * u - scale * v,
        scale * u - scale * v,
        scale * v
    ]
    
    flat_facet = Facet(flat_n, flat_v, simulation.timesteps_per_day, simulation.max_days, simulation.n_layers, config.calculate_energy_terms)
    
    # Set world_to_local_rotation to match the crater's rotation at equator
    flat_facet.world_to_local_rotation = rotation_to_equator.T
    
    smooth_shape = [flat_facet]
    
    smooth_data = ThermalData(1, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)
    # set_visible_facets expects a LIST of arrays (one per facet)
    # For a single isolated facet, it has no visible neighbors
    smooth_data.set_visible_facets([np.array([], dtype=np.int64)])
    flat_facet.visible_facets = np.array([], dtype=np.int64)
    
    # Dummy View Factors
    smooth_data.set_secondary_radiation_view_factors([np.array([])])
    # Empty thermal view factors
    from numba.typed import List
    numba_vfs_smooth = List()
    # Need to match the expected type inside Numba functions (float64[:])
    # The solver expects a list of arrays. For a single facet with no self-view, it's an empty array.
    numba_vfs_smooth.append(np.array([], dtype=np.float64))
    
    smooth_data.thermal_view_factors = numba_vfs_smooth
    
    smooth_data = calculate_insolation(smooth_data, smooth_shape, simulation, config)
    # Create fresh solver instance for smooth reference to avoid state contamination
    smooth_solver = TemperatureSolverFactory.create(config.temp_solver)
    smooth_data = smooth_solver.initialize_temperatures(smooth_data, simulation, config)
    res_smooth = smooth_solver.solve(smooth_data, smooth_shape, simulation, config)
    
    smooth_temps = res_smooth['final_day_temperatures'][0] # (T,)

    # Reconstruct Sun Vectors for downstream geometry calc
    timesteps = simulation.timesteps_per_day
    sun_vectors_body = np.zeros((timesteps, 3))
    
    for t in range(timesteps):
        rot = calculate_rotation_matrix(simulation.rotation_axis, (2 * np.pi / timesteps) * t)
        # S_body = R^T * S_inertial
        s_body = np.dot(rot.T, simulation.sunlight_direction)
        sun_vectors_body[t] = s_body / np.linalg.norm(s_body)

    # Return temperatures and sun vectors (geometry comes from precomputed dict).
    # return_thermal_data adds the solved ThermalData as a 4th element, so callers
    # can reach the per-subfacet insolation (needed to reproduce scattered optical
    # flux, e.g. the Magri et al. 2018 Table A1 bolometric row, which validates the
    # crater geometry and scattering with no thermal physics involved).  Default
    # False keeps the 3-tuple that process_single_case unpacks.
    if return_thermal_data:
        return rough_temps, smooth_temps, sun_vectors_body, thermal_data
    return rough_temps, smooth_temps, sun_vectors_body



def plot_wireframe(mesh, filename="TEMPEST_RAD/diagnostics/plots/crater_wireframe.png"):
    """
    Save a 3D wireframe plot of the crater mesh.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = []
    for entry in mesh:
        # entry['vertices'] is (3,3) array
        v = entry['vertices']
        # Close the loop for wireframe
        v_loop = np.vstack([v, v[0]])
        ax.plot(v_loop[:,0], v_loop[:,1], v_loop[:,2], 'k-', lw=0.5, alpha=0.3)
        vertices.append(v)
        
    vertices_flat = np.vstack(vertices).reshape(-1, 3)
    max_range = np.array([vertices_flat[:,0].max()-vertices_flat[:,0].min(), 
                          vertices_flat[:,1].max()-vertices_flat[:,1].min(), 
                          vertices_flat[:,2].max()-vertices_flat[:,2].min()]).max() / 2.0

    mid_x = (vertices_flat[:,0].max()+vertices_flat[:,0].min()) * 0.5
    mid_y = (vertices_flat[:,1].max()+vertices_flat[:,1].min()) * 0.5
    mid_z = (vertices_flat[:,2].max()+vertices_flat[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"Crater Mesh ({len(mesh)} facets)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    out_dir = os.path.dirname(filename)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved wireframe to {filename}")

def process_single_case(theta, opening_angle, lat, config, precomputed):
    """
    Run simulation and compute viewing geometry grid.
    Returns: 4D array (Time, Wave, Emi, Azi), 1D array (Normalization_Factor)
    """
    try:
        rough_temps_sim, smooth_temps_sim, sun_vectors_sim = simulate_crater_diurnal_cycle(theta, opening_angle, lat, config, SIM_TIMESTEPS, precomputed)
    except Exception as e:
        print(f"Error in simulation (Th={theta}, ang={opening_angle}, lat={lat}): {e}")
        traceback.print_exc()
        return np.full((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), np.nan, dtype=np.float32), np.nan

    # Resample to LUT grid
    indices = np.linspace(0, SIM_TIMESTEPS-1, LUT_TIMESTEPS, dtype=int)
    rough_temps = rough_temps_sim[:, indices]
    smooth_temps = smooth_temps_sim[indices]
    sun_vectors = sun_vectors_sim[indices]
    
    # Result: Time x Wavelength x Emi x Azi
    result_grid = np.zeros((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), dtype=np.float32)
    
    # Geometry arrays from precomputed (computed once in main, shared by all workers)
    normals = precomputed['canonical_normals']
    areas = precomputed['canonical_areas']
    triangles = precomputed['canonical_vertices']
    centers = precomputed['canonical_centers']
    facet_normal = precomputed['aperture_normal']
    world_to_local = precomputed['world_to_local']
    aperture_area = precomputed['aperture_area']
    
    # Pre-calculated Geometry terms
    em_rad = np.radians(EMISSION_ANGLES)
    az_rad = np.radians(AZIMUTH_ANGLES)
    d_em = np.diff(em_rad); d_em = np.append(d_em, d_em[-1])
    d_az = np.diff(az_rad); d_az = np.append(d_az, d_az[-1])

    # Pre-compute cos(e)*sin(e) solid-angle weight grid for normalization.
    # Energy conservation requires: ∫ R(e,φ) cos(e) sin(e) de dφ = π
    # i.e., the cos(e)*sin(e)-weighted mean of R must equal 1.0.
    cos_e_arr = np.cos(em_rad)
    sin_e_arr = np.sin(em_rad)
    cse_weights = np.zeros((len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)))
    for ie in range(len(EMISSION_ANGLES)):
        for ia in range(len(AZIMUTH_ANGLES)):
            cse_weights[ie, ia] = cos_e_arr[ie] * sin_e_arr[ie] * d_em[ie] * d_az[ia] * 2.0
    ref_integral = np.sum(cse_weights)  # ≈ π

    # Loop over Timesteps
    for t_idx in range(LUT_TIMESTEPS):
        t_sub = rough_temps[:, t_idx] # (N_sub,)
        t_smooth = smooth_temps[t_idx] # scalar
        
        # Sun Basis for Azimuth
        sun_vec_norm = sun_vectors[t_idx] / np.linalg.norm(sun_vectors[t_idx])
        s_dot_n = np.dot(sun_vec_norm, facet_normal)
        sun_proj = sun_vec_norm - s_dot_n * facet_normal
        if np.linalg.norm(sun_proj) > 1e-6:
            basis_x = sun_proj / np.linalg.norm(sun_proj) 
        else:
            up = np.array([0, 0, 1]) if np.abs(np.dot([0,0,1], facet_normal)) < 0.9 else np.array([1, 0, 0])
            basis_x = np.cross(facet_normal, up)
            basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(facet_normal, basis_x)
        
        # Loop Viewing Angles
        for i_e, emi_deg in enumerate(EMISSION_ANGLES):
            emi_rad = np.radians(emi_deg)
            sin_e = np.sin(emi_rad)
            cos_e = np.cos(emi_rad)

            for i_a, azi_deg in enumerate(AZIMUTH_ANGLES):
                azi_rad = np.radians(azi_deg)
                
                # View Vector
                v_local_x = sin_e * np.cos(azi_rad)
                v_local_y = sin_e * np.sin(azi_rad)
                v_local_z = cos_e
                view_vec = v_local_x * basis_x + v_local_y * basis_y + v_local_z * facet_normal
                view_vec /= np.linalg.norm(view_vec)
                
                # Transform to crater-local frame for visibility test
                d_local = world_to_local.dot(view_vec)
                
                # Check if view enters crater opening (d_local[2] > 0 = faces aperture)
                if d_local[2] <= 0.0:
                    visible_mask = np.zeros(len(normals), dtype=np.bool_)
                else:
                    # Fast numba visibility: Möller–Trumbore shadow test, zero allocations
                    visible_mask = compute_visibility_mask(d_local, normals, centers, triangles)
                
                # cos(theta_i) for each subfacet vs view direction in local frame
                cos_i_all = normals.dot(d_local)
                
                # Loop Wavelengths
                for i_w, wave in enumerate(WAVELENGTHS_MICRONS):
                    # 1. Compute spectral intensity (power per steradian per micron)
                    # CORRECT formula for outgoing radiance from a rough crater:
                    #   I_rough = Σ_visible B(λ, T_i) * A_i * cos(θ_i)
                    # where cos(θ_i) is angle between subfacet normal and view direction.
                    rad_sub = planck_function(wave, t_sub)
                    cos_i_vis = np.maximum(cos_i_all[visible_mask], 0.0)
                    rough_intensity = np.sum(rad_sub[visible_mask] * areas[visible_mask] * cos_i_vis)
                    
                    # Smooth intensity: I_smooth = B(λ, T_smooth) * A_aperture * cos(e)
                    if t_smooth < 5.0:
                        smooth_intensity = 0.0
                    else:
                        rad_smooth_val = planck_function(wave, t_smooth)
                        smooth_intensity = rad_smooth_val * aperture_area * cos_e

                    # Store radiance ratio: R = I_rough / I_smooth = L_rough / L_smooth
                    if cos_e < 1e-12:
                        # Grazing limb (e = 90°): the smooth reference projects to
                        # zero area, so R = 0/0 is undefined (not 1).
                        # Storing 1.0 here (the old way) put a spurious step in
                        # the last row — R falls to ~0.45 by e = 87.5° and then jumped
                        # back to exactly 1.  NaN makes the interpolator refuse the
                        # cell instead; queries are held below EMI_CAP in lut.py so it
                        # is never reached in practice.
                        result_grid[t_idx, i_w, i_e, i_a] = np.nan
                    elif smooth_intensity < 1e-15:
                        result_grid[t_idx, i_w, i_e, i_a] = 1.0  # cold-night fallback
                    else:
                        ratio = rough_intensity / smooth_intensity
                        # Clamp to physical range to avoid interpolation artifacts
                        # at extreme viewing geometries (e~89°) or Wien-tail conditions
                        result_grid[t_idx, i_w, i_e, i_a] = np.clip(ratio, 0.0, 50.0)

    # --- PER-TIMESTEP BOLOMETRIC CLOSURE DIAGNOSTIC alpha(t) ---
    # alpha(t) is the scalar that would force the Planck-weighted bolometric
    # angular mean of R to 1.0 at each timestep.  Historically it was applied to
    # the LUT; that was masking a missing-self-heating bug (alpha ~ 2.1).  With
    # self-heating, intra-crater scattering, and warm-start convergence fixed,
    # energy closure is inherent in the simulation, and forcing instantaneous
    # closure against the smooth reference would erase genuine diurnal beaming
    # (the cavity is really warmer than the smooth facet at night and lags it in
    # the morning).  alpha is therefore recorded as a diagnostic only; set
    # LUT_APPLY_NORM=1 to restore the old behaviour for comparison runs.
    #
    # Closure metric: ∑_λ w_λ · ∫ R(λ,e,φ) cos(e) sin(e) de dφ vs π · ∑_λ w_λ
    #             w_λ = B(λ, T_smooth) · Δλ   (Planck-weighted bandwidth)
    wave_arr = np.array(WAVELENGTHS_MICRONS)
    dwave = np.gradient(wave_arr)

    norm_factors_t = np.zeros(LUT_TIMESTEPS)
    for t_idx in range(LUT_TIMESTEPS):
        t_sm = smooth_temps[t_idx]

        # Planck weights; fall back to bandwidth-only for very cold timesteps
        # (nightside) where the Wien exponential underflows.
        if t_sm > 50.0:
            planck_w = planck_function(wave_arr, t_sm) * dwave
        else:
            planck_w = dwave.copy()
        planck_total = np.sum(planck_w)

        # Bolometric-weighted angular integral across all wavelengths
        total_weighted = sum(
            planck_w[i_w] * np.sum(result_grid[t_idx, i_w, :, :] * cse_weights)
            for i_w in range(len(WAVELENGTHS_MICRONS))
        )

        expected = planck_total * ref_integral  # π · ∑ w_λ
        alpha = expected / total_weighted if total_weighted > 1e-15 else 1.0
        if os.environ.get('LUT_APPLY_NORM', '0') == '1':
            result_grid[t_idx, :, :, :] *= alpha
        norm_factors_t[t_idx] = alpha

    # Diagnostic: average normalization factor across all timesteps
    norm_factor = float(np.mean(norm_factors_t))
                
    return result_grid, norm_factor

def save_theta_hdf5(theta_val, opening_angle_idx, lut_slice, factors_slice, 
                    lat_status, elapsed_per_lat, output_dir):
    """
    Save a single theta value's LUT data to its own HDF5 file.
    
    Args:
        theta_val: The theta value
        opening_angle_idx: Index of the opening angle used
        lut_slice: Array of shape (n_lat, n_time, n_wave, n_emi, n_azi)
        factors_slice: Array of shape (n_lat,) - normalization factors
        lat_status: dict mapping lat_index -> 'ok' or error string
        elapsed_per_lat: dict mapping lat_index -> time in seconds
        output_dir: Directory to save into
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{OUTPUT_PREFIX}_theta_{theta_val:.6f}.h5"
    fpath = os.path.join(output_dir, fname)
    
    with h5py.File(fpath, 'w') as f:
        # Main LUT data - shape: (n_lat, n_time, n_wave, n_emi, n_azi)
        f.create_dataset("lut", data=lut_slice, compression="gzip", compression_opts=1)
        f.create_dataset("normalization_factors", data=factors_slice)
        
        # Axes
        f.create_dataset("theta", data=np.array([theta_val]))
        f.create_dataset("opening_angle", data=np.array(OPENING_ANGLES))
        f.create_dataset("wavelength", data=np.array(WAVELENGTHS_MICRONS))
        f.create_dataset("latitude", data=LATITUDE_VALUES)
        f.create_dataset("emission", data=EMISSION_ANGLES)
        f.create_dataset("azimuth", data=AZIMUTH_ANGLES)
        f.create_dataset("sun_phase_steps", data=LUT_TIMESTEPS)
        
        # Metadata
        f.attrs['generator_version'] = 'v2_per_theta'
        f.attrs['theta'] = theta_val
        f.attrs['opening_angle'] = OPENING_ANGLES[opening_angle_idx]
        f.attrs['crater_subfacets'] = CRATER_SUBFACETS
        f.attrs['sim_timesteps'] = SIM_TIMESTEPS
        f.attrs['lut_timesteps'] = LUT_TIMESTEPS
        f.attrs['view_factor_rays'] = VIEW_FACTOR_RAYS
        f.attrs['reference_albedo'] = REFERENCE_ALBEDO
        f.attrs['reference_emissivity'] = REFERENCE_EMISSIVITY
        f.attrs['rotation_period_hours'] = ROTATION_PERIOD_HOURS
        f.attrs['solar_distance_au'] = SOLAR_DISTANCE_AU
        f.attrs['n_successful_lats'] = sum(1 for s in lat_status.values() if s == 'ok')
        f.attrs['n_failed_lats'] = sum(1 for s in lat_status.values() if s != 'ok')
        
        # Per-latitude diagnostics
        status_strs = [lat_status.get(i, 'not_run') for i in range(len(LATITUDE_VALUES))]
        f.create_dataset("lat_status", data=np.array(status_strs, dtype='S64'))
        timing_arr = np.array([elapsed_per_lat.get(i, 0.0) for i in range(len(LATITUDE_VALUES))])
        f.create_dataset("lat_elapsed_seconds", data=timing_arr)
    
    size_mb = os.path.getsize(fpath) / (1024**2)
    return fpath, size_mb


def generate_diagnostics(output_dir, all_theta_results):
    """
    Generate summary diagnostic plots and a status log after the full run.
    
    Args:
        output_dir: Directory containing the per-theta HDF5 files
        all_theta_results: list of dicts with keys:
            'theta', 'elapsed', 'n_ok', 'n_fail', 'size_mb', 'mean_norm_factor'
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    diag_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    # --- 1. Status log ---
    log_path = os.path.join(diag_dir, 'run_summary.txt')
    with open(log_path, 'w') as log:
        log.write("TEMPEST LUT Generation Summary\n")
        log.write("=" * 60 + "\n")
        log.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"N_JOBS: {N_JOBS}\n")
        log.write(f"CRATER_SUBFACETS: {CRATER_SUBFACETS}\n")
        log.write(f"SIM_TIMESTEPS: {SIM_TIMESTEPS} -> LUT_TIMESTEPS: {LUT_TIMESTEPS}\n")
        log.write(f"Grid: {len(LATITUDE_VALUES)} lat x {LUT_TIMESTEPS} time x {len(WAVELENGTHS_MICRONS)} wave x {len(EMISSION_ANGLES)} emi x {len(AZIMUTH_ANGLES)} azi\n")
        n_cells = len(LATITUDE_VALUES) * LUT_TIMESTEPS * len(WAVELENGTHS_MICRONS) * len(EMISSION_ANGLES) * len(AZIMUTH_ANGLES)
        log.write(f"Cells per theta: {n_cells:,} ({n_cells*4/1024**2:.1f} MB uncompressed)\n")
        log.write(f"Theta values attempted: {len(all_theta_results)}\n\n")
        
        total_ok = sum(r['n_ok'] for r in all_theta_results)
        total_fail = sum(r['n_fail'] for r in all_theta_results)
        log.write(f"Total latitude cases: {total_ok + total_fail}  (OK: {total_ok}, Failed: {total_fail})\n\n")
        
        log.write(f"{'Theta':>12} {'Time (min)':>10} {'OK':>4} {'Fail':>4} {'Size MB':>8} {'Mean Norm':>10}\n")
        log.write("-" * 55 + "\n")
        for r in all_theta_results:
            log.write(f"{r['theta']:>12.6f} {r['elapsed']/60:>10.1f} {r['n_ok']:>4} {r['n_fail']:>4} {r['size_mb']:>8.1f} {r['mean_norm_factor']:>10.4f}\n")
    
    print(f"  Saved run summary to {log_path}")
    
    if len(all_theta_results) < 2:
        return
    
    # --- 2. Timing plot ---
    thetas = [r['theta'] for r in all_theta_results]
    times_min = [r['elapsed']/60 for r in all_theta_results]
    n_oks = [r['n_ok'] for r in all_theta_results]
    n_fails = [r['n_fail'] for r in all_theta_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time per theta
    ax = axes[0, 0]
    ax.bar(range(len(thetas)), times_min, color='steelblue')
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:.3f}" for t in thetas], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Wall-clock time per theta')
    
    # Success/failure per theta
    ax = axes[0, 1]
    ax.bar(range(len(thetas)), n_oks, color='green', label='OK')
    ax.bar(range(len(thetas)), n_fails, bottom=n_oks, color='red', label='Failed')
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:.3f}" for t in thetas], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Latitude cases')
    ax.set_title('Success/failure per theta')
    ax.legend()
    
    # Normalization factors
    ax = axes[1, 0]
    norms = [r['mean_norm_factor'] for r in all_theta_results]
    ax.plot(thetas, norms, 'o-', color='darkorange')
    ax.set_xscale('log')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Mean normalization factor')
    ax.set_title('Energy conservation (1.0 = perfect)')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    
    # File sizes
    ax = axes[1, 1]
    sizes = [r['size_mb'] for r in all_theta_results]
    ax.bar(range(len(thetas)), sizes, color='mediumpurple')
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:.3f}" for t in thetas], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('File size (MB)')
    ax.set_title('HDF5 file size per theta')
    ax.axhline(500, color='red', ls='--', alpha=0.5, label='500 MB limit')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(diag_dir, 'run_summary.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved diagnostic plots to {plot_path}")
    
    # --- 3. Per-theta sample slices (if files exist) ---
    for r in all_theta_results:
        if r['n_ok'] == 0:
            continue
        try:
            fpath = r.get('fpath', '')
            if not fpath or not os.path.exists(fpath):
                continue
            with h5py.File(fpath, 'r') as f:
                lut = f['lut'][:]
            
            # Plot: mean ratio vs emission angle at a few latitudes, azimuth=0 (opposition)
            fig, axes2 = plt.subplots(1, 3, figsize=(15, 5))
            theta_val = r['theta']
            
            # Pick 4 latitudes to plot
            lat_indices = [0, len(LATITUDE_VALUES)//4, len(LATITUDE_VALUES)//2, -1]
            wave_idx = len(WAVELENGTHS_MICRONS) // 2  # mid wavelength
            
            # Emission angle dependence at azimuth=0
            ax = axes2[0]
            for li in lat_indices:
                # Average over all timesteps, azimuth=0
                profile = np.nanmean(lut[li, :, wave_idx, :, 0], axis=0)  # (emi,)
                ax.plot(EMISSION_ANGLES, profile, label=f'lat={LATITUDE_VALUES[li]:.0f}°')
            ax.set_xlabel('Emission angle (°)')
            ax.set_ylabel('Radiance ratio')
            ax.set_title(f'θ={theta_val:.4f}, λ={WAVELENGTHS_MICRONS[wave_idx]}μm, azi=0°')
            ax.legend(fontsize=7)
            ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
            
            # Azimuth dependence at emission=45°
            ax = axes2[1]
            emi_idx = len(EMISSION_ANGLES) // 2
            for li in lat_indices:
                profile = np.nanmean(lut[li, :, wave_idx, emi_idx, :], axis=0)  # (azi,)
                ax.plot(AZIMUTH_ANGLES, profile, label=f'lat={LATITUDE_VALUES[li]:.0f}°')
            ax.set_xlabel('Azimuth angle (°)')
            ax.set_ylabel('Radiance ratio')
            ax.set_title(f'θ={theta_val:.4f}, λ={WAVELENGTHS_MICRONS[wave_idx]}μm, emi={EMISSION_ANGLES[emi_idx]:.0f}°')
            ax.legend(fontsize=7)
            ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
            
            # Wavelength dependence at nadir (emi=0)
            ax = axes2[2]
            for li in lat_indices:
                profile = np.nanmean(lut[li, :, :, 0, 0], axis=0)  # (wave,)
                ax.plot(WAVELENGTHS_MICRONS, profile, 'o-', label=f'lat={LATITUDE_VALUES[li]:.0f}°')
            ax.set_xlabel('Wavelength (μm)')
            ax.set_ylabel('Radiance ratio')
            ax.set_title(f'θ={theta_val:.4f}, nadir view')
            ax.set_xscale('log')
            ax.legend(fontsize=7)
            ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
            
            plt.suptitle(f'Theta = {theta_val:.4f}', fontsize=14)
            plt.tight_layout()
            slice_path = os.path.join(diag_dir, f'theta_{theta_val:.6f}_slices.png')
            plt.savefig(slice_path, dpi=120)
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate slice plot for theta={r['theta']}: {e}")


def main():
    print("="*80)
    print("TEMPEST Roughness LUT Generator (Per-Theta Saving)")
    print("="*80)
    run_start_time = time.time()
    
    # --- File size estimate ---
    n_cells_per_theta = (len(LATITUDE_VALUES) * LUT_TIMESTEPS * len(WAVELENGTHS_MICRONS) 
                         * len(EMISSION_ANGLES) * len(AZIMUTH_ANGLES))
    est_size_mb = n_cells_per_theta * 4 / (1024**2)
    print(f"\nEstimated file size per theta: {est_size_mb:.1f} MB (uncompressed float32)")
    if est_size_mb > 500:
        print(f"  WARNING: Exceeds 500 MB target! Consider reducing resolution.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create Reference Config
    config = ReferenceConfig()
    
    # Generate and Plot Wireframe
    print("\nGenerating geometry preview...")
    dummy_normal = np.array([0.0, 0.0, 1.0])
    dummy_vs = [np.array([-1,-1,0]), np.array([1,-1,0]), np.array([0,1,0])]
    dummy_facet = Facet(dummy_normal, dummy_vs, 1, 1, 1, False)
    
    config.kernel_profile_angle_degrees = OPENING_ANGLES[0]
    config.apply_kernel_based_roughness = True
    dummy_facet.generate_spherical_depression(config, Simulation(config))
    plot_wireframe(Facet._canonical_subfacet_mesh, 
                   filename=os.path.join(OUTPUT_DIR, "diagnostics", "crater_wireframe.png"))
    
    # Pre-calculate View Factors (done once, cached to disk and reused for ALL theta/latitude combos)
    print("\nPre-calculating view factors (one-off, cached to disk)...")
    mesh = Facet._canonical_subfacet_mesh
    n_facets = len(mesh)
    actual_cores = config.validate_jobs()
    print(f"  Crater mesh:   {n_facets} subfacets")
    print(f"  VF rays:       {VIEW_FACTOR_RAYS:,} per facet")
    print(f"  Parallel jobs: {actual_cores} cores")
    print(f"  This may take 5-30 minutes on first run (results are cached).")
    
    shape_model = []
    rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
    for entry in mesh:
        canonical_n = np.array(entry['normal'])
        canonical_v = np.array(entry['vertices'])
        rotated_n = np.dot(rotation_to_equator, canonical_n)
        rotated_v = np.array([np.dot(rotation_to_equator, v) for v in canonical_v])
        new_f = Facet(rotated_n, rotated_v, 1, 1, 1, False)
        shape_model.append(new_f)
        
    thermal_data = ThermalData(n_facets, 1, 1, 1, False)
    all_indices = np.arange(n_facets)
    visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
    thermal_data.set_visible_facets(visible_facets_list)
    
    vf_start = time.time()
    # Temporarily enable verbose output for the one-off VF computation
    config.silent_mode = False
    print(f"\n  [1/3] Computing visible facet pairs...")
    # (calculate_all_view_factors handles visible facets internally + caching)
    cached_all_view_factors = calculate_all_view_factors(shape_model, thermal_data, config, VIEW_FACTOR_RAYS)
    vf_mid = time.time()
    print(f"  [1/3] Done ({vf_mid - vf_start:.1f}s)")

    # calculate_thermal_view_factors reads thermal_data.secondary_radiation_view_factors;
    # without this the thermal VFs are all empty and the crater solve runs with NO
    # self-heating (a hemispherical cavity then loses exactly half its emitted power
    # out of the energy budget -> the alpha~2 normalisation factors).
    thermal_data.set_secondary_radiation_view_factors(cached_all_view_factors)

    print(f"  [2/3] Computing thermal view factors...")
    cached_thermal_view_factors = calculate_thermal_view_factors(shape_model, thermal_data, config)
    vf_elapsed = time.time() - vf_start
    print(f"  [2/3] Done ({time.time() - vf_mid:.1f}s)")
    print(f"  [3/3] View factor cache ready. Total: {vf_elapsed:.1f}s ({vf_elapsed/60:.1f} min)")
    # Restore silent mode for parallel workers
    config.silent_mode = True

    # --- Build pre-computed geometry dict (passed to all workers) ---
    # This eliminates redundant mesh generation and view factor loading per case.
    # Workers receive numpy arrays (efficiently serialized by joblib) instead of
    # regenerating 10k-facet meshes and racing to compute view factors.
    canonical_normals = np.array([entry['normal'] for entry in mesh], dtype=np.float64)
    canonical_vertices = np.array([entry['vertices'] for entry in mesh], dtype=np.float64)
    canonical_areas = np.array([entry['area'] for entry in mesh], dtype=np.float64)
    if 'center' in mesh[0]:
        canonical_centers = np.array([entry['center'] for entry in mesh], dtype=np.float64)
    else:
        canonical_centers = np.array([np.mean(entry['vertices'], axis=0) for entry in mesh], dtype=np.float64)
    
    rotated_normals = np.array([np.dot(rotation_to_equator, entry['normal']) for entry in mesh], dtype=np.float64)
    rotated_vertices = np.array([[np.dot(rotation_to_equator, v) for v in entry['vertices']] for entry in mesh], dtype=np.float64)
    
    # Convert view factors to serializable numpy arrays so joblib can pass them
    # to workers without any disk I/O. This eliminates ~3500 NFS reads per theta.
    cached_vf_arrays = [np.array(vf, dtype=np.float64) for vf in cached_all_view_factors]
    cached_tvf_arrays = []
    for vf in cached_thermal_view_factors:
        arr = np.array(vf, dtype=np.float64)
        if arr.size == 0:
            cached_tvf_arrays.append(np.array([], dtype=np.float64))
        else:
            cached_tvf_arrays.append(arr.flatten() if arr.ndim > 1 else arr)

    precomputed = {
        'canonical_normals': canonical_normals,
        'canonical_vertices': canonical_vertices,
        'canonical_areas': canonical_areas,
        'canonical_centers': canonical_centers,
        'rotated_normals': rotated_normals,
        'rotated_vertices': rotated_vertices,
        'rotation_to_equator': rotation_to_equator,
        'world_to_local': rotation_to_equator.T,
        'aperture_normal': np.array([1.0, 0.0, 0.0]),  # crater opening faces +X after rotation
        'aperture_area': 1.0,  # normalized by mesh generation
        'n_facets': n_facets,
        # Pre-loaded view factors — workers use these directly, zero disk I/O
        'all_view_factors': cached_vf_arrays,
        'thermal_view_factors': cached_tvf_arrays,
    }
    
    # Warm up numba JIT for compute_visibility_mask (one-time compilation cost)
    print("  Compiling visibility kernel (one-time)...")
    _warmup_dir = canonical_normals[0].copy()
    _warmup_dir /= np.linalg.norm(_warmup_dir)
    _ = compute_visibility_mask(_warmup_dir, canonical_normals, canonical_centers, canonical_vertices)
    print("  Ready.")

    # --- Print configuration summary ---
    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Theta values:    {len(THETA_VALUES)} ({THETA_VALUES[0]:.4f} to {THETA_VALUES[-1]:.4f})")
    print(f"  Opening angles:  {OPENING_ANGLES}")
    print(f"  Latitudes:       {len(LATITUDE_VALUES)} (0° to {LATITUDE_VALUES[-1]:.0f}°, {LATITUDE_VALUES[1]-LATITUDE_VALUES[0]:.1f}° step)")
    print(f"  Wavelengths:     {len(WAVELENGTHS_MICRONS)} bands ({WAVELENGTHS_MICRONS[0]}-{WAVELENGTHS_MICRONS[-1]} μm)")
    print(f"  Emission angles: {len(EMISSION_ANGLES)} (0° to {EMISSION_ANGLES[-1]:.0f}°)")
    print(f"  Azimuth angles:  {len(AZIMUTH_ANGLES)} (0° to {AZIMUTH_ANGLES[-1]:.0f}°)")
    print(f"  Time steps:      {LUT_TIMESTEPS} output ({SIM_TIMESTEPS} internal)")
    print(f"  Crater facets:   {n_facets}")
    print(f"  Parallel jobs:   {N_JOBS}")
    per_theta_shape = f"{len(LATITUDE_VALUES)}×{LUT_TIMESTEPS}×{len(WAVELENGTHS_MICRONS)}×{len(EMISSION_ANGLES)}×{len(AZIMUTH_ANGLES)}"
    print(f"  Per-theta shape: {per_theta_shape}")
    print(f"  Est. size/theta: {est_size_mb:.1f} MB")
    print(f"  Total thetas:    {len(THETA_VALUES)} → est. total: {est_size_mb * len(THETA_VALUES):.0f} MB")
    print(f"  Output dir:      {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # MAIN LOOP: Process one theta value at a time, save immediately
    # =========================================================================
    all_theta_results = []
    
    for i_th, theta in enumerate(THETA_VALUES):
        theta_start = time.time()
        print(f"\n{'─'*60}")
        print(f"Theta {i_th+1}/{len(THETA_VALUES)}: {theta:.6f}")
        print(f"{'─'*60}")
        
        # Check if this theta was already completed (resume support)
        existing_fname = f"{OUTPUT_PREFIX}_theta_{theta:.6f}.h5"
        existing_fpath = os.path.join(OUTPUT_DIR, existing_fname)
        if os.path.exists(existing_fpath):
            try:
                with h5py.File(existing_fpath, 'r') as f:
                    n_ok = int(f.attrs.get('n_successful_lats', 0))
                    n_fail = int(f.attrs.get('n_failed_lats', 0))
                if n_ok + n_fail == len(LATITUDE_VALUES) and n_fail == 0:
                    print(f"  SKIPPING: Already completed ({n_ok} lats OK). Delete file to re-run.")
                    size_mb = os.path.getsize(existing_fpath) / (1024**2)
                    all_theta_results.append({
                        'theta': theta, 'elapsed': 0, 'n_ok': n_ok, 'n_fail': n_fail,
                        'size_mb': size_mb, 'mean_norm_factor': 1.0, 'fpath': existing_fpath
                    })
                    continue
                else:
                    print(f"  Found partial result ({n_ok} ok, {n_fail} fail). Re-running...")
            except Exception:
                print(f"  Found corrupt file. Re-running...")
        
        # Run all latitudes in parallel
        results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(process_single_case)(theta, OPENING_ANGLES[0], lat, config, precomputed)
            for lat in LATITUDE_VALUES
        )
        
        # Assemble this theta's tensor: (n_lat, n_time, n_wave, n_emi, n_azi)
        lut_slice = np.full((len(LATITUDE_VALUES), LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS),
                             len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), np.nan, dtype=np.float32)
        factors_slice = np.full(len(LATITUDE_VALUES), np.nan, dtype=np.float32)
        lat_status = {}
        elapsed_per_lat = {}
        
        for i_lat, (lat, result) in enumerate(zip(LATITUDE_VALUES, results)):
            if result is None:
                lat_status[i_lat] = 'returned_none'
                continue
            grid, factor = result
            if np.isnan(factor):
                lat_status[i_lat] = 'nan_factor'
                print(f"  FAILED: lat={lat:.1f}° (returned NaN)")
            else:
                lut_slice[i_lat] = grid
                factors_slice[i_lat] = factor
                lat_status[i_lat] = 'ok'
        
        n_ok = sum(1 for s in lat_status.values() if s == 'ok')
        n_fail = len(LATITUDE_VALUES) - n_ok
        theta_elapsed = time.time() - theta_start
        
        # Save immediately
        fpath, size_mb = save_theta_hdf5(
            theta, 0, lut_slice, factors_slice, lat_status, elapsed_per_lat, OUTPUT_DIR
        )
        
        mean_nf = float(np.nanmean(factors_slice[~np.isnan(factors_slice)])) if n_ok > 0 else np.nan
        print(f"  Saved: {fpath} ({size_mb:.1f} MB)")
        print(f"  Results: {n_ok}/{len(LATITUDE_VALUES)} OK, {n_fail} failed")
        print(f"  Mean norm factor: {mean_nf:.4f}")
        print(f"  Time: {theta_elapsed/60:.1f} min")
        
        all_theta_results.append({
            'theta': theta, 'elapsed': theta_elapsed, 'n_ok': n_ok, 'n_fail': n_fail,
            'size_mb': size_mb, 'mean_norm_factor': mean_nf, 'fpath': fpath
        })
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    total_elapsed = time.time() - run_start_time
    print(f"\n{'='*80}")
    print(f"LUT Generation Complete!")
    print(f"{'='*80}")
    print(f"  Total time:     {total_elapsed/3600:.1f} hours ({total_elapsed/60:.1f} min)")
    print(f"  Theta completed: {len(all_theta_results)}/{len(THETA_VALUES)}")
    total_ok = sum(r['n_ok'] for r in all_theta_results)
    total_fail = sum(r['n_fail'] for r in all_theta_results)
    print(f"  Total cases:    {total_ok + total_fail} (OK: {total_ok}, Failed: {total_fail})")
    total_size = sum(r['size_mb'] for r in all_theta_results)
    print(f"  Total disk:     {total_size:.0f} MB")
    print(f"  Output dir:     {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\nGenerating diagnostics...")
    generate_diagnostics(OUTPUT_DIR, all_theta_results)
    print(f"\nDone!")
    print("="*80)

if __name__ == "__main__":
    main()
