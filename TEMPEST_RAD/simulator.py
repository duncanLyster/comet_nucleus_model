"""
TEMPEST_RAD: Radiance Observation Simulator.

This module generates synthetic disk-resolved observations (images/spectra) from
TEMPEST thermal outputs using a High-Fidelity Roughness Lookup Table.

Usage Examples:

1. Spectral Mode (Single Wavelength):
   python TEMPEST_RAD/simulator.py --config private/data/config/moon/moon_config.yaml \
                                   --temps output/run_latest/ \
                                   --time 12.0 --wavelength 10.0 --phase 30.0 \
                                   --roughness_rms_angle 20.0

2. Bolometric Mode (Integrated Brightness Temperature + Wireframe):
   python TEMPEST_RAD/simulator.py --config private/data/config/moon/moon_config.yaml \
                                   --temps output/run_latest/ \
                                   --mode bolometric --wireframe \
                                   --time 6.0 --phase 45.0

Arguments:
  --config      Path to the TEMPEST config.yaml used for the run.
  --temps       Path to the temperature output (directory or .csv/.npy file).
  --time        Simulation time in hours (modulo rotation period).
  --wavelength  Observation wavelength in microns (Spectral mode only).
  --mode        'spectral' (default) or 'bolometric' (integrated brightness temp).
  --wireframe   Overlay shape model wireframe on the plot.
  --phase_angle Phase angle in degrees (default: 30.0).

Workflow:
1. Load Smooth Temperature Map (from TEMPEST).
2. Load Roughness LUT (High-Fidelity Radiance Ratios).
3. Set Observation Geometry (Observer/Sun vectors).
4. For every visible facet:
   - Calculate B_smooth(lambda, T).
   - Look up Radiance Ratio R(lat, phase, emi, azi).
   - L_obs = (1 - f) * B_smooth + f * (B_smooth * R).
5. Project to Camera Plane.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button
from pathlib import Path
import h5py

# Ensure src directory is in the Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.utilities.config import Config
from src.utilities.utils import rotate_vector
from TEMPEST_RAD.lut import RoughnessLUT
from stl import mesh as stl_mesh_module

class FacetData:
    """Minimal container for Facet data needed for Rad calculations."""
    def __init__(self, normal, center, area):
        self.normal = normal
        self.center = center
        self.area = area

def load_shape_model(filename):
    """Load STL and return simplified facet data."""
    mesh = stl_mesh_module.Mesh.from_file(filename)
    facets = []
    # Calculate centers
    centers = np.mean(mesh.vectors, axis=1)
    # Calculate areas
    v0 = mesh.vectors[:, 0, :]
    v1 = mesh.vectors[:, 1, :]
    v2 = mesh.vectors[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    for i in range(len(mesh.normals)):
        facets.append(FacetData(mesh.normals[i], centers[i], areas[i]))
    return facets, mesh

def planck_function(wavelength_um, temp_k):
    """Spectral Radiance B(lambda, T)."""
    c1 = 1.191042e8
    c2 = 1.4387752e4
    t_safe = np.maximum(temp_k, 1e-5)
    return c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))

def compute_geometry(facets, sun_vec, obs_vec, rot_axis):
    """
    Compute per-facet geometry parameters for LUT.
    
    Returns:
        latitudes: (N,) relative to rotation axis
        sun_phases: (N,) GLOBAL body rotation phase (0-360°, same for all facets)
                    This represents the rotation angle of the body, NOT per-facet local hour angles.
                    For compatibility with LUT which was generated with uniform rotation phases.
        emissions: (N,) emission angle
        azimuths: (N,) relative azimuth
    """
    n_facets = len(facets)
    normals = np.array([f.normal for f in facets])
    centers = np.array([f.center for f in facets])
    
    # Normalize all vectors (STL normals may not be unit vectors)
    sun_vec = sun_vec / np.linalg.norm(sun_vec)
    obs_vec = obs_vec / np.linalg.norm(obs_vec)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    normal_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_mags = np.maximum(normal_mags, 1e-12)  # Avoid division by zero
    normals = normals / normal_mags
    
    # 1. Latitude (from facet CENTER POSITION, not normal direction)
    # For LUT: geographic latitude = where on body surface the facet is located
    center_norms = np.linalg.norm(centers, axis=1)
    centers_unit = centers / center_norms[:, np.newaxis]
    sin_lats = np.dot(centers_unit, rot_axis)
    latitudes = np.abs(np.degrees(np.arcsin(np.clip(sin_lats, -1.0, 1.0))))
    
    # 2. Emission Angle
    # cos(e) = N dot Obs
    cos_emi = np.dot(normals, obs_vec)
    emissions = np.degrees(np.arccos(np.clip(cos_emi, 0.0, 1.0)))
    
    # 3. Sun Phase (Global Body Rotation Phase)
    # FIXED: Return a single global rotation phase for all facets, not per-facet local hour angles.
    # The LUT was generated with uniform body rotation phases (0-360°).
    #
    # To compute the global phase, we use a reference point on the equator (at +X initially)
    # and measure how far it has rotated toward the sun's equatorial projection.
    #
    # Reference direction: Start at +X axis (0° phase)
    # Sun's equatorial projection: Project sun vector onto equatorial plane
    # Phase = angle from +X to sun projection, measured around rotation axis
    
    # Project sun onto equatorial plane
    s_proj = sun_vec - np.dot(sun_vec, rot_axis) * rot_axis
    s_norm = np.linalg.norm(s_proj)
    
    if s_norm > 1e-6:
        s_proj = s_proj / s_norm
        
        # Reference direction in equatorial plane (typically +X)
        # For a body rotating around +Z, the reference is +X axis
        ref_dir = np.array([1.0, 0.0, 0.0])
        ref_proj = ref_dir - np.dot(ref_dir, rot_axis) * rot_axis
        ref_norm = np.linalg.norm(ref_proj)
        if ref_norm > 1e-6:
            ref_proj = ref_proj / ref_norm
        else:
            # Rotation axis is along X, use Y as reference
            ref_proj = np.array([0.0, 1.0, 0.0])
        
        # Angle between reference and sun projection
        cos_phase = np.dot(ref_proj, s_proj)
        phase_rad = np.arccos(np.clip(cos_phase, -1.0, 1.0))
        
        # Determine sign using cross product
        cross_val = np.cross(ref_proj, s_proj)
        sign = np.dot(cross_val, rot_axis)
        
        if sign < 0:
            phase_rad = 2 * np.pi - phase_rad
        
        # This is the global rotation phase (same for all facets)
        global_phase = np.degrees(phase_rad)
    else:
        # Sun along rotation axis (polar incidence) - phase is ill-defined
        global_phase = 0.0
    
    # --- LUT phase query mode -------------------------------------------------
    # The LUT time axis is the CRATER'S LOCAL SOLAR TIME with NOON AT PHASE 0
    # (verified 2026-07-05 against the generator's sun vectors: at t=0 the sun
    # is along the aperture normal, sunset at 90, midnight at 180, sunrise at
    # 270; the smooth reference peaks at ~17 deg = noon + thermal lag).  Each
    # facet must be queried at its OWN local hour angle: the crater's internal
    # shadow/temperature pattern depends on where the facet is in its day, not
    # on any global clock.  The pre-2026-07-05 mapping added a spurious 180 deg
    # (mod(180+h)) which read the MIDNIGHT cavity state for noon facets - the
    # night-warm cavity (angle-flat R~1.45) masqueraded as beaming and erased
    # the dayside anti-sunward dimming (R~0.5-0.75) entirely, so the modelled
    # beaming phase function could never dip below 1 at any phase angle.
    # Modes (env LUT_PHASE_MODE): 'local' (default, correct) | 'localneg'
    # (hour-angle sign A/B) | 'legacy180' (the buggy 180-offset mapping, kept
    # for comparison) | 'global' (frozen global clock: one phase for every
    # facet, valid only for a single-facet/unresolved target).
    import os as _os
    _mode = _os.environ.get('LUT_PHASE_MODE', 'local')
    if _mode == 'global':
        sun_phases = np.full(n_facets, global_phase)
    else:
        # On a lumpy body the crater's diurnal cycle follows the facet NORMAL,
        # not the facet's position: a facet behaves like a horizontal element
        # at lat_eff = arcsin(n.axis) whose longitude is the normal's azimuth
        # (then cos(incidence) = cos(lat_eff)cos(dec)cos(H)+..., matching the
        # LUT's equatorial-crater/sun-declination convention).
        n_proj = normals - np.outer(np.dot(normals, rot_axis), rot_axis)
        nn = np.maximum(np.linalg.norm(n_proj, axis=1), 1e-12)
        n_proj = n_proj / nn[:, np.newaxis]
        ref_proj_l = np.array([1.0, 0.0, 0.0])
        ref_proj_l = ref_proj_l - np.dot(ref_proj_l, rot_axis) * rot_axis
        if np.linalg.norm(ref_proj_l) > 1e-6:
            ref_proj_l = ref_proj_l / np.linalg.norm(ref_proj_l)
        else:
            ref_proj_l = np.array([0.0, 1.0, 0.0])
        cosf = np.clip(n_proj @ ref_proj_l, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosf))
        sgn = np.einsum('ij,j->i', np.cross(np.broadcast_to(ref_proj_l, n_proj.shape), n_proj), rot_axis)
        facet_lon = np.where(sgn < 0, 360.0 - ang, ang)
        # hour angle: facet is at local noon when its normal's longitude ==
        # subsolar longitude (global_phase); tsteps() convention: sun longitude
        # increases with sim time, so afternoon = (lam - lon) > 0.
        if _mode == 'localneg':
            h = facet_lon - global_phase
        else:
            h = global_phase - facet_lon
        if _mode == 'legacy180':
            sun_phases = np.mod(180.0 + h, 360.0)
        else:
            sun_phases = np.mod(h, 360.0)
        # effective latitude from the NORMAL (diurnal amplitude of a tilted facet)
        latitudes = np.abs(np.degrees(np.arcsin(np.clip(np.dot(normals, rot_axis), -1.0, 1.0))))
        
    # 4. Relative Azimuth
    # Angle between Projected Sun and Projected Observer on Facet Plane
    # S_facet = S - (S.N)N
    # O_facet = O - (O.N)N
    s_dot_n = np.einsum('ij,ij->i', normals, np.full_like(normals, sun_vec))
    o_dot_n = np.einsum('ij,ij->i', normals, np.full_like(normals, obs_vec))
    
    s_facet = sun_vec - normals * s_dot_n[:, np.newaxis]
    o_facet = obs_vec - normals * o_dot_n[:, np.newaxis]
    
    # Normalize
    ns_facet = np.linalg.norm(s_facet, axis=1)
    no_facet = np.linalg.norm(o_facet, axis=1)
    
    valid_azi = (ns_facet > 1e-6) & (no_facet > 1e-6)
    azimuths = np.zeros(n_facets)
    
    if np.any(valid_azi):
        sf_norm = s_facet[valid_azi] / ns_facet[valid_azi, np.newaxis]
        of_norm = o_facet[valid_azi] / no_facet[valid_azi, np.newaxis]
        
        cos_azi = np.einsum('ij,ij->i', sf_norm, of_norm)
        azimuths[valid_azi] = np.degrees(np.arccos(np.clip(cos_azi, -1.0, 1.0)))
        
    return latitudes, sun_phases, emissions, azimuths

def rms_to_fraction(rms_angle_deg):
    """
    Convert RMS slope angle to fractional coverage f for 90-degree hemispherical craters.
    Heuristic: tan(theta_rms) approx f * tan(50 deg)
    This is a simplified mapping.
    """
    
    # Heuristic mapping
    # 0 deg -> 0.0
    # 10 deg -> ~0.15
    # 20 deg -> ~0.35
    # 30 deg -> ~0.60
    # >40 deg -> 1.0
    
    # Let's use a linear-ish map for now or the tan approximation
    # tan(rms) = f * tan(57) (Mean slope of hemisphere is 57 deg)
    
    tan_rms = np.tan(np.radians(rms_angle_deg))
    tan_hemi = np.tan(np.radians(57.0))
    
    f = tan_rms / tan_hemi
    return np.clip(f, 0.0, 1.0)

def calculate_theta(config):
    """
    Calculate dimensionless Thermal Parameter (Theta) from TEMPEST config.
    Theta = (Gamma * sqrt(omega)) / (epsilon * sigma * Tss^3)
    """
    # Try to get parameters from attributes or config_data dictionary
    def get_param(name, default=None):
        if hasattr(config, name):
            return getattr(config, name)
        elif hasattr(config, 'config_data') and name in config.config_data:
            return config.config_data[name]
        return default

    gamma = get_param('thermal_inertia')
    if gamma is None:
        raise ValueError("Config missing 'thermal_inertia'")
        
    period_hours = get_param('rotation_period_hours')
    if period_hours is None:
        raise ValueError("Config missing 'rotation_period_hours'")
        
    period_sec = period_hours * 3600.0
    omega = 2 * np.pi / period_sec
    
    epsilon = get_param('emissivity', 0.95)
    albedo = get_param('albedo', 0.1)
    
    # Solar Constant at distance
    solar_dist_au = get_param('solar_distance_au', 1.0)
    solar_dist_m = solar_dist_au * 1.496e11
    
    solar_lum = get_param('solar_luminosity', 3.828e26)
    
    solar_flux = solar_lum / (4 * np.pi * solar_dist_m**2)
    boltzmann = 5.670374419e-8
    
    tss = ((1 - albedo) * solar_flux / (epsilon * boltzmann)) ** 0.25
    
    theta = (gamma * np.sqrt(omega)) / (epsilon * boltzmann * tss**3)
    return theta

def main():
    parser = argparse.ArgumentParser(description="TEMPEST_RAD: Radiance Observation Simulator")
    parser.add_argument('--config', type=str, required=True, help="Path to TEMPEST config file")
    parser.add_argument('--temps', type=str, default="output/temperatures.csv", help="Path to temperature output")
    parser.add_argument('--lut', type=str, default="roughness_lut_spectral_v1.h5", help="Path to Roughness LUT")
    parser.add_argument('--time', type=float, default=0.0, help="Simulation time (hours) to observe")
    parser.add_argument('--wavelength', type=float, default=10.0, help="Observation wavelength (microns)")
    parser.add_argument('--mode', type=str, choices=['spectral', 'bolometric'], default='spectral', help="Simulation mode: 'spectral' (single wavelength) or 'bolometric' (integrated).")
    parser.add_argument('--roughness_rms_angle', type=float, default=28.0, help="RMS Slope Angle (degrees) to control roughness mixing.")
    parser.add_argument('--wireframe', action='store_true', help="Overlay wireframe on the plot.")
    # Observer Geometry (Simple Phase Angle mode)
    parser.add_argument('--phase_angle', type=float, default=30.0, help="Phase angle (degrees) for observation")
    parser.add_argument('--obs_dist', type=float, default=1000.0, help="Observer distance (km)")
    
    args = parser.parse_args()
    
    # 1. Load Config & Model
    print(f"Loading Configuration: {args.config}")
    config = Config(args.config)
    
    print(f"Loading Shape Model: {config.path_to_shape_model_file}")
    facets, mesh = load_shape_model(config.path_to_shape_model_file)
    n_facets = len(facets)
    
    # 2. Load Temperatures
    print(f"Loading Temperatures from {args.temps}")
    
    temps_all = None
    
    # CASE A: User provided a specific file
    if os.path.isfile(args.temps):
        try:
            # Check extension
            if args.temps.endswith('.npy'):
                temps_all = np.load(args.temps)
            elif args.temps.endswith('.csv'):
                # Try loading with numpy, skipping header if needed
                try:
                    temps_all = np.loadtxt(args.temps, delimiter=',')
                except ValueError:
                    # Likely has header
                    temps_all = np.loadtxt(args.temps, delimiter=',', skiprows=1)
            else:
                # Try generic load
                temps_all = np.loadtxt(args.temps)
        except Exception as e:
            print(f"Error loading file {args.temps}: {e}")

    # CASE B: User provided a directory (standard TEMPEST output folder)
    elif os.path.isdir(args.temps):
        # Look for temperatures.csv inside
        potential_path = os.path.join(args.temps, "temperatures.csv")
        if os.path.isfile(potential_path):
            print(f"Found standard output file: {potential_path}")
            try:
                try:
                    temps_all = np.loadtxt(potential_path, delimiter=',')
                except ValueError:
                    temps_all = np.loadtxt(potential_path, delimiter=',', skiprows=1)
            except Exception as e:
                print(f"Error loading {potential_path}: {e}")
        else:
            print(f"Could not find 'temperatures.csv' in {args.temps}")

    # Validation
    if temps_all is None:
        print("Could not load temperatures. Creating dummy data for test (300K).")
        temps_all = np.full((n_facets, 100), 300.0)
    else:
        # Check shape (N_facets, N_timesteps)
        if temps_all.shape[0] != n_facets:
            # Check if transposed (N_timesteps, N_facets)
            if temps_all.shape[1] == n_facets:
                print(f"Transposing temperature array from {temps_all.shape} to ({n_facets}, {temps_all.shape[0]})")
                temps_all = temps_all.T
            else:
                raise ValueError(f"Temperature shape {temps_all.shape} does not match N_facets={n_facets}. Please re-run TEMPEST to generate temperatures for the current shape model.")
    
    # 3. Setup Geometry
    print(f"Setting up Geometry (Phase: {args.phase_angle} deg)...")
    sun_vec = np.array(config.sunlight_direction)
    
    # Calculate Rotation Axis from RA/Dec if available
    if hasattr(config, 'ra_degrees') and hasattr(config, 'dec_degrees'):
        ra_rad = np.radians(config.ra_degrees)
        dec_rad = np.radians(config.dec_degrees)
        rot_axis = np.array([
            np.cos(ra_rad) * np.cos(dec_rad), 
            np.sin(ra_rad) * np.cos(dec_rad), 
            np.sin(dec_rad)
        ])
    else:
        rot_axis = np.array([0, 0, 1]) # Default
    
    # Rotate observer by phase angle
    # Simple case: Sun at X, Obs rotated around Z? No, phase usually in ecliptic.
    # We use the TEMPEST utility logic: Perpendicular to Sun-Axis plane
    perp_vec = np.cross(sun_vec, rot_axis)
    if np.linalg.norm(perp_vec) < 1e-6: 
        # Sun is parallel to axis (e.g. pole-on), pick arbitrary perp
        perp_vec = np.array([1, 0, 0]) if abs(rot_axis[2]) < 0.9 else np.array([0, 1, 0])
    
    obs_vec = rotate_vector(sun_vec, perp_vec, np.radians(args.phase_angle))
    
    # 4. Calculate Facet Geometry
    lats, phases, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
    
    # 5. Load LUT
    print(f"Loading Roughness LUT: {args.lut}")
    # Load subset for this Theta and Wavelength
    theta = calculate_theta(config)
    
    # Get TI for display
    ti_val = getattr(config, 'thermal_inertia', config.config_data.get('thermal_inertia', 'N/A'))
    print(f"Calculated Theta: {theta:.2f} (from TI={ti_val})")
    
    target_wave = args.wavelength if args.mode == 'spectral' else None
    
    lut = RoughnessLUT(
        args.lut, 
        target_theta=theta, 
        target_rms=90.0, # Canonical Hemisphere Opening Angle
        target_wavelength=target_wave
    )
    
    # 6. Compute Radiance
    print("Computing Radiance...")
    
    # Extract temperature for the requested time
    # Assuming temps_all covers one rotation period
    period_hours = getattr(config, 'rotation_period_hours', config.config_data.get('rotation_period_hours', 24.0))
    n_timesteps = temps_all.shape[1]
    time_norm = (args.time % period_hours) / period_hours
    timestep_idx = int(time_norm * n_timesteps)
    timestep_idx = np.clip(timestep_idx, 0, n_timesteps - 1)
    print(f"Time: {args.time}h / {period_hours}h -> Index: {timestep_idx}/{n_timesteps}")
    
    temps_smooth = temps_all[:, timestep_idx]
    
    # Apply Linear Mixing Fraction
    f = rms_to_fraction(args.roughness_rms_angle)
    print(f"Roughness RMS: {args.roughness_rms_angle} deg -> Mixing Fraction f: {f:.3f}")

    if args.mode == 'spectral':
        # Smooth Radiance
        rad_smooth = planck_function(args.wavelength, temps_smooth)
        
        # Roughness Factors
        if lut.is_loaded:
            factors = lut.get_correction_factors(lats, phases, emis, azis)
        else:
            factors = np.ones(n_facets)
            
        rad_obs = rad_smooth * ( (1.0 - f) + f * factors )
        plot_data = rad_obs
        cbar_label = f'Radiance (W/m2/sr/um) @ {args.wavelength}um'
        title_extra = f"Wave: {args.wavelength}um"
        
    else: # Bolometric Mode
        print("Calculating Bolometric Radiance (looping over LUT wavelengths)...")
        if not lut.is_loaded or 'wavelength' not in lut.axes:
            print("Error: LUT not loaded in spectral mode or missing wavelength axis.")
            return

        wavelengths = lut.axes['wavelength']
        print(f"Wavelengths: {wavelengths}")
        
        # Store full spectrum for interactivity: (N_facets, N_waves)
        full_spectra = np.zeros((n_facets, len(wavelengths)))
        
        for i, wave in enumerate(wavelengths):
            # Smooth Radiance
            rad_smooth = planck_function(wave, temps_smooth)
            
            # Roughness Factors
            factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)
            
            # Observed Radiance
            rad_obs_wave = rad_smooth * ( (1.0 - f) + f * factors )
            full_spectra[:, i] = rad_obs_wave
            
        # Integrate: Trapezoidal rule over wavelengths
        # Result is Radiance (W/m2/sr)
        rad_bol = np.trapz(full_spectra, x=wavelengths, axis=1)
        
        # Convert to Brightness Temperature
        # sigma * T^4 = pi * L (Lambertian assumption for equivalent T)
        # T = (pi * L / sigma)^0.25
        sigma = 5.670374419e-8
        t_eff = (np.pi * rad_bol / sigma) ** 0.25
        
        plot_data = t_eff
        cbar_label = 'Bolometric Brightness Temp (K)'
        title_extra = "Bolometric"

    # 7. Visualization (Project to Image Plane)
    print("Projecting Image...")
    # Basic scatter plot projection
    # Project centers onto plane perpendicular to obs_vec
    # U = cross(obs, up), V = cross(obs, U)
    
    # Check visibility
    visible_mask = emis < 90.0
    
    obs_vec_n = obs_vec / np.linalg.norm(obs_vec)
    up = np.array([0, 0, 1])
    if np.abs(np.dot(up, obs_vec_n)) > 0.9: up = np.array([1, 0, 0])
    cam_u = np.cross(obs_vec_n, up)
    cam_u /= np.linalg.norm(cam_u)
    cam_v = np.cross(obs_vec_n, cam_u)
    
    # Project Centers
    centers = np.array([f.center for f in facets])
    u_coords = np.dot(centers, cam_u)
    v_coords = np.dot(centers, cam_v)
    
    # Project Vertices for PolyCollection
    # mesh.vectors is (N, 3, 3) -> (N_facets, 3_vertices, 3_coords)
    verts = mesh.vectors
    # Dot product with cam_u and cam_v
    # (N, 3, 3) dot (3,) -> (N, 3)
    u_verts = np.dot(verts, cam_u)
    v_verts = np.dot(verts, cam_v)
    
    # Stack to (N, 3, 2)
    # axis 2: (u, v)
    polys = np.stack((u_verts, v_verts), axis=2)
    
    # Filter visible polygons
    polys_visible = polys[visible_mask]
    data_visible = plot_data[visible_mask]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create PolyCollection
    # If wireframe is requested, set edgecolors
    edge_color = 'face'
    line_width = 0.0
    if args.wireframe:
        edge_color = 'white'
        line_width = 0.5
        
    collection = PolyCollection(
        polys_visible, 
        array=data_visible, 
        cmap='inferno', 
        edgecolors=edge_color,
        linewidths=line_width
    )
    
    ax.add_collection(collection)
    ax.autoscale()
    ax.axis('equal')
    
    cbar = plt.colorbar(collection, label=cbar_label)
    plt.title(f"TEMPEST_RAD Observation ({title_extra})\nPhase: {args.phase_angle}°, Time: {args.time}h, RMS: {args.roughness_rms_angle}° (f={f:.2f})")
    plt.grid(True, alpha=0.3)
    
    # Interactive Spectrum Plotting (Bolometric Mode Only)
    if args.mode == 'bolometric':
        print("Interactive Mode: Click on a facet to see its spectrum.")
        
        def on_click(event):
            if event.inaxes != ax: return
            if event.button != 1: return
            
            # Find closest facet center
            # We use the visible u_coords/v_coords
            vis_u = u_coords[visible_mask]
            vis_v = v_coords[visible_mask]
            
            dist = (vis_u - event.xdata)**2 + (vis_v - event.ydata)**2
            closest_idx_in_visible = np.argmin(dist)
            
            # Map back to original index
            # This is tricky with boolean mask. 
            # visible_indices = np.where(visible_mask)[0]
            # original_idx = visible_indices[closest_idx_in_visible]
            original_idx = np.where(visible_mask)[0][closest_idx_in_visible]
            
            # Get spectrum
            spectrum = full_spectra[original_idx]
            
            # Plot in new window
            fig_spec, ax_spec = plt.subplots()
            ax_spec.plot(wavelengths, spectrum, 'o-', label=f'Facet {original_idx}')
            
            # Also plot smooth spectrum for comparison
            rad_smooth_spec = planck_function(wavelengths, temps_smooth[original_idx])
            ax_spec.plot(wavelengths, rad_smooth_spec, '--', label='Smooth Planck', alpha=0.7)
            
            ax_spec.set_xlabel('Wavelength (microns)')
            ax_spec.set_ylabel('Radiance (W/m2/sr/um)')
            ax_spec.set_title(f'Spectrum of Facet {original_idx}\nT_kin={temps_smooth[original_idx]:.1f}K, T_eff={t_eff[original_idx]:.1f}K')
            ax_spec.legend()
            ax_spec.grid(True)
            plt.show()
            
        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.savefig("tempest_rad_observation.png")
    print("Saved observation to tempest_rad_observation.png")
    plt.show()

if __name__ == "__main__":
    main()
