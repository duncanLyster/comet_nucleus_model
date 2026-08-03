# src/model/insolation.py

'''
This module calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

NOTE: Currently this requires a lot of RAM - look for ways to reduce this and check if it's the main bottleneck in the code.
'''

import time
import gc
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from src.utilities.utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix
)   
from src.model.scattering import BRDFLookupTable
from src.model.insolation_cache import InsolationCache
from src.utilities.locations import Locations
from tqdm import tqdm

def calculate_insolation(thermal_data, shape_model, simulation, config):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    
    Supports caching of scattered light insolation results to enable faster reruns when only thermal properties change.

    TODO: Parallelise this function.
    '''
    # Initialize cache if enabled
    cache = None
    locations = Locations()
    cache_dir = locations.other_cached_data
    
    if config.use_insolation_cache and not config.force_recalculate_insolation:
        cache = InsolationCache(cache_dir)
        
        # Try to load from cache
        n_facets = len(shape_model)
        shape_model_path = (config.path_to_shape_model_file if hasattr(config, 'path_to_shape_model_file') else "") or ""

        cached_insolation = cache.load_insolation(
            config, simulation, shape_model_path, n_facets, config.silent_mode
        )
        
        if cached_insolation is not None:
            conditional_print(config.silent_mode, 
                            "Using cached insolation results (direct + scattered light)")
            thermal_data.insolation = cached_insolation
            return thermal_data
    
    # Initialize insolation array with zeros for all facets and timesteps
    insolation_array = np.zeros((len(shape_model), simulation.timesteps_per_day))
    
    # Precompute rotation matrices and rotated sunlight directions
    rotation_matrices = np.zeros((simulation.timesteps_per_day, 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((simulation.timesteps_per_day, 3), dtype=np.float64)
    
    for t in range(simulation.timesteps_per_day):
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / simulation.timesteps_per_day) * t)
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, simulation.sunlight_direction)
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])

    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    conditional_print(config.silent_mode, f"Extracting shape model data for {len(shape_model)} facets...")
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)
    conditional_print(config.silent_mode, f"Shape model data extracted. Processing {len(chunks)} chunks in parallel...")
    
    # Process chunks in parallel
    insolation_start = time.time()
    parallel = Parallel(n_jobs=config.n_jobs, verbose=0)
    results = parallel(
        delayed(process_insolation_chunk)(
            normals[start_idx:end_idx],
            positions[start_idx:end_idx],
            thermal_data.visible_facets[start_idx:end_idx],
            rotation_matrices,
            rotated_sunlight_directions,
            simulation.solar_luminosity,
            simulation.albedo,
            simulation.solar_distance_m,
            simulation.sunlight_direction.astype(np.float64),  # Ensure float64
            config.include_shadowing,
            shape_model_vertices
        )
        for start_idx, end_idx in chunks
    )
    insolation_end = time.time()

    # Combine results
    conditional_print(config.silent_mode, f"Combining insolation results from parallel chunks...")
    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        thermal_data.insolation[start_idx:end_idx] = results[chunk_idx]
    
    conditional_print(config.silent_mode, 
                    f"Time taken to calculate insolation: {insolation_end - insolation_start:.2f} seconds")
    
    # OPTIMIZATION: Explicitly clean up parallel worker pool to free memory before next phase
    # Loky keeps worker processes alive and they can cache large chunks of data
    del parallel, results
    gc.collect()
    conditional_print(config.silent_mode, "Cleaned up parallel workers.")

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, 
                        f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config,
                                      rotation_matrices, rotated_sunlight_directions)
        scattering_end = time.time()
        conditional_print(config.silent_mode, 
                        f"Time taken to apply light scattering: {scattering_end - scattering_start:.2f} seconds")
    
    # Save to cache if enabled
    if config.use_insolation_cache and cache is not None:
        shape_model_path = (config.path_to_shape_model_file if hasattr(config, 'path_to_shape_model_file') else "") or ""
        cache.save_insolation(
            thermal_data.insolation, config, simulation, shape_model_path, config.silent_mode
        )

    return thermal_data

@jit(nopython=True)
def process_insolation_chunk(normals, positions, visible_facets, rotation_matrices, 
                           rotated_sunlight_directions, solar_luminosity, albedo,
                           solar_distance_m, sunlight_direction, include_shadowing,
                           shape_model_vertices):
    """
    Process insolation calculations for a chunk of facets using only numba-compatible types.
    """
    chunk_size = len(normals)
    timesteps = len(rotation_matrices)
    insolation = np.zeros((chunk_size, timesteps))
    
    # physical softening of the shadow terminator edge
    # smoothing sun size: roughly 2 degrees. (e.g. 4 timesteps out of 720)
    window_size = max(1, timesteps // 180)
    pad_left = window_size // 2
    
    for i in range(chunk_size):
        normal = normals[i]
        position = positions[i]
        
        illum_factors = np.zeros(timesteps, dtype=np.float64)
        cos_zeniths = np.zeros(timesteps, dtype=np.float64)
        
        # 1. First pass: compute binary shadow state and zenith angle
        for t in range(timesteps):
            new_normal = np.dot(rotation_matrices[t], normal)
            new_normal_norm = np.linalg.norm(new_normal)
            sun_dot_normal = np.dot(sunlight_direction, new_normal)
            
            cos_zenith_angle = sun_dot_normal / (np.linalg.norm(sunlight_direction) * new_normal_norm)
            
            if cos_zenith_angle > 0:
                cos_zeniths[t] = cos_zenith_angle
                if len(visible_facets[i]) != 0 and include_shadowing:
                    illum_factors[t] = float(calculate_shadowing(
                        position, 
                        rotated_sunlight_directions[t:t+1],  
                        shape_model_vertices,
                        visible_facets[i]
                    ))
                else:
                    illum_factors[t] = 1.0
                    
        # 2. Second pass: apply moving average to illumination_factors to simulate solar disk diameter
        # This mitigates Crank-Nicolson spike artifacts near terminators
        smoothed_illum = np.zeros(timesteps, dtype=np.float64)
        for t in range(timesteps):
            if window_size <= 1:
                smoothed_illum[t] = illum_factors[t]
            else:
                sum_val = 0.0
                for w in range(window_size):
                    idx = (t - pad_left + w) % timesteps
                    sum_val += illum_factors[idx]
                smoothed_illum[t] = sum_val / window_size
                
        # 3. Third pass: calculate final insolation combining smoothed shadow and original zenith angle
        for t in range(timesteps):
            if cos_zeniths[t] > 0 and smoothed_illum[t] > 0:
                insolation[i, t] = (
                    solar_luminosity * 
                    (1 - albedo) * 
                    smoothed_illum[t] * 
                    cos_zeniths[t] / 
                    (4 * np.pi * solar_distance_m**2)
                )
    
    return insolation

@jit(nopython=True)
def calculate_shadowing(subject_positions, sunlight_directions, shape_model_vertices, visible_facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to rays_triangles_intersections which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not.

    TODO: Use Monte Carlo or even distribution ray tracing for faster shadowing calculations.
    '''

    # Ensure triangles_vertices is an array of shape (m, 3, 3)
    triangles_vertices = shape_model_vertices[visible_facet_indices]

    # Call the intersection function
    intersections, t_values = rays_triangles_intersection(
        subject_positions,
        sunlight_directions,
        triangles_vertices
    )

    # Check for any intersection
    if intersections.any():
        return 0  # The facet is in shadow
        
    return 1 # The facet is not in shadow

def calculate_brdf_values(chunk_normals, chunk_positions, all_normals, all_positions,
                          rotation_matrices, rotated_sunlight_directions,
                          brdf_lut, start_idx, end_idx, chunk_visible_facets):
    """
    Compute per-facet-pair BRDF values for a chunk of source facets.

    Memory-efficient design:
    - Pre-rotates only the source chunk: (chunk_size, T, 3) -- e.g. 5000*360*3*8 = ~43 MB
    - Computes target rotations one facet at a time in the inner loop: (T, 3) each -- trivial
    - Never allocates an (n_facets, T, 3) array, which would be ~2.7 GB at 314k facets
    """
    n_timesteps = len(rotation_matrices)
    brdf_values = {}

    # Pre-rotate all source facets in this chunk at once
    rotated_chunk_normals   = np.einsum('tij,nj->nti', rotation_matrices, chunk_normals)    # (n_chunk, T, 3)
    rotated_chunk_positions = np.einsum('tij,nj->nti', rotation_matrices, chunk_positions)  # (n_chunk, T, 3)

    for i in range(start_idx, end_idx):
        local_i = i - start_idx
        visible_facets = chunk_visible_facets[local_i]
        brdf_values[i] = np.zeros((len(visible_facets), n_timesteps))

        src_normals_t   = rotated_chunk_normals[local_i]    # (T, 3)
        src_positions_t = rotated_chunk_positions[local_i]  # (T, 3)

        sun_cos = np.einsum('tj,tj->t', rotated_sunlight_directions, src_normals_t)
        illuminated_t = np.where(sun_cos > 0)[0]

        if len(illuminated_t) == 0:
            continue

        inc_deg = np.degrees(np.arccos(np.clip(sun_cos[illuminated_t], -1.0, 1.0)))

        for j, target_idx in enumerate(visible_facets):
            # Rotate single target facet on-the-fly: (T, 3) -- negligible memory
            rot_target_normal = np.einsum('tij,j->ti', rotation_matrices, all_normals[target_idx])
            rot_target_pos    = np.einsum('tij,j->ti', rotation_matrices, all_positions[target_idx])

            for k, t in enumerate(illuminated_t):
                # Direction vector from source facet centre to target facet centre
                direction = rot_target_pos[t] - src_positions_t[t]
                dist = np.linalg.norm(direction)
                if dist < 1e-10:
                    continue
                direction /= dist

                # Emission angle: angle between source normal and direction to target
                em_cos = np.dot(src_normals_t[t], direction)
                if em_cos <= 0:
                    continue
                em_deg = np.degrees(np.arccos(np.clip(em_cos, -1.0, 1.0)))

                # Azimuth angle: angle between projected sun and emission in source facet plane
                sun_dir  = rotated_sunlight_directions[t]
                src_norm = src_normals_t[t]
                sun_proj = sun_dir  - np.dot(sun_dir,  src_norm) * src_norm
                em_proj  = direction - np.dot(direction, src_norm) * src_norm
                sun_len  = np.linalg.norm(sun_proj)
                em_len   = np.linalg.norm(em_proj)
                if sun_len < 1e-10 or em_len < 1e-10:
                    az_deg = 0.0
                else:
                    cos_az = np.dot(sun_proj / sun_len, em_proj / em_len)
                    az_deg = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))

                brdf_values[i][j, t] = brdf_lut.query(inc_deg[k], em_deg, az_deg)

    return brdf_values

def process_scattering_chunk(start_idx, end_idx, chunk_input_light, chunk_visible_facets,
                             chunk_view_factors, chunk_areas, all_areas,
                             timesteps_per_day, albedo, iteration,
                             brdf_lut=None, chunk_normals=None, chunk_positions=None,
                             all_normals=None, all_positions=None,
                             rotation_matrices=None, rotated_sunlight_directions=None):
    """
    Process a chunk of facets for scattering calculation.
    Returns (dest_indices, compact_array) to minimise returned-data size.

    When brdf_lut is None (Lambertian), uses brdf=1.0 and skips the expensive
    BRDF calculation. For non-Lambertian LUTs, call calculate_brdf_values which
    is memory-efficient (only pre-rotates the source chunk, not all n_facets).
    """
    # Collect unique destination facet indices for this chunk
    dest_set = set()
    for vf_arr in chunk_visible_facets:
        dest_set.update(vf_arr.tolist())
    dest_indices = np.array(sorted(dest_set), dtype=np.int64)
    dest_to_local = {g: l for l, g in enumerate(dest_indices)}
    compact_scattered = np.zeros((len(dest_indices), timesteps_per_day))

    # Compute BRDF values if a non-Lambertian LUT is provided
    if brdf_lut is not None:
        brdf_values = calculate_brdf_values(
            chunk_normals, chunk_positions, all_normals, all_positions,
            rotation_matrices, rotated_sunlight_directions,
            brdf_lut, start_idx, end_idx, chunk_visible_facets
        )
    else:
        brdf_values = None  # Lambertian: brdf=1.0 throughout

    for i in range(start_idx, end_idx):
        local_i = i - start_idx
        visible_facets = np.asarray(chunk_visible_facets[local_i])
        view_factors   = np.asarray(chunk_view_factors[local_i], dtype=np.float64)
        if len(visible_facets) == 0:
            continue
        area_i = chunk_areas[local_i]
        # absorbed flux at the destination: albedo * E_i(t) * F_ij * A_i / A_j
        # (F_ij is the diffuse view factor; A_i F_ij = A_j F_ji reciprocity)
        coef = albedo * view_factors * area_i / all_areas[visible_facets]   # (n_vis,)
        dloc = np.array([dest_to_local[v] for v in visible_facets])
        src = chunk_input_light[local_i]                                    # (T,)
        if brdf_values is not None:
            contrib = coef[:, None] * brdf_values[i] * src[None, :]         # (n_vis, T)
        else:
            contrib = coef[:, None] * src[None, :]
        np.add.at(compact_scattered, dloc, contrib)

    return dest_indices, compact_scattered

def apply_scattering(thermal_data, shape_model, simulation, config, 
                    rotation_matrices, rotated_sunlight_directions):
    """
    Memory-optimized scattering using BRDF lookup tables. Works with any number of jobs (including 1).
    
    Key optimizations:
    - Accumulates scattered light directly into thermal_data.insolation (no intermediate copies)
    - Explicit garbage collection between iterations to free worker memory
    - Reduced chunk_size for better memory control on large models
    
    Pre-extracts compact numpy arrays and slices per-chunk to minimize worker memory.
    """
    import gc
    
    # Initialize BRDF lookup table if needed
    # Lambertian is detected by filename; for Lambertian we skip BRDF (brdf=1.0 always)
    is_lambertian = 'lambertian' in config.scattering_lut.lower()
    brdf_lut = None if is_lambertian else BRDFLookupTable(config.scattering_lut)

    n_facets = len(shape_model)
    # OPTIMIZATION: Store the initial insolation reference (not a copy) for first iteration input
    initial_insolation = thermal_data.insolation
    
    # Pre-extract compact numpy arrays from shape_model.
    # For non-Lambertian BRDF: all_normals/all_positions (~15 MB each at 314k facets)
    # are passed to workers so they can rotate target facets on-the-fly.
    all_normals   = np.array([facet.normal   for facet in shape_model], dtype=np.float64)
    all_positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    all_areas     = np.array([facet.area     for facet in shape_model], dtype=np.float64)

    # Convert lists to numpy arrays
    visible_facets_list = [np.array(x) for x in thermal_data.visible_facets]
    view_factors_list = [np.array(x) for x in thermal_data.secondary_radiation_view_factors]
    
    # Get number of jobs and create chunks
    actual_n_jobs = config.validate_jobs()
    
    # OPTIMIZATION: For large models, reduce chunk_size to control per-worker memory
    # Default was n_facets // (n_jobs * 4), now we use n_facets // (n_jobs * 8) for better memory distribution
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 8))
    else:
        # If user specified chunk size, scale down by 50% for high-res models (>200k facets)
        if n_facets > 200000:
            config.chunk_size = max(1, config.chunk_size // 2)
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]
    
    # OPTIMIZATION: Initialize total_scattered_light as float32 to save 50% memory for accumulation
    # (intermediate precision sufficient for radiation calculation)
    timesteps = simulation.timesteps_per_day
    total_scattered_light = np.zeros((n_facets, timesteps), dtype=np.float32)
    
    for iteration in range(config.n_scatters):
        if iteration == 0:
            input_light = initial_insolation
        else:
            input_light = scattered_light
        
        # Create delayed functions
        delayed_funcs = [
            delayed(process_scattering_chunk)(
                start_idx, end_idx,
                input_light[start_idx:end_idx],              # Only this chunk's input
                visible_facets_list[start_idx:end_idx],      # Only this chunk's visible facets
                view_factors_list[start_idx:end_idx],        # Only this chunk's view factors
                all_areas[start_idx:end_idx],                # Source facet areas
                all_areas,                                   # Destination facet areas
                simulation.timesteps_per_day,
                simulation.albedo,
                iteration,
                brdf_lut,                                    # None for Lambertian
                all_normals[start_idx:end_idx] if brdf_lut is not None else None,   # Source chunk normals
                all_positions[start_idx:end_idx] if brdf_lut is not None else None, # Source chunk positions
                all_normals   if brdf_lut is not None else None,  # All normals for target lookup
                all_positions if brdf_lut is not None else None,  # All positions for target lookup
                rotation_matrices          if brdf_lut is not None else None,
                rotated_sunlight_directions if brdf_lut is not None else None
            )
            for start_idx, end_idx in chunks
        ]
        
        # Use context manager to ensure proper thread pool cleanup between iterations
        # This prevents resource contention that causes performance degradation on iteration 2+
        with Parallel(n_jobs=actual_n_jobs, verbose=0, backend='threading') as parallel:
            # Accumulate scattered light in float32, then add to float64 total
            # Process chunks with real-time progress bar
            # Workers return sparse (dest_indices, compact_array) to avoid allocating
            # a full n_facets output array per worker (saves ~900 MB per worker at 314k facets)
            scattered_light = np.zeros((n_facets, timesteps), dtype=np.float32)
            with tqdm(total=len(chunks), desc=f"Iteration {iteration + 1}/{config.n_scatters}") as pbar:
                for dest_indices, compact_result in parallel(delayed_funcs):
                    scattered_light[dest_indices] += compact_result.astype(np.float32)
                    pbar.update(1)
        
        total_scattered_light += scattered_light
        gc.collect()  # Clean up after each iteration to free worker memory
    
    # Final cleanup after all iterations complete
    conditional_print(config.silent_mode, "Cleaned up scattering workers.")
    
    # OPTIMIZATION: Accumulate directly into thermal_data.insolation instead of creating intermediate copies
    # Convert accumulated float32 back to float64 and add to the original insolation
    thermal_data.insolation = initial_insolation + total_scattered_light.astype(np.float64)
    
    # Clean up large temporary arrays
    del total_scattered_light, all_normals, all_positions
    gc.collect()
    
    return thermal_data
