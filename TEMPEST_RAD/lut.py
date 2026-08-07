"""
Roughness Lookup Table (LUT) Handler.

This module manages the loading and interpolation of the 7D High-Fidelity Roughness LUT.
Dimensions: (Theta, OpeningAngle, Latitude, SunPhase, Wavelength, Emission, Azimuth)
"""

import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator
from src.utilities.utils import conditional_print

class RoughnessLUT:
    def __init__(self, lut_path, target_theta=None, target_rms=None, target_wavelength=None):
        self.lut_path = lut_path
        self.interpolator = None
        self.axes = {}
        self.is_loaded = False
        
        # Load only the subset if targets are provided
        if os.path.exists(lut_path):
            self.load_subset(target_theta, target_rms, target_wavelength)
        else:
            print(f"Warning: Roughness LUT not found at {lut_path}")

    def load_subset(self, target_theta, target_rms, target_wavelength):
        """
        Load a specific subset of the LUT matching the target parameters.
        This saves RAM by avoiding loading the entire 7D tensor.

        Supports two on-disk formats:
          v2_per_theta  – one file per theta, lut shape (lat, time, wave, emi, azi)
          v1 legacy     – combined file, lut shape (theta, angle, lat, time, wave, emi, azi)
        """
        try:
            with h5py.File(self.lut_path, 'r') as f:

                # Detect format
                gen_version = f.attrs.get('generator_version', b'')
                if isinstance(gen_version, bytes):
                    gen_version = gen_version.decode()
                is_per_theta = (gen_version == 'v2_per_theta')

                wave_axis = f['wavelength'][:]
                lat_axis  = f['latitude'][:]
                emi_axis  = f['emission'][:]
                azi_axis  = f['azimuth'][:]

                if is_per_theta:
                    # v2 per-theta: lut shape is (lat, time, wave, emi, azi)
                    theta_val = float(f.attrs['theta'])
                    print(f"LUT: v2 per-theta file, Theta={theta_val:.6f}")

                    n_time     = int(f['sun_phase_steps'][()])
                    sun_phases = np.linspace(0, 360, n_time, endpoint=False)

                    if target_wavelength is not None:
                        idx_wave = np.abs(wave_axis - target_wavelength).argmin()
                        print(f"LUT: Selected Wave={wave_axis[idx_wave]} (Target={target_wavelength})")
                        lut_subset = f['lut'][:, :, idx_wave, :, :]
                        points = (lat_axis, sun_phases, emi_axis, azi_axis)
                        self.spectral_mode = False
                    else:
                        lut_subset = f['lut'][:]
                        points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
                        self.spectral_mode = True
                        self.axes['wavelength'] = wave_axis

                else:
                    # v1 legacy: lut shape is (theta, angle, lat, time, wave, emi, azi)
                    theta_axis = f['theta'][:]

                    if target_theta is not None:
                        idx_theta = np.abs(theta_axis - target_theta).argmin()
                        print(f"LUT: Selected Theta={theta_axis[idx_theta]} (Target={target_theta})")
                    else:
                        idx_theta = 0

                    if 'opening_angle' in f:
                        rms_axis = f['opening_angle'][:]
                        print("LUT: Using 'opening_angle' axis.")
                    elif 'rms' in f:
                        rms_axis = f['rms'][:]
                        print("LUT: Using 'rms' axis.")
                    else:
                        raise KeyError("LUT missing 'opening_angle' or 'rms' dataset.")

                    if target_rms is not None:
                        idx_rms = np.abs(rms_axis - target_rms).argmin()
                        print(f"LUT: Selected Angle={rms_axis[idx_rms]} (Target={target_rms})")
                    else:
                        idx_rms = 0

                    dset_name = 'correction_factors' if 'correction_factors' in f else 'lut'

                    if target_wavelength is not None:
                        idx_wave = np.abs(wave_axis - target_wavelength).argmin()
                        print(f"LUT: Selected Wave={wave_axis[idx_wave]} (Target={target_wavelength})")
                        lut_subset = f[dset_name][idx_theta, idx_rms, :, :, idx_wave, :, :]
                        n_time = lut_subset.shape[1]
                        sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                        points = (lat_axis, sun_phases, emi_axis, azi_axis)
                        self.spectral_mode = False
                    else:
                        lut_subset = f[dset_name][idx_theta, idx_rms, :, :, :, :, :]
                        n_time = lut_subset.shape[1]
                        sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                        points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
                        self.spectral_mode = True
                        self.axes['wavelength'] = wave_axis

            # Handle accidental NaNs (a failed (theta, lat) cell) by filling with 1.
            if np.isnan(lut_subset).any():
                lut_subset = np.nan_to_num(lut_subset, nan=1.0)

            # ...but the e = 90 deg row is UNDEFINED by construction, not missing:
            # the smooth reference projects to zero area there, so R = 0/0.  The
            # generator used to store 1.0 in that row, which put a step at the limb
            # (R falls to ~0.45 by e = 87.5 deg and then jumped back to exactly 1).
            # Blank it to NaN so the interpolator refuses the cell rather than
            # silently returning a fabricated value.  Applied at load time so it
            # also corrects the existing LUT sets without regenerating them; EMI_CAP
            # in get_correction_factors() keeps every query well clear of it.
            # Axis order is (..., emission, azimuth) in both spectral and
            # single-wavelength modes, so emission is always second from last.
            i_emi = len(points) - 2
            limb = np.isclose(np.asarray(points[i_emi], dtype=float), 90.0)
            if limb.any():
                sl = [slice(None)] * lut_subset.ndim
                sl[i_emi] = np.where(limb)[0]
                lut_subset = np.array(lut_subset, dtype=float, copy=True)
                lut_subset[tuple(sl)] = np.nan

            self.interpolator = RegularGridInterpolator(
                points, lut_subset, bounds_error=False, fill_value=None
            )
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading Roughness LUT subset: {e}")
            self.is_loaded = False

    def get_correction_factors(self, latitudes, sun_phases, emissions, azimuths, wavelength=None):
        """
        Query the LUT.
        """
        if not self.is_loaded:
            return np.ones_like(emissions)
            
        # Wrap phases
        sun_phases = np.mod(sun_phases, 360.0)
        
        # Mirror Azimuths
        azimuths = np.abs(azimuths)
        azimuths = np.where(azimuths > 180, 360 - azimuths, azimuths)
        
        # Clip Latitudes (just in case)
        latitudes = np.abs(latitudes)
        
        # Cap emission angle to avoid limb divergence.
        # The ratio R = I_rough / I_smooth is ill-conditioned as emission -> 90°
        # because the Lambertian denominator cos(e) -> 0 while the rough radiance
        # stays finite, and the e = 90° row is undefined outright (blanked to NaN
        # above).  Capping at 80° keeps queries in the well-behaved regime; limb
        # facets beyond 80° contribute negligibly to disk-integrated flux
        # (cos 80° = 0.17).
        #
        # 80° is now justified rather than merely convenient.  An exact test —
        # for an ISOTHERMAL crater the ratio must be 1 at every angle, since a ray
        # entering a convex cavity's aperture strikes the interior exactly once,
        # so sum_visible A_i cos(theta_i) == A_aperture cos(e) — gives a pure
        # discretisation error of (200 / 500 / 1000 / 2000 subfacets):
        #     e <= 50°:  <=0.5% / <=0.2% / <=0.1% / <=0.1%
        #     e  = 70°:   -3.1% /   1.2% /   0.3% /   0.3%
        #     e  = 80°:    3.0% /  -0.7% /   0.4% /   0.1%
        #     e  = 85°:   14.2% / -11.1% /   4.7% /  -1.7%
        #     e  = 87.5°: -3.5% / -16.5% /  10.2% / -11.8%
        # i.e. subdividing converges cleanly up to ~80° but NOT beyond it: the
        # binary centroid visibility test cannot converge where only a handful of
        # subfacets are visible and cos(theta) -> 0.  Magri et al. (2018, §4.3)
        # diagnose the same "on/off" limitation in SHERMAN.  Fixing e > 80° needs
        # area-weighted partial visibility, not more facets.
        # See private/dinkinesh_flyby/analysis/.../run_lut_geometry_convergence.py
        EMI_CAP = 80.0
        emissions = np.minimum(emissions, EMI_CAP)
        
        if self.spectral_mode:
            if wavelength is None:
                # Default to first wavelength if none provided
                wavelength = self.axes['wavelength'][0]
                
            # Broadcast wavelength to match other arrays
            n = len(latitudes)
            waves = np.full(n, wavelength)
            
            # Query: (Lat, Time, Wave, Emi, Azi)
            query_points = np.column_stack((
                latitudes, sun_phases, waves, emissions, azimuths
            ))
        else:
            # Fixed wavelength mode
            # Query: (Lat, Time, Emi, Azi)
            query_points = np.column_stack((
                latitudes, sun_phases, emissions, azimuths
            ))
            
        return self.interpolator(query_points)
