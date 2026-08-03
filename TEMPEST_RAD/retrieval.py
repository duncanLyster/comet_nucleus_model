"""
Roughness retrieval — reusable, application-agnostic helpers.

Given a TEMPEST (smooth) temperature field and a roughness LUT, produce the
disk-integrated ROUGH spectrum for an arbitrary observing geometry (flyby, orbit,
ground-based).  This is the piece a fitter/retrieval loops over model parameters
(thermal inertia -> Theta, roughness RMS slope -> mixing fraction f).

Roughness model (matches simulator.py):
    rad_rough = rad_smooth * ((1 - f) + f * correction_factors)
where f in [0, 1] mixes smooth (0) and the 90-degree hemispherical crater (1),
and correction_factors come from the LUT at each facet's (latitude, rotation
phase, emission angle, azimuth) for the Theta matching the model's TI.

FRAME CONVENTION (caller's responsibility): `facets`, `sun_vec`, `obs_vec` and
`rot_axis` must all be in the SAME body frame — the frame in which the
temperatures were computed and about which the body spins (rot_axis).  Vectors are
unit; `sun_vec`/`obs_vec` point from the target toward the Sun/observer.  The
returned spectrum is proportional to disk-integrated radiance (per-facet
cos(emission)*area weighting); absolute 1/dist^2 / solid-angle scaling and any
unit conversion are left to the caller (they cancel in a per-observation scale
fit).
"""
import numpy as np

from TEMPEST_RAD.simulator import compute_geometry, planck_function, calculate_theta, rms_to_fraction  # noqa: F401
from TEMPEST_RAD.lut import RoughnessLUT


def theta_to_ti(theta, omega, emissivity, tss):
    """Invert Theta = TI*sqrt(omega)/(eps*sigma*Tss^3) -> TI (for LUT<->TI mapping)."""
    sigma = 5.670374419e-8
    return theta * emissivity * sigma * tss ** 3 / np.sqrt(omega)


def disk_integrated_spectrum(facets, temps_t, sun_vec, obs_vec, rot_axis,
                             lut, wavelengths, f):
    """Disk-integrated spectrum for one observation.

    Parameters
    ----------
    facets      : list of simulator.FacetData (normals need not be unit-length)
    temps_t     : (N,) facet temperatures [K] at this observation's rotation phase
    sun_vec,obs_vec,rot_axis : (3,) unit vectors in the body frame (see module docstring)
    lut         : loaded RoughnessLUT (for the Theta matching this model's TI)
    wavelengths : (W,) microns
    f           : roughness mixing fraction in [0, 1]

    Returns
    -------
    (W,) disk-integrated rough radiance [same units as planck_function, relative].
    """
    normals = np.array([fc.normal for fc in facets], dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.array([fc.area for fc in facets], dtype=float)

    cos_e = normals @ obs_vec
    vis = cos_e > 0.0
    w = (cos_e * areas)[vis]                                   # projected-area weights

    lats, phases, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
    lats, phases = lats[vis], phases[vis]
    emis, azis = emis[vis], azis[vis]
    Tvis = temps_t[vis]

    spec = np.empty(len(wavelengths))
    for j, wl in enumerate(wavelengths):
        rad_smooth = planck_function(wl, Tvis)
        if f > 0.0 and lut is not None and lut.is_loaded:
            factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wl)
            rad = rad_smooth * ((1.0 - f) + f * factors)
        else:
            rad = rad_smooth
        spec[j] = np.sum(rad * w)
    return spec


def load_lut_for_theta(theta, theta_files):
    """Pick the LUT whose Theta is nearest `theta`.

    `theta_files` maps theta_value -> path.  NOTE: nearest-neighbour for now; a
    future version should linearly interpolate correction factors between the two
    bracketing Theta files (only one Theta LUT currently exists).
    """
    thetas = np.array(sorted(theta_files))
    pick = float(thetas[np.abs(thetas - theta).argmin()])
    return RoughnessLUT(theta_files[pick], target_theta=pick, target_rms=90.0), pick
