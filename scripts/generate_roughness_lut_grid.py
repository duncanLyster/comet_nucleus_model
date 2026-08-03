"""Generate a grid of roughness LUTs over the thermal parameter Theta.

TEMPEST_RAD's generator produces a directional-emissivity correction factor
R(Theta, lat, phase, wavelength, emission, azimuth) for a spherical-cap crater.
Fitting rough-surface spectra needs R at the Theta matching each candidate
thermal inertia, so LUTs are normally generated as a grid rather than singly.

Theta is the standard thermal parameter,

    Theta = TI * sqrt(omega) / (eps * sigma * T_ss^3)

so a given Theta maps to different thermal inertias for different rotation
periods and heliocentric distances -- pick the grid to span the TI range your
retrieval explores, for your target's spin rate.

CONFIGURATION GOES THROUGH ENVIRONMENT VARIABLES, not by patching module
globals.  joblib's loky workers re-import `TEMPEST_RAD.generator`, so a
monkey-patched global is silently lost in the workers: the parent process sees
the patched value, the workers see the module default, and the returned grids
no longer match the allocated output arrays.  Env vars survive fork/spawn.

Examples
--------
Quick low-resolution grid (minutes, for smoke-testing the pipeline):
    python scripts/generate_roughness_lut_grid.py \
        --theta 0.5,1,2 --subfacets 200 --vf-rays 2000 \
        --sim-timesteps 144 --lut-timesteps 36 --prefix roughness_lut_lowres

Production grid (hours):
    python scripts/generate_roughness_lut_grid.py \
        --theta 0.15,0.5,1,2,4,8 --subfacets 500 --vf-rays 10000 --n-jobs 6

Deeper-than-hemispherical cavity, with converged multiple scattering:
    python scripts/generate_roughness_lut_grid.py \
        --theta 1,2,4 --opening-angle 120 --n-scatters 4 --prefix roughness_lut_steep
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--theta', default='0.15,0.5,1,2,4,8',
                   help='comma-separated Theta values (default: %(default)s)')
    p.add_argument('--opening-angle', default=None,
                   help='crater opening angle(s) in deg, comma-separated. '
                        '90 = hemisphere (generator default); >90 = deeper cavity')
    p.add_argument('--n-scatters', type=int, default=1,
                   help='inter-facet scattering iterations inside the crater; '
                        '1 = single bounce, higher converges the multiple-scattering '
                        'beaming enhancement (default: %(default)s)')
    p.add_argument('--subfacets', type=int, default=500,
                   help='crater kernel resolution (default: %(default)s)')
    p.add_argument('--vf-rays', type=int, default=10000,
                   help='rays per facet for view factors (default: %(default)s)')
    p.add_argument('--sim-timesteps', type=int, default=None,
                   help='thermal-model timesteps per rotation')
    p.add_argument('--lut-timesteps', type=int, default=None,
                   help='phase samples stored in the LUT (<= --sim-timesteps)')
    p.add_argument('--n-jobs', type=int, default=None,
                   help='parallel workers (default: generator default)')
    p.add_argument('--outdir', default=None,
                   help="output directory (default: the generator's 'lut_output', "
                        "resolved relative to the current working directory)")
    p.add_argument('--prefix', default=None,
                   help='output filename prefix (default: generator default)')
    p.add_argument('--apply-norm', action='store_true',
                   help='re-apply the legacy per-timestep bolometric normalisation '
                        'alpha(t).  Off by default: with self-heating and scattering '
                        'correct, energy closure is inherent, and forcing instantaneous '
                        'closure erases genuine diurnal beaming.  For A/B comparison only.')
    args = p.parse_args()

    # Keep BLAS single-threaded: the parallelism is across cases, and nested
    # threading oversubscribes the CPU badly here.
    for var in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[var] = '1'

    os.environ['LUT_THETA_VALUES'] = args.theta
    os.environ['LUT_N_SCATTERS'] = str(args.n_scatters)
    os.environ['LUT_CRATER_SUBFACETS'] = str(args.subfacets)
    os.environ['LUT_VIEW_FACTOR_RAYS'] = str(args.vf_rays)
    os.environ['LUT_APPLY_NORM'] = '1' if args.apply_norm else '0'

    if args.opening_angle is not None:
        os.environ['LUT_OPENING_ANGLES'] = args.opening_angle
    if args.sim_timesteps is not None:
        os.environ['LUT_SIM_TIMESTEPS'] = str(args.sim_timesteps)
    if args.lut_timesteps is not None:
        os.environ['LUT_LUT_TIMESTEPS'] = str(args.lut_timesteps)
    if args.n_jobs is not None:
        os.environ['LUT_N_JOBS'] = str(args.n_jobs)
    if args.outdir is not None:
        os.environ['LUT_OUTPUT_DIR'] = str(Path(args.outdir).expanduser().resolve())
    if args.prefix is not None:
        os.environ['LUT_OUTPUT_PREFIX'] = args.prefix

    # Import only AFTER the env is set -- the generator reads these at import time.
    sys.path.insert(0, str(ROOT))
    import TEMPEST_RAD.generator as gen

    Path(gen.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Theta grid      : {gen.THETA_VALUES}")
    print(f"Opening angles  : {gen.OPENING_ANGLES} deg")
    print(f"Crater subfacets: {gen.CRATER_SUBFACETS}   VF rays: {gen.VIEW_FACTOR_RAYS}")
    print(f"Timesteps       : sim {gen.SIM_TIMESTEPS} -> LUT {gen.LUT_TIMESTEPS}")
    print(f"Scatters        : {gen.N_SCATTERS_ENV}   apply alpha(t): {args.apply_norm}")
    print(f"Output          : {gen.OUTPUT_DIR}")
    gen.main()


if __name__ == '__main__':
    main()
