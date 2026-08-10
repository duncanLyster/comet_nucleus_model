# src/model/simulation.py

import numpy as np
from src.utilities.locations import Locations

class Simulation:
    """
    Container for the physical parameters of a run.

    Only the *independent* inputs (thermal_inertia, density,
    specific_heat_capacity, rotation_period_hours, n_layers, solar_distance_au,
    and an optional timesteps_per_day override) are stored as attributes.
    Everything derived from them -- delta_t, angular_velocity, skin_depth,
    layer_thickness, the conductivity and diffusivity -- is a read-only
    @property evaluated on read.

    This matters because callers routinely mutate a Simulation after
    construction (e.g. the roughness LUT generator sweeps thermal inertia and
    timestep count on a single object). When the derived quantities were plain
    attributes computed once in __init__, such a mutation left them stale: in
    particular delta_t kept the value implied by the *adaptive* timestep count,
    so the integration advanced by the wrong step and the effective rotation
    period became delta_t * timesteps_per_day rather than the true period.
    Deriving on read makes that class of bug unrepresentable.
    """

    # Optional user/caller override of the adaptive timestep count. Class-level
    # default so the timesteps_per_day property is safe to read at any point.
    _timesteps_per_day_override = None

    def __init__(self, config):
        """
        Initialize the simulation using the provided Config object.
        """
        self.config = config
        self.load_configuration()

    def load_configuration(self):
        """
        Load configuration directly from the Config object.
        """
        self._timesteps_per_day_override = None

        # Assign configuration to attributes, converting lists to numpy arrays as needed
        # (a 'timesteps_per_day' key here goes through the property setter below)
        for key, value in self.config.config_data.items():
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, key, value)

        # Compute rotation axis vector
        self._compute_rotation_axis()

        # Report a config-specified timestep count that is unsafe for the explicit solver
        self._warn_if_cfl_violated()

    # --- Derived quantities -------------------------------------------------
    # All computed on read, so they can never go stale when an input changes.

    @property
    def solar_distance_m(self):
        """Heliocentric distance in metres."""
        return self.solar_distance_au * 1.496e11

    @property
    def rotation_period_s(self):
        """Rotation period in seconds."""
        return self.rotation_period_hours * 3600

    @property
    def angular_velocity(self):
        """Rotation rate omega (rad/s)."""
        return (2 * np.pi) / self.rotation_period_s

    @property
    def thermal_conductivity(self):
        """k, from the thermal inertia Gamma = sqrt(k rho c)."""
        return (self.thermal_inertia**2 / (self.density * self.specific_heat_capacity))

    @property
    def thermal_diffusivity(self):
        """kappa = k / (rho c)."""
        return self.thermal_conductivity / (self.density * self.specific_heat_capacity)

    @property
    def skin_depth(self):
        """Diurnal thermal skin depth sqrt(kappa / omega)."""
        return (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity)) ** 0.5

    @property
    def layer_thickness(self):
        """Subsurface layer thickness for a grid spanning 8 skin depths."""
        return 8 * self.skin_depth / self.n_layers

    @property
    def timesteps_per_day(self):
        """
        Timesteps per rotation: the caller's value if one was supplied
        (in the config or by assignment), otherwise the adaptive count.
        """
        if self._timesteps_per_day_override is not None:
            return int(self._timesteps_per_day_override)
        return self.calculate_adaptive_timesteps()

    @timesteps_per_day.setter
    def timesteps_per_day(self, value):
        self._timesteps_per_day_override = None if value is None else int(value)

    @property
    def delta_t(self):
        """Integration timestep. Always consistent with the current period and timestep count."""
        return self.rotation_period_s / self.timesteps_per_day

    # ------------------------------------------------------------------------

    def calculate_adaptive_timesteps(self):
        """
        Timestep count required for stability, from CFL plus a limit on the
        insolation coefficient (mainly for the explicit solver).

        This is the value used when the caller has not specified
        timesteps_per_day; it does not consult the override.
        """
        # Stability calculation (CFL limits)
        # Use a safety factor to ensure const3 is well below 0.5
        cfl_safety_factor = 0.8
        cfl_denominator = cfl_safety_factor * (self.layer_thickness**2 / (2 * self.thermal_diffusivity))
        timesteps_cfl = int(np.ceil(self.rotation_period_s / cfl_denominator))

        delta_t_cfl = self.rotation_period_s / timesteps_cfl

        # Calculate insolation coefficient with CFL timestep
        const1_cfl = delta_t_cfl / (self.layer_thickness * self.density * self.specific_heat_capacity)

        # Adaptive constraint: limit const1 for stability
        # Lower limit to 0.01 to ensure radiative stability at high T (~400K)
        max_const1 = 0.01

        if const1_cfl > max_const1:
            # Calculate timestep that keeps const1 reasonable
            required_delta_t = max_const1 * self.layer_thickness * self.density * self.specific_heat_capacity
            adaptive_timesteps = int(np.ceil(self.rotation_period_s / required_delta_t))
            return adaptive_timesteps
        else:
            return timesteps_cfl

    def _warn_if_cfl_violated(self):
        """
        Warn if an explicitly specified timesteps_per_day is too coarse for the
        explicit solver's CFL limit.
        """
        user_timesteps = self._timesteps_per_day_override
        if user_timesteps is None:
            return
        if getattr(self, 'temp_solver', '') != 'tempest_standard':
            return

        cfl_safety_factor = 0.8
        cfl_denominator = cfl_safety_factor * (self.layer_thickness**2 / (2 * self.thermal_diffusivity))
        timesteps_cfl = int(np.ceil(self.rotation_period_s / cfl_denominator))

        if int(user_timesteps) < timesteps_cfl:
            print("\n" + "=" * 80)
            print("  WARNING: CFL STABILITY VIOLATION (explicit solver)")
            print("=" * 80)
            print(f"  You specified timesteps_per_day = {int(user_timesteps)}, but the CFL")
            print(f"  stability criterion requires at least {timesteps_cfl} timesteps.")
            print(f"  The explicit solver will likely produce unphysical results (e.g. all")
            print(f"  temperatures dropping to 2.7 K).")
            print(f"")
            print(f"  Fix: either remove/comment out 'timesteps_per_day' from your config to let TEMPEST")
            print(f"  choose automatically, or increase it to >= {timesteps_cfl}.")
            print("=" * 80 + "\n")

    def _compute_rotation_axis(self):
        """
        Compute the rotation axis unit vector from orbital/seasonal parameters.
        
        Uses obliquity and north_pole_solar_longitude if available (new system).
        Falls back to RA/Dec if provided (legacy system for backwards compatibility).
        
        Coordinate system:
        - Orbital plane is XY, with sunlight along +X
        - Orbital normal (north) is +Z
        
        obliquity: Tilt of rotation axis from orbital normal (degrees)
        north_pole_solar_longitude: Azimuthal angle in orbital plane where pole faces sun (degrees)
        """
        obliquity = getattr(self, 'obliquity_degrees', 0)
        north_pole_lon = getattr(self, 'north_pole_solar_longitude_degrees', 0)
        ra = getattr(self, 'ra_degrees', None)
        dec = getattr(self, 'dec_degrees', None)
        
        if (obliquity != 0 or north_pole_lon != 0) or (ra is None and dec is None):
            # Use new obliquity/north_pole_solar_longitude system
            obliquity_rad = np.radians(obliquity)
            north_pole_lon_rad = np.radians(north_pole_lon)
            
            # Rotation axis in orbital frame (Z = orbital normal, XY = orbital plane with sun at +X)
            # Component perpendicular to orbital plane (Z component)
            z_component = np.cos(obliquity_rad)
            # Component in orbital plane, tilted by north_pole_solar_longitude
            xy_magnitude = np.sin(obliquity_rad)
            x_component = xy_magnitude * np.cos(north_pole_lon_rad)
            y_component = xy_magnitude * np.sin(north_pole_lon_rad)
            
            self.rotation_axis = np.array([x_component, y_component, z_component])
        else:
            # Use legacy RA/Dec system for backwards compatibility
            ra_radians = np.radians(ra)
            dec_radians = np.radians(dec)
            self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                                           np.sin(ra_radians) * np.cos(dec_radians), 
                                           np.sin(dec_radians)])

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        # Surface temperatures for one day only
        self.temperatures = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        # Two columns for current and previous timestep subsurface temperatures
        self.layer_temperatures = np.zeros((n_facets, 2, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]
        self.thermal_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        if calculate_energy_terms:
            # Energy terms for one day only
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day))

    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
    def set_thermal_view_factors(self, view_factors):
        """Set thermal view factors for all facets."""
        self.thermal_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]