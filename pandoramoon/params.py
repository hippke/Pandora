"""Pandora's model parameters: model_params (the parameter container of the class-based interface), the time
grid helper time, and the physical range check (params_valid for compiled code, check_params for users)."""
import numpy as np
from numba import jit

from pandoramoon.grids import timegrid
from pandoramoon._config import settings

_MODEL = settings()["model"]


# Physical parameter ranges (checked before any computation). The model
# functions raise ValueError(PARAMS_ERROR) outside them, pandora_loglike /
# Likelihood return NEG_INF; check_params() names the offending values.
PARAMS_ERROR = (
    "Pandora parameter outside its physical range; "
    "pandoramoon.check_params(...) names it"
)


@jit(cache=True, nopython=True, fastmath=False, parallel=False)
def params_valid(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, supersampling_factor,
):
    """True if all parameters are within their physical ranges:

    - every value finite (no NaN / inf);
    - R_star, per_bary, a_bary, M_planet, per_moon, epoch_distance > 0
      (a retrograde moon is i_moon in (90, 180] deg, not per_moon < 0);
    - r_planet, r_moon, M_moon >= 0;
    - 0 <= ecc_bary < 1, 0 <= ecc_moon < 1;
    - quadratic limb darkening physical (Kipping 2013: intensity positive
      and decreasing towards the limb): u1 >= 0, u1 + u2 <= 1, u1 + 2 u2 >= 0
      (the sums with 1e-12 slack for round-off of converted q1, q2);
    - supersampling_factor >= 1.

    Compiled without fastmath, so that the NaN / inf tests are kept (x - x
    == 0 is False for NaN and +-inf). Physical plausibility of the moon orbit
    (Hill sphere, contact) is not part of it: such moons keep the planet-only
    model (_moon_setup)."""
    finite = (
        u1 - u1 == 0.0 and u2 - u2 == 0.0 and R_star - R_star == 0.0
        and per_bary - per_bary == 0.0 and a_bary - a_bary == 0.0
        and r_planet - r_planet == 0.0 and b_bary - b_bary == 0.0
        and w_bary - w_bary == 0.0 and ecc_bary - ecc_bary == 0.0
        and t0_bary - t0_bary == 0.0 and t0_bary_offset - t0_bary_offset == 0.0
        and M_planet - M_planet == 0.0 and r_moon - r_moon == 0.0
        and per_moon - per_moon == 0.0 and tau_moon - tau_moon == 0.0
        and Omega_moon - Omega_moon == 0.0 and i_moon - i_moon == 0.0
        and ecc_moon - ecc_moon == 0.0 and w_moon - w_moon == 0.0
        and M_moon - M_moon == 0.0 and epoch_distance - epoch_distance == 0.0
    )
    return (
        finite
        and R_star > 0 and per_bary > 0 and a_bary > 0 and M_planet > 0
        and per_moon > 0 and epoch_distance > 0
        and r_planet >= 0 and r_moon >= 0 and M_moon >= 0
        and ecc_bary >= 0 and ecc_bary < 1 and ecc_moon >= 0 and ecc_moon < 1
        and u1 >= 0 and u1 + u2 <= 1 + 1e-12 and u1 + 2 * u2 >= -1e-12
        and supersampling_factor >= 1
    )


def check_params(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance=1.0, supersampling_factor=1
):
    """Python: the list of violated physical ranges of params_valid()
    (empty if all parameters are valid), e.g. ['per_moon > 0']."""
    v = dict(u1=u1, u2=u2, R_star=R_star, per_bary=per_bary, a_bary=a_bary,
             r_planet=r_planet, b_bary=b_bary, w_bary=w_bary, ecc_bary=ecc_bary,
             t0_bary=t0_bary, t0_bary_offset=t0_bary_offset, M_planet=M_planet,
             r_moon=r_moon, per_moon=per_moon, tau_moon=tau_moon, Omega_moon=Omega_moon,
             i_moon=i_moon, ecc_moon=ecc_moon, w_moon=w_moon, M_moon=M_moon,
             epoch_distance=epoch_distance)
    bad = [f"{k} finite" for k, x in v.items() if not np.isfinite(float(x))]
    rules = [
        ("R_star > 0", R_star > 0), ("per_bary > 0", per_bary > 0),
        ("a_bary > 0", a_bary > 0), ("M_planet > 0", M_planet > 0),
        ("per_moon > 0", per_moon > 0), ("epoch_distance > 0", epoch_distance > 0),
        ("r_planet >= 0", r_planet >= 0), ("r_moon >= 0", r_moon >= 0),
        ("M_moon >= 0", M_moon >= 0), ("0 <= ecc_bary < 1", 0 <= ecc_bary < 1),
        ("0 <= ecc_moon < 1", 0 <= ecc_moon < 1), ("u1 >= 0", u1 >= 0),
        ("u1 + u2 <= 1", u1 + u2 <= 1 + 1e-12), ("u1 + 2 u2 >= 0", u1 + 2 * u2 >= -1e-12),
        ("supersampling_factor >= 1", supersampling_factor >= 1),
    ]
    return bad + [name for name, ok in rules if not ok]


class model_params(object):
    def __init__(self):

        # Star parameters
        self.u1 = None
        self.u2 = None
        self.R_star = None

        # Planet parameters
        self.per_bary = None
        self.a_bary = None
        self.r_planet = None
        self.b_bary = None
        self.w_bary = 0
        self.ecc_bary = 0
        self.t0_bary = None
        self.t0_bary_offset = None
        self.M_planet = None

        # Moon parameters
        self.r_moon = None
        self.per_moon = None
        self.tau_moon = None
        self.Omega_moon = None
        self.i_moon = None
        self.ecc_moon = 0
        self.w_moon = 0
        self.M_moon = None

        # Other model parameters
        self.epochs = None
        self.epoch_duration = None
        self.cadences_per_day = None
        self.epoch_distance = None
        self.supersampling_factor = 1
        # Moons with r_moon below this use the small-body approximation
        # occult_small (bug #9: 4 ppm off at k = 0.01, 105 ppm at 0.035);
        # 0 = exact Mandel-Agol for every moon (default since bug #9 was fixed)
        self.occult_small_threshold = _MODEL["occult_small_threshold"]       # settings.cfg [model]
        # moons beyond this fraction of the Hill radius are unphysical (planet-only model); the same default as
        # Likelihood and the fits (was 1.1 here until 2026-10, while docs and fits used 1.0)
        self.hill_sphere_threshold = _MODEL["hill_sphere_threshold"]


class time(object):
    def __init__(self, params):
        self.t0_bary = params.t0_bary
        self.epochs = params.epochs
        self.epoch_duration = params.epoch_duration
        self.cadences_per_day = params.cadences_per_day
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor

    def grid(self):
        return timegrid(
            self.t0_bary, 
            self.epochs, 
            self.epoch_duration, 
            self.cadences_per_day, 
            self.epoch_distance, 
            self.supersampling_factor
        )
