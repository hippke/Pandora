"""Physical constants of Pandora, from physics.cfg (read once at import)."""
from pandoramoon._config import read_cfg

_P = read_cfg("physics.cfg")["physics"]
G = float(_P["G"])          # [m^3 kg^-1 s^-2]
DAY = float(_P["day"])      # [s]
R_SUN = float(_P["R_sun"])  # [m]
M_SUN = float(_P["M_sun"])  # [kg]
M_JUP = float(_P["M_jup"])  # [kg]
