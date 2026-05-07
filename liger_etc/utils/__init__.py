import re
import numpy as np

from liger_iris_sim.utils import LIGER_PROPS, IRIS_PROPS

def parse_resolution_from_grating(grating : str) -> float:
    """
    Parse the resolution from the grating name using regex.
    """
    return float(re.search(r'\d+\.?\d*', grating).group())


def get_instrument_prop(instrument_name : str, prop_name : str | None):
    name = instrument_name.lower()
    if name == 'liger':
        if prop_name is None:
            return LIGER_PROPS
        return LIGER_PROPS[prop_name]
    elif name == 'iris':
        if prop_name is None:
            return IRIS_PROPS
        return IRIS_PROPS[prop_name]
    raise ValueError(f"Unknown instrument: {instrument_name}")


def sci_html(val, unit_str=None, precision=3, font_size="1.1rem", sci_thresh=(1e-3, 1e3)):
    abs_val = abs(val)

    if abs_val != 0 and (abs_val < sci_thresh[0] or abs_val >= sci_thresh[1]):
        exp = int(np.floor(np.log10(abs_val)))
        coeff = val / 10**exp
        num_html = f"{coeff:.{precision}f} &times; 10<sup>{exp}</sup>"
    else:
        num_html = f"{val:.{precision}f}"

    if unit_str:
        unit_html = re.sub(r'\^(-?\d+)', lambda m: f"<sup>{m.group(1)}</sup>", unit_str)
        unit_html = f"&nbsp;&nbsp;{unit_html}"
    else:
        unit_html = ""

    return f"<span style='font-size:{font_size}'>{num_html}{unit_html}</span>"