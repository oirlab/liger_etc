import re

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