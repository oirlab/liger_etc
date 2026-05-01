import streamlit as st

@st.cache_resource
def get_filter_transmission_curve(filter_name : str):
    from liger_iris_drp_resources import load_filter_transmission_curve
    return load_filter_transmission_curve(filter_name)


@st.cache_resource
def get_filters_summary():
    from liger_iris_drp_resources import load_filters_summary as _load_filters_summary
    return _load_filters_summary()


@st.cache_resource
def get_gratings_summary():
    from liger_iris_drp_resources import load_gratings_summary as _load_gratings_summary
    return _load_gratings_summary()


def get_all_instrument_modes(instrument_name : str):
    if instrument_name.lower() == 'liger':
        return _get_all_liger_modes()
    elif instrument_name.lower() == 'iris':
        return _get_all_iris_modes()


@st.cache_resource
def _get_all_liger_modes():

    from liger_iris_drp_resources import make_liger_modes_table
    t = make_liger_modes_table()

    mask_img = t['MODE'] == 'IMG'
    mask_lenslet = t['IFS_MODE'] == 'LENSLET'
    mask_slicer = t['IFS_MODE'] == 'SLICER'

    return {
        'IMG' : t[mask_img],
        'LENSLET' : t[mask_lenslet],
        'SLICER' : t[mask_slicer]
    }


@st.cache_resource
def _get_all_iris_modes():
    
    from liger_iris_drp_resources import make_iris_modes_table
    t = make_iris_modes_table()

    mask_img = t['MODE'] == 'IMG'
    mask_lenslet = t['IFS_MODE'] == 'LENSLET'
    mask_slicer = t['IFS_MODE'] == 'SLICER'

    return {
        'ALL' : t,
        'IMG' : t[mask_img],
        'LENSLET' : t[mask_lenslet],
        'SLICER' : t[mask_slicer]
    }

@st.cache_data
def get_wave_grid(filter_name : str, resolution : float = 10_000.0):
    from liger_iris_sim.utils import generate_wave_grid_for_filter
    if filter_name in (None, 'None') or resolution is None:
        return None
    filter_info = get_filter_info(filter_name)
    wave = generate_wave_grid_for_filter(filter_info, resolution=resolution)
    return wave


def get_filter_info(filter_name : str):
    filter_info = get_filters_summary()[filter_name]
    return filter_info


def get_grating_info(grating_name : str):
    grating_info = get_gratings_summary()[grating_name]
    return grating_info


@st.cache_resource
def get_sky_data(instrument_params : dict, sky_params : dict):
    from liger_etc.utils.resources import get_wave_grid
    from liger_etc.components.instrument_inputs import get_resolution_from_grating
    from liger_iris_sim.sky import get_maunakea_sky_background

    # Unpack and check instrument params
    inst_mode = instrument_params.get('_instrument_mode')
    filter_name = instrument_params.get("filter_name")
    grating = instrument_params.get("grating")
    
    if inst_mode == 'IMG':
        resolution = 10_000
        assert filter_name is not None
    elif grating is not None and filter_name is not None:
        resolution = get_resolution_from_grating(grating)
    else:
        return None

    # Get the data wave grid
    wave = get_wave_grid(
        filter_name=filter_name,
        resolution=resolution
    )
    
    # Calculate the sky data
    sky_data = get_maunakea_sky_background(
        wave,
        resolution=resolution,
        T_tel=sky_params['T_tel'], T_atm=sky_params['T_atm'], T_aos=sky_params['T_aos'],
        Em_tel=sky_params['Em_tel'], Em_atm=sky_params['Em_atm'], Em_aos=sky_params['Em_aos'],
        airmass=sky_params['airmass'],
        plate_scale=instrument_params['plate_scale']
    )

    return sky_data