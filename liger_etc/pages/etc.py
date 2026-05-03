import streamlit as st

from liger_etc.components.instrument_inputs import InstrumentInputs, get_instrument_params
from liger_etc.components.psf_inputs import PSFInputs, get_psf_params
from liger_etc.components.sky_inputs import SkyInputs, get_sky_params
from liger_etc.components.source_inputs import SourceInputs, get_source_params
from liger_etc.components.aperture_inputs import ApertureInputs, get_aperture_params, PSFAperturePlots
from liger_etc.components.exposure_inputs import ExposureInputs, get_exposure_params
from liger_etc.components.results import ResultsSection
from liger_etc.calc.calc_wrappers import (
    calc_snr_imager, calc_flux_imager,
    calc_snr_ifs, calc_flux_ifs,
)


def ETCPage():

    col_title, col_logos = st.columns([3, 1])
    with col_title:
        st.title("Liger Exposure Time Calculator")
    with col_logos:
        _cols = st.columns([1, 1, 1, 1])
        with _cols[0]:
            st.image("liger_etc/static/oirlab_logo.png", width=120)
        with _cols[1]:
            st.image("liger_etc/static/liger_logo.png", width=120)
        with _cols[2]:
            st.image("liger_etc/static/keck_logo.png", width=120)
        with _cols[3]:
            st.image("liger_etc/static/mulab_logo_short.png", width=120)

    InstrumentInputs()
    st.divider()

    PSFInputs()
    st.divider()

    SkyInputs()
    st.divider()

    SourceInputs()
    st.divider()

    col_exp, col_div, col_ap = st.columns([2, 0.1, 1])
    
    with col_exp:
        ExposureInputs()
    
    with col_ap:
        ApertureInputs()

    PSFAperturePlots()
    
    st.divider()

    run_calc = st.button("CALCULATE", type="primary", key="run_calculation")

    if run_calc:
        is_valid, missing = check_params()
        if not is_valid:
            st.error(f"Please fill in all required fields: {', '.join(missing)}")
            sim = None
        else:
            sim = run_sim_calc()
            st.session_state.sim = sim
    elif 'sim' in st.session_state:
        # Check if anything changed and print st.warning if so
        st.warning("Parameters changed. Click 'Calculate' to run again.")
        #sim = st.session_state.sim
        sim = None
    else:
        sim = None

    if sim is not None:
        ResultsSection(sim)

def check_params():
    
    inst_params = get_instrument_params()
    is_valid = True
    missing = []

    if inst_params['_instrument_mode'] is None:
        is_valid = False
        missing.append('Instrument Mode')

    if inst_params['_instrument_mode'] == 'IFS' and inst_params.get('grating') is None:
        is_valid = False
        missing.append('Grating')

    if inst_params.get('filter_name') is None:
        is_valid = False
        missing.append('Filter')
    
    return is_valid, missing
    
    # sky_params = get_sky_params()
    # source_params = get_source_params()
    # exposure_params = get_exposure_params()
    # aperture_params = get_aperture_params()

def run_sim_calc():

    with st.spinner("Calculating..."):

        instrument_params = get_instrument_params()
        sky_params = get_sky_params()
        source_params = get_source_params()
        exposure_params = get_exposure_params()
        aperture_params = get_aperture_params()

        calc_type = exposure_params['calc_type']
        inst_mode = instrument_params['_instrument_mode']

        if inst_mode == 'IMG':
            if calc_type == 'snr':
                return calc_snr_imager(
                    instrument_params, sky_params, source_params,
                    exposure_params, aperture_params,
                )
            elif calc_type == 'flux':
                return calc_flux_imager(
                    instrument_params, sky_params, source_params,
                    exposure_params, aperture_params,
                    desired_snr=exposure_params['desired_snr'],
                )
        elif inst_mode == 'IFS':
            if calc_type == 'snr':
                return calc_snr_ifs(
                    instrument_params, sky_params, source_params,
                    exposure_params, aperture_params,
                )
            elif calc_type == 'flux':
                return calc_flux_ifs(
                    instrument_params, sky_params, source_params,
                    exposure_params, aperture_params,
                    desired_snr=exposure_params['desired_snr'],
                )

        # IFS modes: not yet implemented
        st.warning(f"Calculation for mode '{inst_mode}' is not yet implemented.")
        return None