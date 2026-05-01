import streamlit as st
import re
import numpy as np
import plotly.graph_objects as go

from liger_iris_sim.expose.throughput import get_filter_throughput
from liger_iris_sim.expose.throughput import compute_throughput as _compute_throughput

from liger_etc.utils.resources import get_filters_summary, get_gratings_summary, get_all_instrument_modes, get_filter_transmission_curve, get_filter_info, get_grating_info

from liger_etc.utils import parse_resolution_from_grating

def make_transmission_curve_plot(filter_name : str):
    
    filter_info = get_filters_summary()[filter_name]
    wave, trans = get_filter_transmission_curve(filter_name)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave, y=trans, mode='lines', name=filter_name))
    fig.update_layout(
        title=dict(
            text="<b>Filter Transmission</b>",
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text="<b>Wavelength (μm)</b>",
            font=dict(size=16)
        ),
        yaxis_title=dict(
            text="<b>Transmission</b>",
            font=dict(size=16)
        ),
        template="plotly_white",
        width=400,
        height=400,
        xaxis=dict(
            tickfont=dict(size=14, weight='bold'),
        ),
        yaxis=dict(
            tickfont=dict(size=14, weight='bold'),
        )
    )

    st.plotly_chart(fig)


def InstrumentInputs():

    filter_info = get_filters_summary()
    grating_info = get_gratings_summary()

    plate_scales = {
        'Liger': {
            'IMG': [0.01],
            'IFS': {'LENSLET': [0.014, 0.031], 'SLICER': [0.075, 0.15]},
        },
        'IRIS': {
            'IMG': [0.004],
            'IFS': {'LENSLET': [0.004, 0.009], 'SLICER': [0.025, 0.05]},
        }
    }

    st.markdown('### Instrument Config')

    col_inputs1, col_inputs2, col_tput, col_curve = st.columns([1.1, 1.5, 1.5, 2])

    # ── Col 1: Instrument Inputs ───────────────────────────────────────────────
    with col_inputs1:
        #row1 = st.columns([1, 1])
        #input_rows = st.columns([1, 1, 1])
        #with row1[0]:
        if 'instrument_name' not in st.session_state:
            st.session_state.instrument_name = 'Liger'  # Default to Liger for now, since IRIS 
        instrument_name = st.session_state.instrument_name
            # instrument_name = st.radio(
            #     label='**Instrument**',
            #     options=['Liger', 'IRIS'],
            #     key='instrument_name',
            #     format_func=lambda s: 'Liger (Keck I)' if s == 'Liger' else 'IRIS (TMT)',
            # )
        _mode = st.radio(
            label='**Instrument Mode**',
            options=['Imager', 'Lenslet', 'Slicer'],
            key='instrument_mode',
            help="Select imager mode or IFS mode (lenslet or slicer)."
        )

        _instrument_mode = 'IMG' if _mode == 'Imager' else 'IFS'
        ifs_mode = None if _mode == 'Imager' else _mode.upper()

        plate_scale_options = (
            plate_scales[instrument_name][_instrument_mode]
            if _instrument_mode == 'IMG'
            else plate_scales[instrument_name][_instrument_mode][ifs_mode]
        )

        grating_options = (
            get_valid_gratings_for_mode(instrument_name=instrument_name, ifs_mode=ifs_mode)
            if _instrument_mode == 'IFS' else []
        )

        def _format_grating_option(g):
            if _instrument_mode != 'IFS':
                return 'N/A'
            info = grating_info.get(g)
            if info is None:
                return g
            return f"{g} ({np.round(info['wavemin'], decimals=3)}–{np.round(info['wavemax'], decimals=3)} μm)"

        plate_scale = st.radio(
            label='**Plate Scale (mas)**',
            options=plate_scale_options,
            format_func=lambda s: f'{int(float(s)*1000)} mas',
            key='plate_scale',
            help="Plate scale in mas."
        )

    with col_inputs2:
        if _instrument_mode == 'IMG':
            grating_options = []
            disabled = True
            placeholder = 'N/A for Imager mode'
            st.session_state.grating = None
        else:
            disabled = False
            placeholder = 'Select Grating'
        grating = st.selectbox(
            label='**Grating**',
            options=grating_options,
            key='grating',
            accept_new_options=True,
            format_func=_format_grating_option,
            disabled=disabled,
            placeholder=placeholder,
        )

        if _instrument_mode == 'IFS' and grating is None:
            filter_options = []
        else:
            filter_options = get_valid_filters_for_mode(
                instrument_name=instrument_name, instrument_mode=_instrument_mode, ifs_mode=ifs_mode, grating=grating
            )

        if len(filter_options) > 0:
            wavecens = [filter_info[f]['wavecenter'] for f in filter_options]
            sort_inds = np.argsort(wavecens)
            filter_options = [filter_options[i] for i in sort_inds]

        def _format_filter_option(f):
            info = filter_info[f]
            return f"{f} ({np.round(info['wavemin'], decimals=3)}–{np.round(info['wavemax'], decimals=3)} μm)"

        if len(filter_options) == 0 and _instrument_mode == 'IFS':
            placeholder = 'Select Grating to see filter options'
            disabled = True
            st.session_state.filter_name = None
        else:
            placeholder = 'Select Filter'
            disabled = False
            if st.session_state.get('filter_name') not in filter_options:
                st.session_state.filter_name = filter_options[0]

        filter_name = st.selectbox(
            label='**Filter**',
            key='filter_name',
            options=filter_options,
            accept_new_options=True,
            format_func=_format_filter_option,
            placeholder=placeholder,
            disabled=disabled,
        )

        size = get_num_spatial_elements_for_mode(
            instrument_name=instrument_name,
            instrument_mode=_instrument_mode,
            ifs_mode=ifs_mode,
            grating=grating,
            filter_name=filter_name
        )
        st.session_state.size = size
        _unit = 'pixels' if _instrument_mode == 'IMG' else 'spaxels'
        fov = calc_fov(float(plate_scale), size) if size is not None else None

    with col_inputs1:
        st.markdown(
            f"**Sampling:** {size[0]} × {size[1]} {_unit}" if size is not None else "**Sampling:** N/A",
            help="Number of spatial elements (pixels for imager, spaxels for IFS) in y and x directions."
        )
        st.markdown(
            f'**FoV:** {fov[0]:.2f}" × {fov[1]:.2f}"' if fov is not None else "**FoV:** N/A",
            help="Field of view in arcseconds, calculated from plate scale and number of spatial elements."
        )

    # ── Derived quantities ─────────────────────────────────────────────────────
    wavecen = filter_info[filter_name]['wavecenter'] if filter_name is not None else None

    tput_defaults = {
        'IMG': {'tel': 0.91, 'ao': 0.8, 'filt': 0.9},
        'IFS': {'tel': 0.91, 'ao': 0.8, 'filt': 0.9},
    }
    _tput_defaults = tput_defaults[_instrument_mode]

    if filter_name is not None:
        _, tput_filt = get_filter_throughput(filter_name=filter_name)
        st.session_state.tput_filt = tput_filt
        tput_inst = _compute_throughput(
            instrument_name=instrument_name,
            instrument_mode=_instrument_mode,
            wave=wavecen,
            ifs_mode=ifs_mode,
            instrument_only=True,
        )
        st.session_state.tput_inst = tput_inst
    else:
        st.session_state.tput_filt = None
        st.session_state.tput_inst = None

    # ── Col 2: Throughputs ─────────────────────────────────────────────────────
    with col_tput:
        st.markdown('**Throughput**')

        if 'tput_tel' not in st.session_state:
            st.session_state.tput_tel = _tput_defaults['tel']
        tput_tel = st.number_input(
            label='**Telescope**',
            step=0.01, min_value=0.0, max_value=1.0,
            key='tput_tel',
            placeholder=_tput_defaults['tel'],
            help="Telescope throughput"
        )

        if 'tput_ao' not in st.session_state:
            st.session_state.tput_ao = _tput_defaults['ao']
        tput_ao = st.number_input(
            label='**AO**',
            step=0.01, min_value=0.0, max_value=1.0,
            key='tput_ao',
            placeholder=_tput_defaults['ao'],
            help="AO throughput"
        )

        st.markdown(
            f"**Instrument-only:** {np.round(st.session_state.tput_inst, decimals=3)}"
            if st.session_state.tput_inst is not None else "**Instrument-only:** N/A",
            help="Wavelength dependent, instrument-only throughput. This does not include the filter."
        )

        st.markdown(
            f"**Filter:** {np.round(st.session_state.tput_filt, decimals=3)}"
            if filter_name is not None else "**Filter:** N/A",
            help="Maximum filter throughput over bandpass."
        )

        if wavecen is not None:
            tput_tot = compute_throughput(get_instrument_params())
        else:
            tput_tot = None

        st.markdown(
            f"**Total:** {np.round(tput_tot, decimals=3)}" if tput_tot is not None else "**Total:** N/A",
            help="Total throughput"
        )

        def _reset_tput_values():
            for key, value in _tput_defaults.items():
                st.session_state[f'tput_{key}'] = value

        st.button('Reset Throughput', on_click=_reset_tput_values, key='reset_tputs')

    # ── Col 3: Filter Transmission Curve ──────────────────────────────────────
    with col_curve:
        if filter_name is not None:
            make_transmission_curve_plot(filter_name)
        else:
            st.info('Select a filter to view its transmission curve.')



def get_instrument_params():
    state = st.session_state
    _instrument_mode = 'IMG' if state.instrument_mode == 'Imager' else 'IFS'
    ifs_mode = None if _instrument_mode == 'IMG' else state.instrument_mode.upper()
    filter_name = state.get('filter_name')
    if filter_name is not None:
        filter_info = get_filter_info(filter_name)
    else:
        filter_info = None
    grating_name = state.get('grating')
    if grating_name is not None:
        grating_info = get_grating_info(grating_name)
    else:
        grating_info = None
    if grating_name is not None:
        resolution = get_resolution_from_grating(grating_name)
    elif _instrument_mode == 'IMG':
        resolution = 10_000.0
    else:
        resolution = None
    params = dict(
        instrument_name=state.get('instrument_name'),
        instrument_mode=state.get('instrument_mode'),
        _instrument_mode=_instrument_mode,
        ifs_mode=ifs_mode,
        plate_scale=float(state.get('plate_scale')),
        grating=grating_name,
        filter_name=filter_name,
        resolution=resolution,
        tput_tel=state.get('tput_tel'),
        tput_ao=state.get('tput_ao'),
        tput_filt=state.get('tput_filt'),
        tput_inst=state.get('tput_inst'),
        filter_info=filter_info,
        grating_info=grating_info,
        size=state.get('size'),
        telescope_name='Keck I' if state.get('instrument_name') == 'Liger' else 'TMT',

    )
    params['tput_tot'] = compute_throughput(params)
    return params

def get_resolution_from_grating(grating : str | None = None):
    if grating is None:
        grating = st.session_state.get('grating')
    if grating is not None:
        return parse_resolution_from_grating(grating)
    return None

@st.cache_data
def get_valid_gratings_for_mode(
    instrument_name : str,
    ifs_mode : str
):
    modes = get_all_instrument_modes(instrument_name=instrument_name)
    gratings = np.unique(modes[ifs_mode.upper()]['GRATING'])
    return gratings


@st.cache_data
def get_valid_filters_for_mode(
    instrument_name : str,
    instrument_mode : str,
    grating : str | None = None,
    ifs_mode : str | None = None,
) -> list[str]:
    
    modes = get_all_instrument_modes(instrument_name=instrument_name)

    # Imager mode
    if instrument_mode == 'IMG':
        filters = np.unique(modes['IMG']['FILTER']).tolist()
        return filters
    
    # IFS mode
    if instrument_mode == 'IFS':
        if grating is None:
            return []
        else:
            _table = modes[ifs_mode.upper()]
            inds = np.where(_table['GRATING'] == grating)[0]
            return np.unique(_table['FILTER'][inds]).tolist()


@st.cache_data
def get_num_spatial_elements_for_mode(
    instrument_name : str | None = None,
    instrument_mode : str | None = None,
    ifs_mode : str | None = None,
    grating : str | None = None,
    filter_name : str | None = None,
):
    """
    Get the number of spatial elements in (y, x) for a given instrument configuration.
    """

    modes_table = get_all_instrument_modes(instrument_name=instrument_name)

    # Find valid rows to get FOV
    if instrument_mode == 'IMG':
        _table = modes_table['IMG']
        inds = np.where(
            (_table['FILTER'] == filter_name)
        )[0]
        k = inds[0]
        return _table['NUM_SPATIAL_ELEMENTS'][k]
    else:
        if grating is None or filter_name is None:
            return None
        _table = modes_table[ifs_mode.upper()]
        inds = np.where(
            (_table['GRATING'] == grating) &
            (_table['FILTER'] == filter_name)
        )[0]
        k = inds[0]
        return _table['NUM_SPATIAL_ELEMENTS'][k]


@st.cache_data
def compute_throughput(
    instrument_params : dict,
) -> float:
    """
    Compute the throughput for a given instrument configuration.
    """
    from liger_iris_sim.expose import compute_throughput as _compute_throughput
    tputs = ('tput_tel', 'tput_ao', 'tput_filt', 'tput_inst')
    if any(instrument_params.get(key) is None for key in tputs):
        return None
    else:
        return _compute_throughput(
            instrument_name=instrument_params['instrument_name'],
            instrument_mode=instrument_params['_instrument_mode'],
            wave=instrument_params['filter_info']['wavecenter'] if instrument_params['filter_info'] is not None else None,
            ifs_mode=instrument_params['ifs_mode'],
            tel=instrument_params['tput_tel'],
            ao=instrument_params['tput_ao'],
            filt=instrument_params['tput_filt'],
        )
        
@st.cache_data
def calc_fov(plate_scale : float, size : tuple[int, int]) -> tuple[float, float]:
    """
    Calculate the field of view in arcseconds given the plate scale in mas and the number of spatial elements.
    """
    fov_y = size[0] * plate_scale
    fov_x = size[1] * plate_scale
    return (fov_y, fov_x)