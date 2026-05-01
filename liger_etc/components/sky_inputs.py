import streamlit as st
import numpy as np
import scipy.constants

from liger_iris_sim.sky import get_maunakea_spectral_sky_emission, get_maunakea_spectral_sky_transmission

from liger_etc.utils.resources import get_filters_summary, get_wave_grid, get_filter_info, get_grating_info
from liger_etc.components.instrument_inputs import get_instrument_params

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from liger_etc.components.instrument_inputs import get_resolution_from_grating

from liger_etc.utils.resources import get_sky_data

sky_defaults = {
    'T_tel': 275,
    'T_atm': 258,
    'T_aos': 243,
    'T_zod': 5800,
    'Em_tel': 0.09,
    'Em_atm': 0.2,
    'Em_aos': 0.01,
    'airmass': 1.0
}

def SkyInputs():
    
    st.markdown('### Sky Config')

    input_col, plot_col = st.columns([1, 2])

    def _make_Temp_input(param : str):
        return st.number_input(
            label=r"$T_{" + param + r"}\ (K)$",
            key="T_" + param,
            step=1, min_value=0,
            value=sky_defaults[f"T_{param}"],
            placeholder=sky_defaults[f"T_{param}"]
        )
    
    def _make_Em_input(param : str):
        return st.text_input(
            label=r"$\epsilon_{" + param + r"}$",
            key="Em_" + param,
            value=sky_defaults[f"Em_{param}"],
            placeholder=sky_defaults[f"Em_{param}"]
        )

    
    with input_col:
        col_T, col_Em, col_am  = st.columns([1, 1, 1])
        with col_T:
            _make_Temp_input("tel")
            _make_Temp_input("atm")
            _make_Temp_input("aos")
        with col_Em:
            _make_Em_input("tel")
            _make_Em_input("atm")
            _make_Em_input("aos")
        with col_am:
            st.number_input(
                label="Air Mass",
                key="airmass",
                step=0.1, min_value=1.0,
                value=1.0,
                placeholder=1.0
            )

            def _reset_values():
                for key, value in sky_defaults.items():
                    if key.startswith('Em'):
                        st.session_state[key] = str(value)
                    else:
                        st.session_state[key] = value
            
            # Reset button
            st.button(
                label="**Reset Sky Params**",
                on_click=_reset_values,
                key="reset_sky"
            )

            em_log_scale = st.checkbox("Show emission in log scale", value=False)

    instrument_params = get_instrument_params()
    sky_params = get_sky_params()
    with input_col:
        SkySummary(
            instrument_params=instrument_params,
            sky_params=sky_params,
        )
    
    # Make sky plot
    with plot_col:
        SkyPlot(
            instrument_params=instrument_params,
            sky_params=sky_params,
            em_log_scale=em_log_scale
        )

    return sky_params


def get_sky_params():

    sky_params = dict(
        T_tel=float(st.session_state.T_tel),
        T_atm=float(st.session_state.T_atm),
        T_aos=float(st.session_state.T_aos),
        Em_tel=float(st.session_state.Em_tel),
        Em_atm=float(st.session_state.Em_atm),
        Em_aos=float(st.session_state.Em_aos),
        airmass=float(st.session_state.airmass)
    )

    instrument_params = get_instrument_params()

    return sky_params

_H_ERG_S = 6.626e-27   # erg·s
_C_CM_S  = 2.998e10    # cm/s

def _phot_to_erg(phot_flux, wave_um):
    """Convert phot/s/m² to erg/s/cm² per wavebin (same spatial unit)."""
    energy = _H_ERG_S * _C_CM_S / (wave_um * 1e-4)  # erg/photon
    return phot_flux * energy / 1e4                  # 1 m² = 1e4 cm²

@st.cache_data
def SkySummary(
    instrument_params : dict,
    sky_params : dict
):
    
    inst_mode = instrument_params['_instrument_mode']
    filter_name = instrument_params["filter_name"]
    resolution = instrument_params.get("resolution")
    if inst_mode == 'IMG':
       st.info(f"Showing resolution of R={resolution:,} for Imager mode.")
    
    sky_data = get_sky_data(instrument_params, sky_params)

    if sky_data is None:
        return
    
    wave = sky_data['wave']
    
    def _format_sky(value, units="", precision=3):
        formatted = f"{value:.{precision}e}"
        mantissa, exp = formatted.split("e")
        exp = int(exp)
        if units:
            return rf"${mantissa} \times 10^{{{exp}}}\ \mathrm{{{units}}}$"
        return rf"${mantissa} \times 10^{{{exp}}}$"
    
    st.markdown(f"##### Background Sky")
    if resolution is not None:
        st.markdown(f"**R = {f'{resolution:,}'}**")
        dw = wave[1] - wave[0]
        dw_ms = dw / wave[0] * scipy.constants.c / 1e3  # km/s
        st.markdown(f"**δλ = {f'{dw*1000:.2f}'} Å ({dw_ms:.1f} km/s)**")
    else:
        st.markdown(f"**Resolution: N/A**")

    if inst_mode == 'IMG':
        sky_em_tot_pix = np.sum(sky_data['sky_em'])
        plate_scale = instrument_params['plate_scale']
        sky_em_tot_spat = sky_em_tot_pix / plate_scale**2  # phot/s/m²/arcsec²
        erg_em_tot_pix  = np.sum(_phot_to_erg(sky_data['sky_em'], wave))  # erg/s/cm²/pix
        erg_em_tot_spat = erg_em_tot_pix / plate_scale**2                  # erg/s/cm²/arcsec²
        st.markdown("**Total background sky emission:**")
        st.markdown(f"**{_format_sky(sky_em_tot_spat, 'phot/s/m^2/arcsec^2')}**")
        st.markdown(f"**{_format_sky(sky_em_tot_pix, 'phot/s/m^2/pixel')}**")
        st.markdown(f"**{_format_sky(erg_em_tot_spat, 'erg/s/cm^2/arcsec^2')}**")
        st.markdown(f"**{_format_sky(erg_em_tot_pix, 'erg/s/cm^2/pixel')}**")

@st.cache_data
def SkyPlot(
    instrument_params : dict,
    sky_params : dict,
    em_log_scale : bool = True
):
    
    from liger_etc.utils.resources import get_sky_data
    sky_data = get_sky_data(instrument_params, sky_params)

    if sky_data is None:
        filter_name = instrument_params.get("filter_name")
        grating = instrument_params.get("grating")
        if filter_name is None and grating is None:
            st.info("Select a grating and filter to show the sky background in IFS mode.")
        return

    wave = sky_data['wave']

    # Compute erg/s/cm²/pixel version of sky emission for secondary y-axis
    sky_em_erg = _phot_to_erg(sky_data['sky_em'], wave)

    # Make Two row plotly figure of sky emission total and sky transmission
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        specs=[[{}], [{"secondary_y": True}]]
    )
    fig.add_trace(
        go.Scatter(
            x=wave, y=sky_data['sky_trans'],
            mode='lines',
            line=dict(color="rgba(255, 105, 180, 1.0)")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=wave, y=sky_data['sky_em'],
            mode='lines',
            line=dict(color="rgba(255, 105, 180, 1.0)"),
            name="Total"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=wave, y=sky_data['bbspec'],
            mode='lines',
            line=dict(color="royalblue"),
            name="BB"
        ),
        row=2, col=1
    )
    # Secondary y-axis: erg/s/cm²/pixel (invisible trace to drive axis scaling)
    fig.add_trace(
        go.Scatter(
            x=wave, y=sky_em_erg,
            mode='lines',
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1, secondary_y=True
    )
    fig.update_xaxes(
        title_text="<b>Wavelength (μm)</b>",
        tickfont=dict(size=14, weight='bold'),
        row=2, col=1
    )

    fig.update_yaxes(
        title_text="<b>Transmission</b>",
        tickfont=dict(size=14, weight='bold'),
        row=1, col=1
    )

    fig.update_yaxes(
        title_text="<b>γ s⁻¹ m⁻² pixel⁻¹</b>",
        tickfont=dict(size=14, weight='bold'),
        type="log" if em_log_scale else "linear",
        row=2, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="<b>erg s⁻¹ cm⁻² pixel⁻¹</b>",
        tickfont=dict(size=14, weight='bold'),
        type="log" if em_log_scale else "linear",
        row=2, col=1, secondary_y=True
    )

    fig.update_layout(
        template="plotly_white",
        width=400,
        height=600,
        showlegend=False
    )
    st.plotly_chart(fig)
    
