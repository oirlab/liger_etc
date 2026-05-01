import numpy as np
import plotly.graph_objects as go
import streamlit as st

from liger_etc.components.instrument_inputs import get_instrument_params

_KAPA_URL = "http://altair.dyn.berkeley.edu:8501/"

_LOG_FLOOR = 1e-4  # relative floor for log10 display
_HALF_SPAN_MAS = 500.0


def _psf_preview_plots(psf: np.ndarray, plate_scale: float, inst_mode: str) -> tuple[go.Figure, go.Figure]:
    """
        Two standalone Plotly figures:
            1) PSF heatmap
            2) x-slice (blue) and y-slice (orange) through the centre
    Axes are labelled in arcsec (IMG) or spaxels (IFS).
    """
    ny, nx = psf.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Spatial axes in arcsec
    x_as = (np.arange(nx) - cx) * plate_scale * 1000  # Convert from arcsec to mas
    y_as = (np.arange(ny) - cy) * plate_scale * 1000

    axis_label = 'Δ (mas)'

    psf_norm = psf / max(float(psf.max()), 1e-30)
    psf_log  = np.log10(np.maximum(psf_norm, _LOG_FLOOR))

    ix = int(round(cx))
    iy = int(round(cy))
    slice_x = np.maximum(psf_norm[iy, :], _LOG_FLOOR)
    slice_y = np.maximum(psf_norm[:, ix], _LOG_FLOOR)

    # ── Heatmap figure ───────────────────────────────────────────────────────
    fig_psf = go.Figure()
    fig_psf.add_trace(go.Heatmap(
        x=x_as, y=y_as, z=psf_log,
        colorscale='Inferno',
        colorbar=dict(
            title=dict(text='Relative Intensity', side='right', font=dict(size=11, weight='bold')),
            tickvals=[-4, -3, -2, -1, 0],
            ticktext=['1e-4', '1e-3', '1e-2', '1e-1', '1'],
            tickfont=dict(size=14, weight='bold'),
            thickness=12,
            len=0.86,
            x=1.02,
            xanchor='left',
            y=0.5,
        ),
        zmin=-4, zmax=0,
        customdata=np.maximum(psf_norm, _LOG_FLOOR),
        hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Relative Intensity: %{customdata:.3e}<extra></extra>',
    ))

    fig_psf.update_xaxes(
        title_text=f'<b>{axis_label}</b>',
        range=[-_HALF_SPAN_MAS, _HALF_SPAN_MAS],
        tickfont=dict(size=14, weight='bold'),
    )
    fig_psf.update_yaxes(
        title_text=f'<b>{axis_label}</b>',
        range=[-_HALF_SPAN_MAS, _HALF_SPAN_MAS],
        scaleanchor='x',
        scaleratio=1,
        tickfont=dict(size=14, weight='bold'),
    )
    fig_psf.update_layout(
        title=dict(text='<b>PSF</b>', font=dict(size=16)),
        template='plotly_white',
        height=380,
        margin=dict(l=60, r=95, t=55, b=50),
        showlegend=False,
    )

    # ── Slice figure ─────────────────────────────────────────────────────────
    fig_slice = go.Figure()
    fig_slice.add_trace(go.Scatter(
        x=x_as, y=slice_x,
        mode='lines', line=dict(color='royalblue', width=2),
        name='x-slice',
        hovertemplate='x: %{x:.3f}<br>Relative Intensity: %{y:.3e}<extra></extra>',
    ))
    fig_slice.add_trace(go.Scatter(
        x=y_as, y=slice_y,
        mode='lines', line=dict(color='darkorange', width=2),
        name='y-slice',
        hovertemplate='y: %{x:.3f}<br>Relative Intensity: %{y:.3e}<extra></extra>',
    ))

    fig_slice.update_xaxes(
        title_text=f'<b>{axis_label}</b>',
        range=[-_HALF_SPAN_MAS, _HALF_SPAN_MAS],
        tickfont=dict(size=14, weight='bold'),
    )
    fig_slice.update_yaxes(
        title_text='<b>Relative Intensity</b>',
        type='log',
        range=[np.log10(_LOG_FLOOR), 0.0],
        tickfont=dict(size=14, weight='bold'),
    )
    fig_slice.update_layout(
        title=dict(text='<b>Central slices</b>', font=dict(size=16)),
        template='plotly_white',
        height=380,
        margin=dict(l=60, r=20, t=55, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1),
        showlegend=True,
    )
    return fig_psf, fig_slice


def PSFInputs():
    instrument_params = get_instrument_params()
    filter_info = instrument_params.get('filter_info')
    inst_mode = instrument_params['_instrument_mode']
    plate_scale = instrument_params.get('plate_scale')

    st.markdown('### PSF')

    col_inputs, col_heatmap, col_slices = st.columns([1.0, 1.15, 1.15])

    with col_inputs:
        psf_option = st.radio(
            label='**PSF Source**',
            options=['Default PSF', 'Analytic PSF'],
            key='psf_option',
            horizontal=True,
        )

        if psf_option == 'Analytic PSF':
            st.number_input(
                label='**Strehl Ratio**',
                min_value=0.01,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format='%.2f',
                key='psf_strehl',
            )
            st.markdown(f'[KAPA Strehl Calculator]({_KAPA_URL})')
            st.number_input(
                label='**Fried Parameter r₀ (cm)**',
                min_value=0.1,
                max_value=200.0,
                value=40.0,
                step=0.5,
                format='%.2f',
                key='psf_fried_param',
            )

            # Show wavelength info — auto, not user-controlled
            if filter_info is not None:
                wmin = filter_info['wavemin']
                wcen = filter_info['wavecenter']
                wmax = filter_info['wavemax']
                if inst_mode == 'IMG':
                    st.caption(
                        f'PSF averaged over bandpass: '
                        f'{wmin:.4g} μm,  {wcen:.4g} μm,  {wmax:.4g} μm'
                    )
                else:
                    st.caption(
                        f'Monochromatic PSF at central wavelength: {wcen:.4g} μm'
                    )
            else:
                st.caption('Select a filter to see PSF wavelength details.')

    # ── PSF preview plots shown in separate side-by-side columns ───────────
    if filter_info is not None and plate_scale is not None:
        from liger_etc.calc.calc_wrappers import get_active_psf
        with st.spinner('Loading PSF…'):
            psf = get_active_psf(instrument_params)
        fig_psf, fig_slice = _psf_preview_plots(psf, float(plate_scale), inst_mode)
        with col_heatmap:
            st.plotly_chart(fig_psf, width='content')
        with col_slices:
            st.plotly_chart(fig_slice, width='content')


def get_psf_params() -> dict:
    state = st.session_state
    return dict(
        psf_option=state.get('psf_option', 'Default PSF'),
        strehl=float(state.get('psf_strehl', 0.50)),
        fried_param=float(state.get('psf_fried_param', 40.0)),
    )
