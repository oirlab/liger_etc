import streamlit as st
import numpy as np
import plotly.graph_objects as go

#from liger_iris_sim.utils import LIGER_PROPS, IRIS_PROPS
from liger_etc.utils import get_instrument_prop

from liger_etc.components.instrument_inputs import get_instrument_params

def get_diffrac_limit(instrument_name : str, wave : float, n : float  = 2.0) -> float:
    colldiam = get_instrument_prop(instrument_name, 'colldiam')
    return n * wave / 1E6 / colldiam * 206265 * 1E3 # mas

def ApertureInputs():

    instrument_params = get_instrument_params()

    st.markdown("### Aperture Config")

    instrument_name = instrument_params.get('instrument_name')
    #inst_mode = instrument_params.get('_instrument_mode')
    #ifs_mode = instrument_params.get('ifs_mode')
    #filter_name = instrument_params.get("filter_name")
    plate_scale = instrument_params.get('plate_scale')
    filter_info = instrument_params.get('filter_info')
    if plate_scale is not None and filter_info is not None:
        wavecen = filter_info.get('wavecenter')
        diffrac_lim = get_diffrac_limit(instrument_name, wavecen)
        st.text("Diffraction limit R = 2λ/D = " + f"{np.round(diffrac_lim, decimals=1)} mas")
    else:
        diffrac_lim = get_diffrac_limit(instrument_name, 1.5)
        st.text("Diffraction limit R = 2λ/D = N/A (Select instrument and filter)")

    aperture_rad = st.number_input(
        label='**Aperture Radius (mas)**',
        min_value=0.0,
        #max_value=None,
        value=np.round(diffrac_lim, decimals=1),
        step=0.1,
        key='aperture_rad',
        help="Aperture radius for photometry (mas)",
        placeholder=np.round(diffrac_lim, decimals=1)
    )

    num_pix = aperture_rad / (plate_scale * 1E3)
    st.markdown(f"**$N_{{pix}}$ = {np.round(num_pix, decimals=2)}**")

def _make_sersic_kernel(shape: tuple, re_pix: float, n: float) -> np.ndarray:
    """Normalized 2D Sersic surface brightness profile as a convolution kernel."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    b_n = max(2.0 * n - 1.0 / 3.0, 0.01)
    kernel = np.exp(-b_n * ((r / max(re_pix, 0.1)) ** (1.0 / n) - 1.0))
    s = kernel.sum()
    return (kernel / s).astype(np.float64) if s > 0 else kernel.astype(np.float64)


def _make_top_hat_kernel(shape: tuple, radius_pix: float) -> np.ndarray:
    """Normalized uniform-disk (top-hat) kernel for convolution."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    kernel = (r <= max(radius_pix, 0.0)).astype(np.float64)
    s = kernel.sum()
    return (kernel / s).astype(np.float64) if s > 0 else kernel.astype(np.float64)


def PSFAperturePlots():
    """
    Full-width section: encircled energy curve + log-scale PSF heatmap.
    Both plots overlay the diffraction-limit and user aperture circles.
    For extended sources, the PSF is convolved with the selected source
    profile (top-hat or Sérsic) for display.
    """
    from liger_etc.calc.calc_wrappers import get_active_psf
    from liger_etc.components.source_inputs import get_source_params
    from scipy.signal import fftconvolve

    instrument_params = get_instrument_params()
    filter_info = instrument_params.get('filter_info')
    plate_scale = instrument_params.get('plate_scale')
    instrument_name = instrument_params.get('instrument_name')

    if filter_info is None or plate_scale is None:
        return  # no filter selected yet

    aperture_rad_mas = float(st.session_state.get('aperture_rad', 0.0))
    wavecen = float(filter_info['wavecenter'])
    diffrac_lim_mas = get_diffrac_limit(instrument_name, wavecen)
    ps_mas = plate_scale * 1000.0  # mas / pix

    psf = get_active_psf(instrument_params)

    source_params = get_source_params()
    source_type = source_params.get('source_type', 'point')
    source_profile = source_params.get('source_profile')

    # ── Extended-source profile convolution (display) ─────────────────────
    psf_disp = psf.astype(np.float64)
    convolved_label = ''
    if source_type == 'extended':
        kernel = None
        if source_profile == 'sersic':
            re_arcsec = float(source_params.get('sersic_eff_radius') or 1.0)
            sn = float(source_params.get('sersic_index') or 1.0)
            kernel = _make_sersic_kernel(psf.shape, re_arcsec / plate_scale, sn)
            convolved_label = ' (⊛ Sérsic)'
        elif source_profile == 'top-hat':
            radius_arcsec = float(source_params.get('top_hat_radius') or 0.10)
            kernel = _make_top_hat_kernel(psf.shape, radius_arcsec / plate_scale)
            convolved_label = ' (⊛ Top-hat)'

        if kernel is not None:
            psf_disp = fftconvolve(psf_disp, kernel, mode='same')
            psf_disp = np.maximum(psf_disp, 0.0)
            tot = psf_disp.sum()
            if tot > 0:
                psf_disp /= tot

    ny, nx = psf_disp.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # ── Encircled Energy ──────────────────────────────────────────────────
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist_mas_2d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * ps_mas
    order = np.argsort(dist_mas_2d.ravel())
    dist_sorted = dist_mas_2d.ravel()[order]
    psf_sorted = psf_disp.ravel()[order]
    ee = np.cumsum(psf_sorted) / max(float(psf_sorted.sum()), 1e-30)

    ee_at_ap = float(np.interp(aperture_rad_mas, dist_sorted, ee))
    ee_at_dl = float(np.interp(diffrac_lim_mas, dist_sorted, ee))
    r_max = min(
        max(aperture_rad_mas * 4.0, diffrac_lim_mas * 5.0),
        float(dist_sorted[-1]),
    )

    theta = np.linspace(0, 2 * np.pi, 300)

    fig_ee = go.Figure()
    fig_ee.add_trace(go.Scatter(
        x=dist_sorted, y=ee,
        mode='lines',
        line=dict(color='royalblue', width=2),
        name='EE',
        hovertemplate='r: %{x:.1f} mas<br>EE: %{y:.3f}<extra></extra>',
    ))
    # Diffraction-limit vline + marker
    fig_ee.add_shape(type='line',
        x0=diffrac_lim_mas, x1=diffrac_lim_mas, y0=0, y1=1, yref='paper',
        line=dict(color='hotpink', dash='dot', width=1.5))
    fig_ee.add_trace(go.Scatter(
        x=[diffrac_lim_mas], y=[ee_at_dl],
        mode='markers',
        marker=dict(color='hotpink', size=10),
        name=f'2λ/D = {diffrac_lim_mas:.1f} mas  ({ee_at_dl * 100:.1f}%)',
    ))
    # User aperture vline + marker
    fig_ee.add_shape(type='line',
        x0=aperture_rad_mas, x1=aperture_rad_mas, y0=0, y1=1, yref='paper',
        line=dict(color='green', dash='dash', width=1.5))
    fig_ee.add_trace(go.Scatter(
        x=[aperture_rad_mas], y=[ee_at_ap],
        mode='markers',
        marker=dict(color='green', size=10, line=dict(color='gray', width=1)),
        name=f'Aperture = {aperture_rad_mas:.1f} mas  ({ee_at_ap * 100:.1f}%)',
    ))
    fig_ee.update_layout(
        title=dict(text=f'<b>Encircled Energy{convolved_label}</b>', font=dict(size=14)),
        xaxis=dict(
            title=dict(text='<b>Radius (mas)</b>', font=dict(size=12)),
            range=[0, r_max], tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text='<b>EE</b>', font=dict(size=12)),
            range=[0, 1.02], tickfont=dict(size=11),
        ),
        template='plotly_white',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # ── PSF Heatmap ───────────────────────────────────────────────────────
    crop_mas = max(aperture_rad_mas * 3.0, diffrac_lim_mas * 4.0)
    crop_pix = max(int(np.ceil(crop_mas / ps_mas)) + 2, 15)
    ix0 = max(0, int(round(cx)) - crop_pix)
    ix1 = min(nx, int(round(cx)) + crop_pix + 1)
    iy0 = max(0, int(round(cy)) - crop_pix)
    iy1 = min(ny, int(round(cy)) + crop_pix + 1)

    psf_crop = psf_disp[iy0:iy1, ix0:ix1]
    x_mas = (np.arange(ix0, ix1) - cx) * ps_mas
    y_mas = (np.arange(iy0, iy1) - cy) * ps_mas
    psf_log = np.log10(np.maximum(psf_crop / max(float(psf_crop.max()), 1e-30), 1e-4))
    _span = (crop_pix + 0.5) * ps_mas

    _L, _R, _T, _B = 55, 110, 55, 45
    _INNER = 330
    fig_psf = go.Figure()
    fig_psf.add_trace(go.Heatmap(
        x=x_mas, y=y_mas, z=psf_log,
        colorscale='Inferno',
        colorbar=dict(
            title=dict(text='log₁₀(PSF)', side='right', font=dict(size=11)),
            tickvals=[-4, -3, -2, -1, 0],
            ticktext=['-4', '-3', '-2', '-1', '0'],
            thickness=12,
        ),
        zmin=-4, zmax=0,
        hovertemplate='Δx: %{x:.1f} mas<br>Δy: %{y:.1f} mas<br>log₁₀: %{z:.2f}<extra></extra>',
    ))
    #x0 = (nx - 1) / 2 * plate_scale * 1000
    #y0 = (ny - 1) / 2 * plate_scale * 1000
    #x0 = 0.5 * ps_mas if nx % 2 == 1 else 0.0
    #y0 = 0.5 * ps_mas if ny % 2 == 1 else 0.0
    #breakpoint()
    y0 = 0.0
    x0 = 0.0
    fig_psf.add_trace(go.Scatter(
        #x=diffrac_lim_mas * np.cos(theta),
        #y=diffrac_lim_mas * np.sin(theta),
        x=x0 + diffrac_lim_mas * np.cos(theta),
        y=y0 + diffrac_lim_mas * np.sin(theta),
        mode='lines',
        line=dict(color='hotpink', width=1.5),
        name=f'2λ/D ({diffrac_lim_mas:.1f} mas)',
        hoverinfo='skip',
    ))
    fig_psf.add_trace(go.Scatter(
        #x=aperture_rad_mas * np.cos(theta),
        #y=aperture_rad_mas * np.sin(theta),
        x=x0 + aperture_rad_mas * np.cos(theta),
        y=y0 + aperture_rad_mas * np.sin(theta),
        mode='lines',
        line=dict(color='green', width=1.5, dash='dash'),
        name=f'Aperture ({aperture_rad_mas:.1f} mas)',
        hoverinfo='skip',
    ))
    fig_psf.update_layout(
        title=dict(text=f'<b>PSF{convolved_label}</b> — log scale', font=dict(size=14)),
        xaxis=dict(
            title=dict(text='<b>Δx (mas)</b>', font=dict(size=12)),
            range=[-_span, _span], autorange=False, tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text='<b>Δy (mas)</b>', font=dict(size=12)),
            scaleanchor='x', scaleratio=1,
            range=[-_span, _span], autorange=False, tickfont=dict(size=11),
        ),
        margin=dict(l=_L, r=_R, t=_T, b=_B),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        autosize=False,
        width=_INNER + _L + _R,
        height=_INNER + _T + _B,
    )

    col_ee, col_psf = st.columns([1.1, 1])
    with col_ee:
        st.plotly_chart(fig_ee, width='content')
    with col_psf:
        st.plotly_chart(fig_psf, width='content')


def get_aperture_params():
    instrument_params = get_instrument_params()
    instrument_name = instrument_params.get('instrument_name')
    #wavecen = instrument_params.get('filter_info', {}).get('wavecenter')
    filter_info = instrument_params.get('filter_info')
    if filter_info is not None:
        aperture_rad_diff_lim = get_diffrac_limit(
            instrument_name,
            instrument_params['filter_info']['wavecenter']
        )
    else:
        aperture_rad_diff_lim = None
    aperture_params = {
        'aperture_rad': st.session_state.aperture_rad,
        'aperture_rad_diff_lim': aperture_rad_diff_lim,
    }
    return aperture_params