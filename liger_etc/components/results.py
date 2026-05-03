import numpy as np
import streamlit as st
import plotly.graph_objects as go

from liger_etc.components.instrument_inputs import get_instrument_params
from liger_etc.components.psf_inputs import get_psf_params
from liger_etc.components.sky_inputs import get_sky_params
from liger_etc.components.source_inputs import get_source_params
from liger_etc.components.aperture_inputs import get_aperture_params
from liger_etc.components.exposure_inputs import get_exposure_params
from liger_etc.calc.calc_wrappers import get_active_psf, get_psf
from liger_etc.utils import get_instrument_prop
from liger_etc.utils.download_results import download_results


def _get_psf_export(instrument_params: dict, psf_params: dict) -> tuple[np.ndarray | None, dict]:
    """Build PSF array and metadata dict for result export."""
    try:
        psf = get_active_psf(instrument_params)
    except Exception as exc:
        return None, {'error': str(exc)}

    filter_info = instrument_params.get('filter_info') or {}
    mode = instrument_params.get('_instrument_mode')
    option = psf_params.get('psf_option', 'default')
    wcen = float(filter_info.get('wavecenter')) if filter_info.get('wavecenter') is not None else None

    psf_info = {
        'psf_option': option,
        'instrument_name': instrument_params.get('instrument_name'),
        'instrument_mode': mode,
        'filter_name': instrument_params.get('filter_name'),
        'plate_scale': instrument_params.get('plate_scale'),
        'wavecenter_um': wcen,
        'shape': tuple(int(v) for v in psf.shape),
    }

    if option == 'analytic':
        wave_used = [wcen] if wcen is not None else []
        if mode == 'IMG':
            wmin = filter_info.get('wavemin')
            wmax = filter_info.get('wavemax')
            if wmin is not None and wcen is not None and wmax is not None:
                wave_used = [float(wmin), float(wcen), float(wmax)]
        psf_info.update({
            'model': 'analytic_psf',
            'strehl': psf_params.get('strehl'),
            'fried_param': psf_params.get('fried_param'),
            'wave_used_um': wave_used,
        })
    else:
        lib_info = {}
        if wcen is not None:
            try:
                _, lib_info = get_psf(
                    instrument_name=instrument_params['instrument_name'],
                    instrument_mode=instrument_params['_instrument_mode'],
                    wave=wcen,
                    plate_scale=instrument_params['plate_scale'],
                )
            except Exception as exc:
                lib_info = {'error': str(exc)}
        psf_info.update({
            'model': 'library_psf',
            'library_info': lib_info,
        })

    return psf, psf_info


def ResultsSection(sim: dict):
    exposure_params = get_exposure_params()
    instrument_params = get_instrument_params()
    psf_params = get_psf_params()
    sky_params = get_sky_params()
    source_params = get_source_params()
    aperture_params = get_aperture_params()
    psf, psf_info = _get_psf_export(instrument_params, psf_params)

    inst_mode = instrument_params['_instrument_mode']
    calc_type = exposure_params['calc_type']

    if inst_mode == 'IMG':
        if calc_type == 'snr':
            ImagerResults_SNR(
                sim,
                instrument_params,
                sky_params,
                source_params,
                exposure_params
            )
        elif calc_type == 'flux':
            ImagerResults_flux(
                sim,
                instrument_params,
                sky_params,
                source_params,
                exposure_params
            )
    elif inst_mode == 'IFS':
        if calc_type == 'snr':
            IFSResults_SNR(sim, instrument_params, sky_params, source_params, exposure_params)
        elif calc_type == 'flux':
            IFSResults_flux(sim, instrument_params, sky_params, source_params, exposure_params)

    st.divider()
    download_results(
        payload={
            'meta': {},
            'params': {
                'instrument': instrument_params,
                'psf': psf_params,
                'psf_info': psf_info,
                'sky': sky_params,
                'source': source_params,
                'exposure': exposure_params,
                'aperture': aperture_params,
            },
            'sim': sim,
            'psf': psf,
        }
    )


def _check_sim(sim):
    if sim is None:
        st.error("No results available. Run the calculation first.")
        return False
    if 'error' in sim:
        st.error(sim['error'])
        return False
    return True


def SNR_ITIME_PLOT(
    S_ap: float,
    B_ap: float,
    read_noise: float,
    N_pix_ap: float,
    n_frames: int,
    itime_current: float,
    aperture_rad_mas: float,
):
    """
    Plot aperture SNR vs total integration time analytically.

    Formula (n_frames fixed, itime varies):
        SNR(T) = S_ap * T / sqrt((S_ap + B_ap) * T + N_pix_ap * RN² * n_frames)
    where T = itime * n_frames.
    """
    T_current = itime_current * n_frames
    T_min = max(1.0, T_current / 100.0)
    #T_max = T_current * 100.0
    T_max = T_current * 10.0
    T_arr = np.logspace(np.log10(T_min), np.log10(T_max), 300)

    RN_var = N_pix_ap * read_noise ** 2 * n_frames
    snr_arr = S_ap * T_arr / np.sqrt((S_ap + B_ap) * T_arr + RN_var)
    snr_current = S_ap * T_current / np.sqrt((S_ap + B_ap) * T_current + RN_var)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T_arr, y=snr_arr,
        mode='lines',
        name='SNR',
        line=dict(color='royalblue', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[T_current], y=[snr_current],
        mode='markers',
        name=f'Current ({T_current:.0f} s)',
        marker=dict(color='crimson', size=12, symbol='circle'),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"<b>SNR vs Integration Time</b>"
                f"<br><sup>Aperture R = {aperture_rad_mas:.1f} mas, {n_frames} frame(s)</sup>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="<b>Total Integration Time (s)</b>", font=dict(size=14)),
            type='log',
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="<b>Aperture SNR</b>", font=dict(size=14)),
            tickfont=dict(size=14),
        ),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
    )
    st.plotly_chart(fig, width='stretch')


def SNRHeatmapPlot(
    snr_image: np.ndarray,
    plate_scale: float,
    xpix: int,
    ypix: int,
    aperture_rad_mas: float,
    dl_mas: float | None = None,
    title: str = 'SNR Map',
):
    """
    Interactive heatmap of a 50×50 pixel cutout centred on the source.
    Axes are labelled in mas relative to the source position.
    User aperture drawn as a dashed white circle; optional diffraction-limit
    aperture drawn as a solid pink annulus.
    """
    half = 25
    ny, nx = snr_image.shape
    x0 = max(0, xpix - half)
    x1 = min(nx, xpix + half + 1)   # +1 so arange gives -half…+half (51 pixels, symmetric)
    y0 = max(0, ypix - half)
    y1 = min(ny, ypix + half + 1)

    cutout = snr_image[y0:y1, x0:x1]
    x_mas = (np.arange(x0, x1) - xpix) * plate_scale * 1000.0
    y_mas = (np.arange(y0, y1) - ypix) * plate_scale * 1000.0

    # Symmetric range: outer edge of the ±half pixels, centred exactly on 0
    half_span = (half + 0.5) * plate_scale * 1000.0
    x_range = [-half_span, half_span]
    y_range = [-half_span, half_span]

    theta = np.linspace(0, 2 * np.pi, 300)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_mas,
        y=y_mas,
        z=cutout,
        colorscale='Viridis',
        colorbar=dict(title=dict(text='SNR', side='right')),
        hovertemplate='Δx: %{x:.1f} mas<br>Δy: %{y:.1f} mas<br>SNR: %{z:.2f}<extra></extra>',
    ))
    if dl_mas is not None:
        fig.add_trace(go.Scatter(
            x=dl_mas * np.cos(theta),
            y=dl_mas * np.sin(theta),
            mode='lines',
            line=dict(color='hotpink', width=1.5),
            name=f'2λ/D (R={dl_mas:.0f} mas)',
            hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter(
        x=aperture_rad_mas * np.cos(theta),
        y=aperture_rad_mas * np.sin(theta),
        mode='lines',
        line=dict(color='white', width=1.5, dash='dash'),
        name=f'Aperture (R={aperture_rad_mas:.0f} mas)',
        hoverinfo='skip',
    ))
    # Make the inner plot area square by deriving width from height + margins.
    # No scaleanchor needed: equal data spans + square plot area = square pixels.
    _L, _R, _T, _B = 60, 120, 70, 50   # left, right, top, bottom margins (px)
    _INNER = 370                         # square inner plot area side (px)
    fig_w = _INNER + _L + _R            # 550
    fig_h = _INNER + _T + _B            # 490

    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=16)),
        xaxis=dict(
            title=dict(text='<b>\u0394x (mas)</b>', font=dict(size=14)),
            range=x_range,
            autorange=False,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text='<b>\u0394y (mas)</b>', font=dict(size=14)),
            scaleanchor='x',
            scaleratio=1,
            range=y_range,
            autorange=False,
            tickfont=dict(size=14),
        ),
        margin=dict(l=_L, r=_R, t=_T, b=_B),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        autosize=False,
        width=fig_w,
        height=fig_h,
    )
    st.plotly_chart(fig, width='content')


def ImagerResults_SNR(
    sim: dict,
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
):
    st.markdown("### Results")
    if not _check_sim(sim):
        return
    
    instrument_name = instrument_params.get('instrument_name')

    itime      = float(exposure_params['input_itime'])
    n_frames   = int(exposure_params['input_n_frames'])
    read_noise = float(exposure_params['read_noise'])

    S_ap             = sim['S_ap']
    B_ap             = sim['B_ap']
    N_pix_ap         = sim['N_pix_ap']
    aperture_rad_mas = sim['aperture_user']
    dl_mas           = sim.get('aperture_diff_lim')

    filter_name      = instrument_params.get('filter_name', 'N/A')
    filter_info      = instrument_params.get('filter_info') or {}
    # instrument_label = (
    #     'IRIS (TMT)' if instrument_params.get('instrument_name', '').lower() == 'iris'
    #     else 'Liger (Keck)'
    # )

    col_metrics, col_plot = st.columns([1, 2])

    with col_metrics:
        # ── Setup summary ──────────────────────────────────────────────────────
        st.markdown("##### Summary")
        st.markdown(f"**Instrument:** {instrument_name}")
        collarea = get_instrument_prop(instrument_name, 'collarea')
        colldiam = get_instrument_prop(instrument_name, 'colldiam')
        st.markdown(f"**Telescope:** {instrument_params.get('telescope_name')} (D={colldiam:.2f} m, A={collarea:.1f} m²)")
        if instrument_params.get('_instrument_mode') == 'IFS':
            st.markdown(f"**Spec. Res:** {instrument_params.get('resolution', 'N/A')}")
        if filter_info:
            st.markdown("**Filter:** " + filter_name + f" ({filter_info.get('wavemin', 0):.3f}–{filter_info.get('wavemax', 0):.3f} μm)")
        st.markdown(f"**Itime:** {itime:.1f} s × {n_frames} = {itime * n_frames:.1f} s")

        flux_method = source_params.get('flux_method', 'mag_vega')
        if flux_method == 'mag_vega':
            st.markdown(f"**Mag (Vega):** {source_params.get('mag_vega', 0):.2f}")
        elif flux_method == 'flux_tot':
            st.markdown(f"**Flux:** {source_params.get('flux_tot', 0):.3e} phot/s/m²")
        elif flux_method == 'flux_density':
            st.markdown(f"**Flux Density:** {source_params.get('flux_density', 0):.3e} phot/s/m²/μm")

        st.divider()

        # ── SNR metrics ────────────────────────────────────────────────────────
        st.markdown("##### SNR")
        scale = int(instrument_params['plate_scale'] * 1000)
        st.metric(f"Peak SNR ({scale} mas)", f"{sim['snr_peak']:.1f}", help="The peak pixel SNR over the image.")
        st.metric(
            f"Aperture SNR  (R = {aperture_rad_mas:.1f} mas)",
            f"{sim['snr_ap_user']:.1f}",
            help=f"SNR integrated within the user-defined aperture.",
        )
        if dl_mas is not None and sim.get('snr_ap_diff_lim') is not None:
            st.metric(
                f"Aperture SNR  (R = 2λ/D = {dl_mas:.1f} mas)",
                f"{sim['snr_ap_diff_lim']:.1f}",
                help=f"SNR integrated within the diffraction-limit aperture (2λ/D)."
            )

        st.divider()

        # ── Signal budget ──────────────────────────────────────────────────────
        xpix, ypix = sim['xpix'], sim['ypix']
        src_peak_rate = float(sim['source_rate'][ypix, xpix])
        sky_rate_pix  = float(sim['sky_em_rate'])
        dark_rate_pix = float(sim['dark_rate'])
        st.markdown("##### Signal Budget", help="Electron count rates for the source and noise contributions.")
        st.markdown(
            f"| | Rate (e⁻/s) |\n"
            f"|---|---|\n"
            f"| **Source** (peak pixel) | {src_peak_rate:.3f} |\n"
            f"| **Source** (aperture) | {S_ap:.3f} |\n"
            f"| **Sky** (per pixel) | {sky_rate_pix:.4f} |\n"
            f"| **Dark** (per pixel) | {dark_rate_pix:.4f} |\n"
            f"| **Background** (aperture) | {B_ap:.4f} |\n"
            f"| **Read Noise** (per pixel) | {read_noise:.2f} |"
            f"| **Read Noise** (aperture) | {(read_noise * np.sqrt(N_pix_ap)):.2f} |"
        )

    with col_plot:
        SNR_ITIME_PLOT(
            S_ap=S_ap,
            B_ap=B_ap,
            read_noise=read_noise,
            N_pix_ap=N_pix_ap,
            n_frames=n_frames,
            itime_current=itime,
            aperture_rad_mas=aperture_rad_mas,
        )
        snr_2d = np.array(sim['snr'], dtype=float)
        SNRHeatmapPlot(
            snr_image=snr_2d,
            plate_scale=instrument_params['plate_scale'],
            xpix=sim['xpix'],
            ypix=sim['ypix'],
            aperture_rad_mas=aperture_rad_mas,
            dl_mas=dl_mas,
            title='SNR Map',
        )


def ImagerResults_flux(
    sim: dict,
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
):
    st.markdown("### Results")
    if not _check_sim(sim):
        return

    T = sim['total_time']
    filter_info = instrument_params.get('filter_info') or {}
    wave_eff_um = float(filter_info.get('wavecenter', 1.65))
    # 1 phot/s/m² = (hc/λ) J/phot × 1e3 erg/J/cm² → erg/s/cm²
    phot_to_erg = 1.986e-16 / wave_eff_um  # erg s cm² / (phot/s/m²)
    erg_flux_lim = sim['photon_flux_lim'] * phot_to_erg
    st.markdown(
        f"For **{T:.1f} s** total integration, "
        f"SNR ≥ {sim['desired_snr']:.1f} in aperture (R = {sim['aperture_user']:.1f} mas):"
    )
    st.metric("Limiting Magnitude (Vega)", f"{sim['mag_lim']:.2f}")
    st.metric("Limiting Photon Flux", f"{sim['photon_flux_lim']:.3e} phot/s/m²")
    st.metric("Limiting Flux", f"{erg_flux_lim:.3e} erg/s/cm²")


# ── IFS plot helpers ─────────────────────────────────────────────────────────

def ObservedSpectrumPlot(
    wave: np.ndarray,
    obs_spec: np.ndarray,
    src_spec: np.ndarray,
    noise_spec: np.ndarray,
    aperture_rad_mas: float,
    title_suffix: str = '',
):
    """
    Aperture-integrated observed spectrum with ±1σ error bars.
    Also overlays the noiseless source spectrum for reference.
    """
    fig = go.Figure()

    # Noiseless source
    fig.add_trace(go.Scatter(
        x=wave, y=src_spec,
        mode='lines',
        name='Source (noiseless)',
        line=dict(color='royalblue', width=1.5, dash='dot'),
    ))

    # Observed with error band
    fig.add_trace(go.Scatter(
        x=np.concatenate([wave, wave[::-1]]),
        y=np.concatenate([obs_spec + noise_spec, (obs_spec - noise_spec)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,100,100,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='±1σ',
    ))
    fig.add_trace(go.Scatter(
        x=wave, y=obs_spec,
        mode='lines',
        name='Observed',
        line=dict(color='crimson', width=1.5),
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Aperture Spectrum (R = {aperture_rad_mas:.1f} mas"
                f"{' — ' + title_suffix if title_suffix else ''})</b>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text='<b>Wavelength (μm)</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text='<b>Flux (e⁻)</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
    )
    st.plotly_chart(fig, width='stretch')


def SNRSpectrumPlot(
    wave: np.ndarray,
    snr_spec: np.ndarray,
    aperture_rad_mas: float,
    snr_spec_dl: np.ndarray | None = None,
    dl_mas: float | None = None,
):
    """Per-channel aperture SNR spectrum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wave, y=snr_spec,
        mode='lines',
        name=f'SNR (R={aperture_rad_mas:.1f} mas)',
        line=dict(color='royalblue', width=2),
    ))
    if snr_spec_dl is not None and dl_mas is not None:
        fig.add_trace(go.Scatter(
            x=wave, y=snr_spec_dl,
            mode='lines',
            name=f'SNR (R=2λ/D={dl_mas:.1f} mas)',
            line=dict(color='darkorange', width=1.5, dash='dash'),
        ))
    fig.update_layout(
        title=dict(text='<b>SNR Spectrum</b>', font=dict(size=16)),
        xaxis=dict(
            title=dict(text='<b>Wavelength (μm)</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text='<b>SNR per channel</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=380,
    )
    st.plotly_chart(fig, width='stretch')


# ── IFS result panels ─────────────────────────────────────────────────────────

def _ifs_summary_col(
    sim: dict,
    instrument_params: dict,
    source_params: dict,
    exposure_params: dict,
    extra_metrics: list[tuple[str, str]] | None = None,
):
    """Shared left-column summary for all IFS result modes."""
    instrument_name = instrument_params.get('instrument_name')
    filter_name     = instrument_params.get('filter_name', 'N/A')
    filter_info     = instrument_params.get('filter_info') or {}
    itime           = float(exposure_params['input_itime'])
    n_frames        = int(exposure_params['input_n_frames'])

    st.markdown("##### Summary")
    st.markdown(f"**Instrument:** {instrument_name}")
    try:
        collarea = get_instrument_prop(instrument_name, 'collarea')
        colldiam = get_instrument_prop(instrument_name, 'colldiam')
        st.markdown(f"**Telescope:** {instrument_params.get('telescope_name')} (D={colldiam:.2f} m, A={collarea:.1f} m²)")
    except Exception:
        pass
    st.markdown(f"**Spec. Res:** {instrument_params.get('resolution', 'N/A')}")
    if filter_info:
        st.markdown("**Filter:** " + filter_name + f" ({filter_info.get('wavemin', 0):.3f}–{filter_info.get('wavemax', 0):.3f} μm)")
    st.markdown(f"**Itime:** {itime:.1f} s × {n_frames} = {itime * n_frames:.1f} s")

    if exposure_params.get('calc_type') != 'flux':
        flux_method = source_params.get('flux_method', 'mag_vega')
        if flux_method == 'mag_vega':
            st.markdown(f"**Mag (Vega):** {source_params.get('mag_vega', 0):.2f}")
        elif flux_method == 'flux_tot':
            st.markdown(f"**Flux:** {source_params.get('flux_tot', 0):.3e} phot/s/m²")
        elif flux_method == 'flux_density':
            st.markdown(f"**Flux Density:** {source_params.get('flux_density', 0):.3e} phot/s/m²/μm")

    st.divider()

    if extra_metrics:
        for label, value, _help in extra_metrics:
            st.metric(label, value, help=_help)


def IFSResults_SNR(
    sim: dict,
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
):
    st.markdown('### Results')
    if not _check_sim(sim):
        return

    wave             = sim['wave']
    snr_spec         = sim['snr_spec']
    snr_spec_dl      = sim.get('snr_spec_dl')
    src_spec         = sim['src_spec']
    noise_spec       = sim['noise_spec']
    obs_spec         = sim['obs_spec']
    aperture_rad_mas = sim['aperture_user']
    dl_mas           = sim.get('aperture_diff_lim')
    itime            = float(exposure_params['input_itime'])
    n_frames         = int(exposure_params['input_n_frames'])

    median_snr    = float(np.nanmedian(snr_spec))
    peak_snr      = float(np.nanmax(snr_spec))
    peak_snr_dl   = float(np.nanmax(snr_spec_dl)) if snr_spec_dl is not None else None

    col_left, col_right = st.columns([1, 2])

    with col_left:
        _ifs_summary_col(
            sim, instrument_params, source_params, exposure_params,
            extra_metrics=[
                ('Peak SNR', f'{peak_snr:.1f}', 'Maximum signal-to-noise ratio across bandpass'),
                (f'Median SNR', f'{median_snr:.1f}', 'Median signal-to-noise ratio across bandpass'),
                (f'Aperture R = {aperture_rad_mas:.1f} mas', '', 'Aperture radius in milliarcseconds'),
            ] + ([
                (f'Peak SNR (2λ/D = {dl_mas:.1f} mas)', f'{peak_snr_dl:.1f}', 'Maximum signal-to-noise ratio at diffraction limit over bandpass'),
            ] if peak_snr_dl is not None else [])
        )

    with col_right:
        ObservedSpectrumPlot(
            wave=wave,
            obs_spec=obs_spec,
            src_spec=src_spec,
            noise_spec=noise_spec,
            aperture_rad_mas=aperture_rad_mas,
        )
        SNRSpectrumPlot(
            wave=wave,
            snr_spec=snr_spec,
            aperture_rad_mas=aperture_rad_mas,
            snr_spec_dl=snr_spec_dl,
            dl_mas=dl_mas,
        )
        #snr_cube = np.array(sim['snr'], dtype=float)
        snr_2d = np.nansum(sim['snr']**2, axis=0)**0.5
        SNRHeatmapPlot(
            snr_image=snr_2d,
            plate_scale=instrument_params['plate_scale'],
            xpix=sim['xpix'],
            ypix=sim['ypix'],
            aperture_rad_mas=aperture_rad_mas,
            dl_mas=dl_mas,
            title='SNR integrated over bandpass',
        )


def IFSResults_flux(
    sim: dict,
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
):
    st.markdown('### Results')
    if not _check_sim(sim):
        return

    wave             = sim['wave']
    flux_lim_density = sim['flux_lim_density']
    mag_lim          = sim['mag_lim']
    T                = sim['total_time']
    desired_snr      = sim['desired_snr']
    aperture_rad_mas = sim['aperture_user']

    col_left, col_right = st.columns([1, 2])

    with col_left:
        _ifs_summary_col(sim, instrument_params, source_params, exposure_params)
        wave_eff_um = float((instrument_params.get('filter_info') or {}).get('wavecenter', 1.65))
        phot_to_erg = 1.986e-16 / wave_eff_um
        erg_flux_lim = sim['photon_flux_lim'] * phot_to_erg
        st.markdown(
            f"For **{T:.1f} s** total integration, SNR ≥ {desired_snr:.1f} "
            f"per channel (R = {aperture_rad_mas:.1f} mas):"
        )
        st.metric('Limiting Magnitude (Vega, continuum)', f'{mag_lim:.2f}')
        st.metric('Limiting Flux', f"{sim['photon_flux_lim']:.3e} phot/s/m²")
        st.metric('Limiting Flux (erg/s/cm²)', f"{erg_flux_lim:.3e} erg/s/cm²")

    with col_right:
        finite_mask = np.isfinite(flux_lim_density)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wave[finite_mask], y=flux_lim_density[finite_mask],
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='Limiting flux density',
        ))
        fig.update_layout(
            title=dict(
                text=(
                    f'<b>Limiting Flux Density per Channel</b>'
                    f'<br><sup>SNR ≥ {desired_snr:.1f}, T = {T:.1f} s</sup>'
                ),
                font=dict(size=16),
            ),
            xaxis=dict(title=dict(text='<b>Wavelength (μm)</b>', font=dict(size=13))),
            yaxis=dict(title=dict(text='<b>Flux Density (phot/s/m²/μm)</b>', font=dict(size=13))),
            template='plotly_white',
            height=400,
        )
        st.plotly_chart(fig, width='stretch')