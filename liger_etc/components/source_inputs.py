import streamlit as st
import numpy as np
import plotly.graph_objects as go

from liger_etc.utils.resources import get_filter_info
from liger_etc.components.instrument_inputs import get_instrument_params
# imported lazily below to avoid circular import at module level
# from liger_etc.components.instrument_inputs import get_instrument_params


# ─── Unit Conversion Helpers ─────────────────────────────────────────────────

_H_ERG_S = 6.626e-27   # erg·s
_C_CM_S  = 2.998e10    # cm/s

def _phot_m2_to_erg_cm2(phot, wave_um):
    """phot/s/m²[/μm] → erg/s/cm²[/μm] using E=hc/λ at *wave_um* μm."""
    e_per_phot = _H_ERG_S * _C_CM_S / (float(wave_um) * 1e-4)  # erg/photon
    return float(phot) * e_per_phot / 1e4                        # m² → cm²

def _erg_cm2_to_phot_m2(erg, wave_um):
    """erg/s/cm²[/μm] → phot/s/m²[/μm] at *wave_um* μm."""
    e_per_phot = _H_ERG_S * _C_CM_S / (float(wave_um) * 1e-4)
    return float(erg) * 1e4 / e_per_phot


# ─── Spectrum Helper Functions ────────────────────────────────────────────────

def get_flat_spectrum(wave_um):
    """Flat (constant) spectrum.

    Parameters
    ----------
    wave_um : array-like
        Wavelength grid in microns.

    Returns
    -------
    np.ndarray
        Array of ones with the same length as *wave_um*.
    """
    return np.ones(len(np.asarray(wave_um)), dtype=float)


def get_vega_spectrum():
    """Retrieve the Vega reference spectrum via synphot.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(wave_um, flux)`` where *wave_um* is wavelength in microns and
        *flux* is in the native FLAM units of the synphot Vega model.
    """
    from synphot import SourceSpectrum
    vega = SourceSpectrum.from_vega()
    wave_um = vega.waveset.to('um').value
    flux = vega(vega.waveset).value
    return wave_um, flux


def get_blackbody_spectrum(wave_um, T_eff):
    """Planck blackbody spectral radiance at *T_eff*.

    Computed directly from the Planck function — no external library required.

    Parameters
    ----------
    wave_um : array-like
        Wavelength grid in microns.
    T_eff : float
        Effective temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Spectral radiance B_λ in SI units (W m⁻² sr⁻¹ m⁻¹).
    """
    h   = 6.62607015e-34   # J s
    c   = 2.99792458e8     # m s⁻¹
    k_B = 1.380649e-23     # J K⁻¹
    wave_m = np.asarray(wave_um, dtype=float) * 1e-6
    B = (2.0 * h * c**2 / wave_m**5) / np.expm1(h * c / (wave_m * k_B * float(T_eff)))
    return B


def get_phoenix_spectrum(T_eff, log_g=4.44):
    """Retrieve a PHOENIX stellar spectrum via *expecto*.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(wave_um, flux)`` where *wave_um* is wavelength in microns and
        *flux* is in the native FLAM units of the PHOENIX model.
    """
    from expecto import get_spectrum
    sp = get_spectrum(T_eff=float(T_eff), log_g=float(log_g), cache=True)
    wave_um = sp.spectral_axis.to('um').value
    flux = np.asarray(sp.flux.value).squeeze()  # ensure 1-D
    # sort by wavelength (specutils doesn't guarantee order)
    order = np.argsort(wave_um)
    return wave_um[order], flux[order]


def get_emission_line_spectrum(wave_um, center_um, fwhm_kms):
    """Gaussian emission-line profile normalised to unit integral.

    Parameters
    ----------
    wave_um : array-like
        Wavelength grid in microns.
    center_um : float
        Central wavelength of the emission line in microns.
    fwhm_kms : float
        Full-width at half-maximum of the line in km/s.

    Returns
    -------
    np.ndarray
        Gaussian profile whose integral over *wave_um* equals 1.
    """
    C_KMS    = 2.99792458e5  # speed of light, km s⁻¹
    sigma_um = (
        float(center_um)
        * (float(fwhm_kms) / C_KMS)
        / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    )
    wave_arr = np.asarray(wave_um, dtype=float)
    profile  = np.exp(-0.5 * ((wave_arr - float(center_um)) / sigma_um) ** 2)
    norm = np.trapezoid(profile, wave_arr)
    return profile / norm if norm > 0.0 else profile


# ── Source spectrum preview ────────────────────────────────────────────────────

def SourceSpectrumPlot(
    source_params: dict,
    instrument_params: dict | None = None,
):
    """
    Plot the intrinsic (above-atmosphere) spatially-integrated source spectrum
    in phot/s/m²/μm, scaled to the user's specified flux.
    """
    filter_info = (instrument_params or {}).get('filter_info')
    filter_name = (instrument_params or {}).get('filter_name')

    # Wavelength grid: 300-point linspace over filter bandpass, else broad NIR
    if filter_info:
        wave = np.linspace(
            float(filter_info['wavemin']), float(filter_info['wavemax']), 300
        )
    else:
        wave = np.linspace(0.9, 2.5, 300)

    # Build spectrum shape template
    spectrum_type   = source_params.get('spectrum_type', 'flat')
    spectrum_params = source_params.get('spectrum_params') or {}

    try:
        if spectrum_type == 'flat':
            template = get_flat_spectrum(wave)
        elif spectrum_type == 'blackbody':
            template = get_blackbody_spectrum(wave, spectrum_params.get('T_eff', 5800))
        elif spectrum_type == 'vega':
            vw, vf = get_vega_spectrum()
            template = np.interp(wave, vw, vf, left=0.0, right=0.0)
        elif spectrum_type == 'phoenix':
            vw, vf = get_phoenix_spectrum(
                spectrum_params.get('T_eff', 5800),
                spectrum_params.get('log_g', 4.44),
            )
            template = np.interp(wave, vw, vf, left=0.0, right=0.0)
        elif spectrum_type == 'emission_line':
            center = spectrum_params.get('center_um', float(wave[len(wave) // 2]))
            template = get_emission_line_spectrum(
                wave, center, spectrum_params.get('fwhm_kms', 100.0)
            )
        else:
            template = get_flat_spectrum(wave)
    except Exception as e:
        st.warning(f'Spectrum error ({spectrum_type}): {e}')
        template = get_flat_spectrum(wave)

    # Normalise so template integrates to 1 over μm  (units: 1/μm)
    norm = np.trapezoid(template, wave)
    if norm > 0:
        template = template / norm

    # Determine total photon flux (phot/s/m²) from the user's input
    flux_method = source_params.get('flux_method', 'mag_vega')
    total_flux  = None
    calibrated  = False

    if flux_method == 'mag_vega' and filter_info:
        try:
            from liger_iris_sim.utils import compute_filter_photon_flux
            total_flux = float(
                compute_filter_photon_flux(
                    source_params.get('mag_vega', 0.0), zp=filter_info['zpphot']
                )
            )
            calibrated = True
        except Exception:
            pass
    elif flux_method == 'flux_tot':
        val = source_params.get('flux_tot')
        if val is not None:
            total_flux = float(val)
            calibrated = True
    elif flux_method == 'flux_density':
        val = source_params.get('flux_density')
        if val is not None and filter_info:
            bw = float(filter_info['wavemax']) - float(filter_info['wavemin'])
            total_flux = float(val) * bw
            calibrated = True

    if total_flux is None or total_flux <= 0:
        # Normalised shape only
        flux_density = template
        y_label = 'Flux Density (normalised)'
    else:
        # phot/s/m²/μm  (template is already 1/μm, so product is phot/s/m²/μm)
        flux_density = template * total_flux
        y_label = 'Flux Density (phot/s/m²/μm)'

    # LSF-convolved version for IFS mode
    resolution = (instrument_params or {}).get('resolution')
    inst_mode  = (instrument_params or {}).get('_instrument_mode')
    flux_density_conv = None
    if resolution and inst_mode == 'IFS' and len(wave) > 1:
        try:
            from liger_iris_sim.sources import convolve_spectrum
            norm_conv = np.trapezoid(template, wave)
            t_conv = convolve_spectrum(wave, template / (norm_conv if norm_conv > 0 else 1.0), resolution)
            if total_flux and total_flux > 0:
                flux_density_conv = t_conv * total_flux
            else:
                flux_density_conv = t_conv
        except Exception:
            pass

    # #subtitle_parts = ['Intrinsic (above atmosphere)']
    # if filter_name and filter_info:
    #     subtitle_parts.append(
    #         f"{filter_name}  "
    #         f"({float(filter_info['wavemin']):.3f}–{float(filter_info['wavemax']):.3f} μm)"
    #     )
    # if resolution and inst_mode == 'IFS':
    #     subtitle_parts.append(f'R={int(resolution):,}')

    fig = go.Figure()
    if flux_density_conv is not None:
        fig.add_trace(go.Scatter(
            x=wave,
            y=flux_density_conv,
            mode='lines',
            line=dict(color='royalblue', width=2),
            name=f'Convolved (R={int(resolution):,})',
            hovertemplate='λ: %{x:.4f} μm<br>%{y:.3e}<extra></extra>',
        ))
    else:
        fig.add_trace(go.Scatter(
            x=wave,
            y=flux_density,
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='Source',
            hovertemplate='λ: %{x:.4f} μm<br>%{y:.3e}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(
            text=(
                '<b>Source Spectrum</b>'
                #f'<br><sup>{", ".join(subtitle_parts)}</sup>'
            ),
            font=dict(size=15),
        ),
        xaxis=dict(
            title=dict(text='<b>Wavelength (μm)</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text=f'<b>{y_label}</b>', font=dict(size=13)),
            tickfont=dict(size=12),
        ),
        template='plotly_white',
        height=350,
    )
    st.plotly_chart(fig, width='stretch')


# ── Component ───────────────────────────────────────────────────────────────

def SourceInputs():

    st.markdown('### Source Config')

    input_unit_options = {
        'mag_vega':     'Mag (Vega)',
        'flux_tot':     'Flux Total',
        'flux_density': 'Flux Density',
    }

    spectrum_options = {
        'flat':          'Flat',
        'vega':          'Vega',
        'blackbody':     'Blackbody',
        #'phoenix':       'Stellar (Phoenix)', # TODO: Cache phoenix spectra to speed this up
        'emission_line': 'Emission Line',
    }

    calc_type = st.session_state.get('calc_type', 'snr')
    limiting_flux_mode = (calc_type == 'flux')
    instrument_params = get_instrument_params()
    inst_mode = instrument_params['_instrument_mode']
    _filter_info = instrument_params.get('filter_info')
    _wc_um = float(_filter_info['wavecenter']) if _filter_info else None
    hide_spectrum = limiting_flux_mode or (inst_mode == 'IMG')

    if hide_spectrum:
        col_flux, col_source, col_source_plot = st.columns([1.0, 1.5, 4.0])
        col_spectrum = None
    else:
        col_flux, col_spectrum, col_source, col_source_plot = st.columns(
            [1.0, 1.5, 1.5, 4.0]
        )

    # ── Flux method + value ───────────────────────────────────────────────
    with col_flux:
        flux_method = st.radio(
            label='**Input Units:**',
            options=list(input_unit_options.keys()),
            index=0,
            format_func=lambda s: input_unit_options[s],
            key='flux_method',
            disabled=limiting_flux_mode,
        )

        if limiting_flux_mode:
            st.caption('Source flux is the output in Limiting Flux mode.')
            flux_value = None
        elif flux_method == 'mag_vega':
            flux_value = st.number_input(
                label='**Mag (Vega):**',
                value=12.0,
                key='mag_vega',
            )
        elif flux_method == 'flux_tot':
            # --- bidirectional phot ↔ erg sync ---
            def _tot_phot_changed():
                if _wc_um:
                    st.session_state['flux_tot_erg'] = _phot_m2_to_erg_cm2(
                        st.session_state['flux_tot'], _wc_um)
            def _tot_erg_changed():
                if _wc_um:
                    st.session_state['flux_tot'] = _erg_cm2_to_phot_m2(
                        st.session_state['flux_tot_erg'], _wc_um)
            # init erg key on first render
            if _wc_um and 'flux_tot_erg' not in st.session_state:
                st.session_state['flux_tot_erg'] = _phot_m2_to_erg_cm2(
                    st.session_state.get('flux_tot', 1e-10), _wc_um)

            flux_value = st.number_input(
                label='**Flux (phot/s/m²):**',
                value=1e-10,
                format='%e',
                key='flux_tot',
                on_change=_tot_phot_changed,
            )
            if _wc_um:
                st.number_input(
                    label='**Flux (erg/s/cm²):**',
                    #value=_phot_m2_to_erg_cm2(1e-10, _wc_um),
                    format='%e',
                    key='flux_tot_erg',
                    on_change=_tot_erg_changed,
                )
            else:
                st.caption('_Select a filter for erg/s/cm² input._')

        else:  # flux_density
            # --- bidirectional phot ↔ erg sync ---
            def _dens_phot_changed():
                if _wc_um:
                    st.session_state['flux_density_erg'] = _phot_m2_to_erg_cm2(
                        st.session_state['flux_density'], _wc_um)
            def _dens_erg_changed():
                if _wc_um:
                    st.session_state['flux_density'] = _erg_cm2_to_phot_m2(
                        st.session_state['flux_density_erg'], _wc_um)
            if _wc_um and 'flux_density_erg' not in st.session_state:
                st.session_state['flux_density_erg'] = _phot_m2_to_erg_cm2(
                    st.session_state.get('flux_density', 1e-10), _wc_um)

            flux_value = st.number_input(
                label='**Flux Density (phot/s/m²/μm):**',
                value=1e-10,
                format='%e',
                key='flux_density',
                on_change=_dens_phot_changed,
            )
            if _wc_um:
                st.number_input(
                    label='**Flux Density (erg/s/cm²/μm):**',
                    value=_phot_m2_to_erg_cm2(1e-10, _wc_um),
                    format='%e',
                    key='flux_density_erg',
                    on_change=_dens_erg_changed,
                )
            else:
                st.caption('_Select a filter for erg/s/cm²/μm input._')

    # ── Spectrum shape + conditional params ───────────────────────────────
    if col_spectrum is None:
        spectrum_type = st.session_state.get('spectrum_type', 'flat')
        spectrum_params = {}
    else:
        with col_spectrum:
            spectrum_type = st.radio(
                label='**Spectrum Shape:**',
                options=list(spectrum_options.keys()),
                index=0,
                format_func=lambda s: spectrum_options[s],
                key='spectrum_type',
            )

            spectrum_params = {}

            if spectrum_type == 'blackbody':
                bb_T_eff = st.number_input(
                    label='**T_eff (K):**',
                    value=5800,
                    step=100,
                    min_value=100,
                    max_value=100_000,
                    key='bb_T_eff',
                    help='Blackbody effective temperature in Kelvin',
                )
                spectrum_params['T_eff'] = bb_T_eff

            elif spectrum_type == 'phoenix':
                ph_cols = st.columns(2)
                with ph_cols[0]:
                    ph_T_eff = st.number_input(
                        label='**T_eff (K):**',
                        value=5800,
                        step=100,
                        min_value=2300,
                        max_value=12_000,
                        key='phoenix_T_eff',
                        help='Effective temperature (2300–12 000 K)',
                    )
                with ph_cols[1]:
                    ph_log_g = st.number_input(
                        label='**log g:**',
                        value=4.44,
                        step=0.5,
                        min_value=0.0,
                        max_value=6.0,
                        format='%.2f',
                        key='phoenix_log_g',
                        help='Surface gravity log₁₀(g / cm s⁻²); solar ≈ 4.44',
                    )
                spectrum_params['T_eff'] = ph_T_eff
                spectrum_params['log_g'] = ph_log_g

            elif spectrum_type == 'emission_line':
                # Default center wavelength from the selected filter, if available
                default_center = 1.25
                filter_name = st.session_state.get('filter_name')
                if filter_name is not None:
                    try:
                        default_center = float(get_filter_info(filter_name)['wavecenter'])
                    except Exception:
                        pass

                el_cols = st.columns(2)
                with el_cols[0]:
                    el_center = st.number_input(
                        label='**Center (μm):**',
                        value=default_center,
                        step=1E-8,
                        min_value=0.8,
                        max_value=2.5,
                        key='el_center_um',
                        help='Central wavelength of the emission line in microns',
                        format="%.6f"
                    )
                with el_cols[1]:
                    el_fwhm = st.number_input(
                        label='**FWHM (km/s):**',
                        value=10.0,
                        step=1.0,
                        min_value=1.0,
                        key='el_fwhm_kms',
                        help='Full-width at half-maximum of the emission line in km/s',
                        format="%.1f"
                    )
                spectrum_params['center_um'] = el_center
                spectrum_params['fwhm_kms']  = el_fwhm

    # ── Source type + profile ─────────────────────────────────────────────
    with col_source:
        if flux_method == 'mag_vega':
            type_options = {
                'point':    'Point',
                'extended': 'Extended (mag/arcsec²)',
            }
        elif flux_method == 'flux_tot':
            type_options = {
                'point':    'Point',
                'extended': 'Extended',
            }
        else:
            type_options = {
                'point':    'Point',
                'extended': 'Extended',
            }

        source_type = st.radio(
            label='**Source Type:**',
            options=list(type_options.keys()),
            key='source_type',
            format_func=lambda s: type_options[s],
        )

        source_profile    = None
        top_hat_radius    = None
        sersic_index      = None
        sersic_eff_radius = None

        if source_type == 'extended':
            profile_options = {
                'top-hat': 'Top-hat',
                'sersic':  'Sérsic',
            }
            source_profile = st.radio(
                label='**Profile:**',
                options=list(profile_options.keys()),
                key='source_profile',
                format_func=lambda s: profile_options[s],
            )

            if source_profile == 'top-hat':
                top_hat_radius = st.number_input(
                    label='**Top-hat Radius (arcsec):**',
                    value=0.10,
                    step=0.01,
                    min_value=0.0,
                    key='top_hat_radius',
                    help='Uniform-disk radius in arcseconds for the top-hat profile',
                    format='%.3f',
                )

            if source_profile == 'sersic':
                sr_cols = st.columns(2)
                with sr_cols[0]:
                    sersic_index = st.number_input(
                        label='**Sérsic index:**',
                        value=1,
                        step=1,
                        min_value=0,
                        max_value=100,
                        key='sersic_index',
                        help='Sérsic index (n=1: exponential disk, n=4: de Vaucouleurs)',
                    )
                with sr_cols[1]:
                    sersic_eff_radius = st.number_input(
                        label='**$R_{eff}$ (arcsec):**',
                        value=1.0,
                        step=0.1,
                        min_value=0.0,
                        key='sersic_eff_radius',
                        help='Effective (half-light) radius in arcseconds',
                    )

    # ── Assemble return dict ──────────────────────────────────────────────
    source_params = dict(
        flux_method=flux_method,
        source_type=source_type,
        source_profile=source_profile,
        spectrum_type=spectrum_type,
        spectrum_params=spectrum_params,
    )

    if flux_method == 'mag_vega':
        source_params['mag_vega'] = flux_value
    elif flux_method == 'flux_tot':
        source_params['flux_tot'] = flux_value
    else:
        source_params['flux_density'] = flux_value

    if source_profile == 'sersic':
        source_params['sersic_index']      = sersic_index
        source_params['sersic_eff_radius'] = sersic_eff_radius
    elif source_profile == 'top-hat':
        source_params['top_hat_radius'] = top_hat_radius

    # ── Spectrum preview ──────────────────────────────────────────────────
    with col_source_plot:
        if limiting_flux_mode:
            pass  # no preview needed; output is the flux
        elif inst_mode == 'IMG':
            st.info('Switch to IFS mode to preview the source spectrum.')
        else:
            SourceSpectrumPlot(source_params, instrument_params)

    return source_params


def get_source_params():
    state = st.session_state
    flux_method = state.get('flux_method', 'mag_vega')
    spectrum_type = state.get('spectrum_type', 'flat')

    # Flux value keyed by method
    flux_value_key = {'mag_vega': 'mag_vega', 'flux_tot': 'flux_tot', 'flux_density': 'flux_density'}
    flux_value = state.get(flux_value_key[flux_method])

    # Spectrum-specific params
    spectrum_params = {}
    if spectrum_type == 'blackbody':
        spectrum_params['T_eff'] = state.get('bb_T_eff')
    elif spectrum_type == 'phoenix':
        spectrum_params['T_eff'] = state.get('phoenix_T_eff')
        spectrum_params['log_g'] = state.get('phoenix_log_g')
    elif spectrum_type == 'emission_line':
        spectrum_params['center_um'] = state.get('el_center_um')
        spectrum_params['fwhm_kms']  = state.get('el_fwhm_kms')

    source_type    = state.get('source_type', 'point')
    source_profile = state.get('source_profile') if source_type == 'extended' else None

    params = dict(
        flux_method=flux_method,
        source_type=source_type,
        source_profile=source_profile,
        spectrum_type=spectrum_type,
        spectrum_params=spectrum_params,
    )
    params[flux_value_key[flux_method]] = flux_value

    if source_profile == 'sersic':
        params['sersic_index']      = state.get('sersic_index')
        params['sersic_eff_radius'] = state.get('sersic_eff_radius')
    elif source_profile == 'top-hat':
        params['top_hat_radius'] = state.get('top_hat_radius')

    return params