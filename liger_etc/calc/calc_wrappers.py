import numpy as np
import streamlit as st
from scipy.signal import fftconvolve

from liger_iris_sim.sources import make_point_source_image, make_point_source_ifs_cube, convolve_spectrum
from liger_iris_sim.expose import expose_imager, expose_ifs
from liger_iris_sim.utils import get_psf as _get_psf
from liger_iris_sim.utils import compute_filter_photon_flux

from liger_etc.utils import get_instrument_prop
from liger_etc.utils.resources import get_sky_data
from liger_etc.calc.photometry import _circular_aperture_flux
from liger_etc.utils.analytic_psf import analytic_psf

# Fixed image size for ETC imager simulations (51×51 pixels)
_IMAGER_SIM_SIZE = (51, 51)


# ── Shared helpers ────────────────────────────────────────────────────────────

@st.cache_resource
def get_psf(
    instrument_name: str,
    instrument_mode : str,
    wave: float,
    plate_scale: float,
) -> tuple[np.ndarray, dict]:
    return _get_psf(
        instrument_name=instrument_name,
        instrument_mode=instrument_mode,
        wave=wave,
        output_plate_scale=plate_scale,
        recenter_to_odd_shape=True,
        xs=0, ys=0
    )

def get_active_psf(instrument_params: dict) -> np.ndarray:
    """
    Return the PSF array for the given instrument configuration, using either
    the default library PSF or the semi-analytic model depending on the
    'psf_option' session-state key set by PSFInputs().

    For IMG mode the analytic PSF is averaged over three bandpass wavelengths
    (wavemin, wavecenter, wavemax).  For IFS mode only the central wavelength
    is used (monochromatic).
    """
    import streamlit as _st

    psf_option = _st.session_state.get('psf_option', 'default')

    name        = instrument_params['instrument_name']
    mode        = instrument_params['_instrument_mode']
    plate_scale = instrument_params['plate_scale']
    filter_info = instrument_params['filter_info']
    wcen        = float(filter_info['wavecenter'])

    if psf_option == 'analytic':
        strehl      = float(_st.session_state.get('psf_strehl', 0.50))
        fried_param = float(_st.session_state.get('psf_fried_param', 40.0))

        if mode == 'IMG':
            # Average over bandpass: wavemin, wavecenter, wavemax (μm → Å)
            wmin = float(filter_info['wavemin'])
            wmax = float(filter_info['wavemax'])
            lvals = [wmin * 1e4, wcen * 1e4, wmax * 1e4]
        else:
            # IFS: monochromatic at central wavelength
            lvals = [wcen * 1e4]

        return analytic_psf(
            strehl=strehl,
            lvals=lvals,
            plate_scale=plate_scale,
            fried_parameter=fried_param,
        )

    # Default: use library PSF
    psf, _ = get_psf(name, mode, wcen, plate_scale)
    return psf


def _flux_to_photons(source_params: dict, filter_info: dict) -> float:
    method = source_params['flux_method']
    if method == 'mag_vega':
        return float(compute_filter_photon_flux(source_params['mag_vega'], zp=filter_info['zpphot']))
    elif method == 'flux_tot':
        return float(source_params['flux_tot'])
    elif method == 'flux_density':
        bandwidth = filter_info['wavemax'] - filter_info['wavemin']
        return float(source_params['flux_density']) * float(bandwidth)
    raise ValueError(f"Unknown flux_method: {method}")


def _photons_to_mag(photon_flux: float, filter_info: dict) -> float:
    zp = filter_info['zpphot']
    return -2.5 * np.log10(photon_flux / zp)


def _make_top_hat_kernel(shape: tuple[int, int], radius_pix: float) -> np.ndarray:
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    kernel = (r <= max(radius_pix, 0.0)).astype(np.float64)
    s = kernel.sum()
    return (kernel / s) if s > 0 else kernel


def _make_sersic_kernel(shape: tuple[int, int], re_pix: float, n: float) -> np.ndarray:
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    b_n = max(2.0 * float(n) - 1.0 / 3.0, 0.01)
    kernel = np.exp(-b_n * ((r / max(float(re_pix), 0.1)) ** (1.0 / max(float(n), 0.1)) - 1.0))
    s = kernel.sum()
    return (kernel / s) if s > 0 else kernel


def _build_effective_psf(instrument_params: dict, source_params: dict) -> np.ndarray:
    """
    Build the PSF used for source injection.
    For extended sources, convolve the instrumental PSF with the selected
    source profile kernel (top-hat or Sersic).
    """
    psf = get_active_psf(instrument_params).astype(np.float64)

    if source_params.get('source_type') != 'extended':
        return psf

    profile = source_params.get('source_profile')
    plate_scale = float(instrument_params['plate_scale'])

    kernel = None
    if profile == 'top-hat':
        radius_arcsec = float(source_params.get('top_hat_radius') or 0.10)
        radius_pix = radius_arcsec / max(plate_scale, 1e-12)
        kernel = _make_top_hat_kernel(psf.shape, radius_pix)
    elif profile == 'sersic':
        re_arcsec = float(source_params.get('sersic_eff_radius') or 1.0)
        n = float(source_params.get('sersic_index') or 1.0)
        re_pix = re_arcsec / max(plate_scale, 1e-12)
        kernel = _make_sersic_kernel(psf.shape, re_pix, n)

    if kernel is None:
        return psf

    psf_eff = fftconvolve(psf, kernel, mode='same')
    psf_eff = np.maximum(psf_eff, 0.0)
    s = psf_eff.sum()
    return psf_eff / s if s > 0 else psf


def _build_point_source_image(
    instrument_params: dict,
    source_params: dict,
    photon_flux: float = 1.0,
) -> np.ndarray:
    psf = _build_effective_psf(instrument_params, source_params)
    size = _IMAGER_SIM_SIZE
    xpix = (size[1] - 1) / 2.0
    ypix = (size[0] - 1) / 2.0
    image = np.zeros(size, dtype=np.float32)
    make_point_source_image(xpix, ypix, photon_flux, psf=psf, image_out=image)
    return image


def _aperture_sum(image: np.ndarray, r_pix: float, error: np.ndarray | None = None) -> float:
    xpix = (image.shape[1] - 1) / 2.0
    ypix = (image.shape[0] - 1) / 2.0
    return float(_circular_aperture_flux(image, xpix, ypix, r_pix, error=error)['aperture_sum'][0])


def _background_rates(instrument_params: dict, sky_params: dict, dark_current: float):
    sky_data = get_sky_data(instrument_params, sky_params)
    sky_emission_rate = float(np.sum(sky_data['sky_em']))
    sky_trans_mean = float(np.mean(sky_data['sky_trans']))
    tput = instrument_params['tput_tot']
    collarea = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    sky_em_rate_pix = sky_emission_rate * collarea * tput
    return sky_em_rate_pix, dark_current, sky_trans_mean, sky_emission_rate


# ── calc_type = 'snr'  ────────────────────────────────────────────────────────

def calc_snr_imager(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
) -> dict:
    itime = float(exposure_params['input_itime'])
    n_frames = int(exposure_params['input_n_frames'])
    read_noise = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput = instrument_params['tput_tot']
    collarea = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale = instrument_params['plate_scale']

    sky_data = get_sky_data(instrument_params, sky_params)
    sky_emission_rate = float(np.sum(sky_data['sky_em']))
    sky_trans_mean = float(np.mean(sky_data['sky_trans']))

    photon_flux = _flux_to_photons(source_params, instrument_params['filter_info'])
    source_rate = _build_point_source_image(instrument_params, source_params, photon_flux)

    sim = expose_imager(
        source_rate,
        itime=itime,
        n_frames=n_frames,
        collarea=collarea,
        sky_emission_rate=sky_emission_rate,
        sky_trans_mean=sky_trans_mean,
        tput=tput,
        read_noise=read_noise,
        dark_current=dark_current,
    )

    size = source_rate.shape

    xpix = int(round((size[1] - 1) / 2.0))
    ypix = int(round((size[0] - 1) / 2.0))
    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)

    def _aperture_sum_data(sim, r_pix):
        sig_ap = _aperture_sum(sim['source_tot'], error=sim['noise_tot'], r_pix=r_pix)
        err_ap = _aperture_sum(sim['noise_tot'] ** 2, error=sim['noise_tot'], r_pix=r_pix)**0.5
        return sig_ap, err_ap
    
    sig_ap_user, err_ap_user = _aperture_sum_data(sim, r_pix)
    snr_ap_user = sig_ap_user / max(err_ap_user, 1e-30)

    dl_mas = aperture_params.get('aperture_rad_diff_lim')
    if dl_mas is not None:
        r_dl = float(dl_mas) / (plate_scale * 1000.0)
        sig_dl, err_dl = _aperture_sum_data(sim, r_pix=r_dl)
        snr_ap_diff_lim = sig_dl / max(err_dl, 1e-30)
    else:
        snr_ap_diff_lim = None

    N_pix_ap = np.pi * r_pix ** 2
    S_ap = sig_ap_user / (itime * n_frames)  # e-/s in aperture
    B_ap = (float(sim['sky_em_rate']) + float(sim['dark_rate'])) * N_pix_ap  # e-/s in aperture

    sim.update(
        calc_type='snr',
        snr_peak=float(np.nanmax(sim['snr'])),
        snr_ap_user=snr_ap_user,
        snr_ap_diff_lim=snr_ap_diff_lim,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=dl_mas,
        xpix=xpix,
        ypix=ypix,
        S_ap=S_ap,
        B_ap=B_ap,
        N_pix_ap=N_pix_ap,
    )
    return sim


# ── calc_type = 'itime'  ──────────────────────────────────────────────────────

def calc_itime_imager(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
    n_frames: int = 1,
) -> dict:
    """
    Analytically find the integration time for a desired aperture SNR.
    Solves: S²t² − snr²(S+B)t − snr²·RN_var = 0  (t = total time)
    """
    desired_snr = float(exposure_params['snr'])
    read_noise = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput = instrument_params['tput_tot']
    collarea = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale = instrument_params['plate_scale']

    sky_em_rate_pix, dark_rate, sky_trans_mean, _ = _background_rates(
        instrument_params, sky_params, dark_current
    )

    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)
    N_pix_ap = np.pi * r_pix ** 2

    photon_flux = _flux_to_photons(source_params, instrument_params['filter_info'])
    unit_img = _build_point_source_image(instrument_params, source_params, 1.0)
    f_ap = _aperture_sum(unit_img, r_pix)
    S_ap = photon_flux * collarea * tput * sky_trans_mean * f_ap

    B_ap = (sky_em_rate_pix + dark_rate) * N_pix_ap
    RN_var = N_pix_ap * (read_noise * np.sqrt(n_frames)) ** 2

    snr2 = desired_snr ** 2
    a = S_ap ** 2
    b = -snr2 * (S_ap + B_ap)
    c = -snr2 * RN_var
    discriminant = b ** 2 - 4.0 * a * c

    if S_ap <= 0 or discriminant < 0:
        return {'calc_type': 'itime', 'error': 'Cannot achieve desired SNR with given parameters.'}

    T = (-b + np.sqrt(discriminant)) / (2.0 * a)
    itime = T / n_frames

    return dict(
        calc_type='itime',
        itime=itime,
        n_frames=n_frames,
        total_time=T,
        desired_snr=desired_snr,
        S_ap=S_ap,
        B_ap=B_ap,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=aperture_params.get('aperture_rad_diff_lim'),
    )


# ── calc_type = 'flux'  ───────────────────────────────────────────────────────

def calc_flux_imager(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
    desired_snr: float = 5.0, # Integrated over user aperture, no bandpass for IMG
) -> dict:
    """
    Analytically find the limiting flux for a given itime and SNR threshold.
    Solves: S²T² − snr²·T·S − snr²(BT + RN_var) = 0  (S = source e-/s in aperture)
    """
    itime = float(exposure_params['input_itime'])
    n_frames = int(exposure_params['input_n_frames'])
    read_noise = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput = instrument_params['tput_tot']
    collarea = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale = instrument_params['plate_scale']
    filter_info = instrument_params['filter_info']

    T = itime * n_frames

    sky_em_rate_pix, dark_rate, sky_trans_mean, _ = _background_rates(
        instrument_params, sky_params, dark_current
    )

    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)
    N_pix_ap = np.pi * r_pix ** 2

    unit_img = _build_point_source_image(instrument_params, source_params, 1.0)
    f_ap = _aperture_sum(unit_img, r_pix)

    B_ap = (sky_em_rate_pix + dark_rate) * N_pix_ap
    RN_var = N_pix_ap * (read_noise * np.sqrt(n_frames)) ** 2

    snr2 = desired_snr ** 2
    a = T ** 2
    b = -snr2 * T
    c = -snr2 * (B_ap * T + RN_var)
    discriminant = b ** 2 - 4.0 * a * c

    if T <= 0 or discriminant < 0:
        return {'calc_type': 'flux', 'error': 'Cannot compute limiting flux with given parameters.'}

    S_ap = (-b + np.sqrt(discriminant)) / (2.0 * a)

    conv = collarea * tput * sky_trans_mean * f_ap
    if conv <= 0:
        return {'calc_type': 'flux', 'error': 'Zero throughput or PSF fraction in aperture.'}

    photon_flux_lim = S_ap / conv
    mag_lim = _photons_to_mag(photon_flux_lim, filter_info)

    return dict(
        calc_type='flux',
        photon_flux_lim=photon_flux_lim,
        mag_lim=mag_lim,
        itime=itime,
        n_frames=n_frames,
        total_time=T,
        desired_snr=desired_snr,
        B_ap=B_ap,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=aperture_params.get('aperture_rad_diff_lim'),
    )


# ── IFS shared helpers ────────────────────────────────────────────────────────


def _get_ifs_spectrum(source_params: dict, wave: np.ndarray, filter_info: dict, resolution: float | None = None) -> np.ndarray:
    """
    Build a normalised spectral template on *wave* then scale to total photon
    flux (phot/s/m²) as supplied by the user.  If *resolution* is given the
    template is convolved with the instrument LSF before normalisation.
    Returns phot/s/m²/wavebin.
    """
    spectrum_type = source_params.get('spectrum_type', 'flat')
    sp = source_params.get('spectrum_params', {})

    if spectrum_type == 'flat':
        template = np.ones(len(wave), dtype=np.float64)
    elif spectrum_type == 'blackbody':
        from liger_etc.components.source_inputs import get_blackbody_spectrum
        template = get_blackbody_spectrum(wave, sp['T_eff'])
    elif spectrum_type == 'phoenix':
        from liger_etc.components.source_inputs import get_phoenix_spectrum
        wave_sp, flux_sp = get_phoenix_spectrum(sp['T_eff'], sp.get('log_g', 4.44))
        template = np.interp(wave, wave_sp, flux_sp, left=0.0, right=0.0)
    elif spectrum_type == 'vega':
        from liger_etc.components.source_inputs import get_vega_spectrum
        wave_v, flux_v = get_vega_spectrum()
        template = np.interp(wave, wave_v, flux_v, left=0.0, right=0.0)
    elif spectrum_type == 'emission_line':
        from liger_etc.components.source_inputs import get_emission_line_spectrum
        template = get_emission_line_spectrum(wave, sp['center_um'], sp['fwhm_kms'])
    else:
        template = np.ones(len(wave), dtype=np.float64)

    # Convolve with instrument LSF before normalising
    if resolution is not None and resolution > 0 and len(wave) > 1:
        template = convolve_spectrum(wave, template.astype(np.float64), resolution)

    # Normalise so sum * dwave = 1, then scale to total photon flux
    norm = np.trapezoid(template, wave)
    if norm > 0:
        template = template / norm  # per μm → re-normalised

    photon_flux_total = _flux_to_photons(source_params, filter_info)  # phot/s/m²
    # Distribute over wavebins: phot/s/m²/wavebin
    dwave = np.gradient(wave)
    return (template * photon_flux_total * dwave).astype(np.float32)


def _ifs_aperture_sum_spectrum(
    cube: np.ndarray,
    r_pix: float,
) -> np.ndarray:
    """Spatially sum a (nw, ny, nx) cube within a circular aperture at centre."""
    from photutils.aperture import CircularAperture, aperture_photometry
    ny, nx = cube.shape[1], cube.shape[2]
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0
    aperture = CircularAperture([(xc, yc)], r=r_pix)
    spectra = np.zeros(cube.shape[0], dtype=np.float64)
    for i in range(cube.shape[0]):
        phot = aperture_photometry(cube[i], aperture)
        spectra[i] = float(phot['aperture_sum'][0])
    return spectra


# ── calc_type = 'snr' (IFS) ──────────────────────────────────────────────────

def calc_snr_ifs(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
) -> dict:
    itime      = float(exposure_params['input_itime'])
    n_frames   = int(exposure_params['input_n_frames'])
    read_noise = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput        = instrument_params['tput_tot']
    collarea    = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale = instrument_params['plate_scale']
    filter_info = instrument_params['filter_info']
    resolution  = instrument_params['resolution']
    ifs_mode    = instrument_params['ifs_mode'].lower()

    sky_data = get_sky_data(instrument_params, sky_params)
    wave     = np.array(sky_data['wave'])

    # PSF — default library or analytic depending on session state
    psf = _build_effective_psf(instrument_params, source_params)

    # Detector size — use the actual IFS FOV, no padding
    size = instrument_params['size']
    xpix_f = (size[1] - 1) / 2.0
    ypix_f = (size[0] - 1) / 2.0

    # Build spectral source cube (nw, ny, nx)  phot/s/m²
    template_spec = _get_ifs_spectrum(source_params, wave, filter_info, resolution=resolution)
    input_cube = np.zeros((len(wave), *size), dtype=np.float32)
    make_point_source_ifs_cube(
        xpix_f, ypix_f,
        wave=wave,
        template=template_spec,
        psf=psf,
        cube_out=input_cube,
    )

    sim = expose_ifs(
        input_cube,
        itime=itime,
        n_frames=n_frames,
        collarea=collarea,
        sky_emission_rate=sky_data['sky_em'],
        sky_transmission=sky_data['sky_trans'],
        tput=tput,
        read_noise=read_noise,
        dark_current=dark_current,
    )

    # Aperture sums → spectra
    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)

    src_spec  = _ifs_aperture_sum_spectrum(sim['source_tot'],  r_pix)
    noise_spec = _ifs_aperture_sum_spectrum(sim['noise_tot'] ** 2, r_pix) ** 0.5
    obs_spec  = _ifs_aperture_sum_spectrum(sim['observed_tot'], r_pix)
    snr_spec  = np.where(noise_spec > 0, src_spec / noise_spec, 0.0)

    # Diffraction-limit aperture
    dl_mas = aperture_params.get('aperture_rad_diff_lim')
    if dl_mas is not None:
        r_dl = float(dl_mas) / (plate_scale * 1000.0)
        src_dl    = _ifs_aperture_sum_spectrum(sim['source_tot'], r_dl)
        noise_dl  = _ifs_aperture_sum_spectrum(sim['noise_tot'] ** 2, r_dl) ** 0.5
        snr_spec_dl = np.where(noise_dl > 0, src_dl / noise_dl, 0.0)
    else:
        snr_spec_dl = None

    N_pix_ap = np.pi * r_pix ** 2
    xpix = int(round(xpix_f))
    ypix = int(round(ypix_f))
    sim.update(
        calc_type='snr',
        wave=wave,
        snr_peak=float(np.nanmax(sim['snr'])),
        snr_spec=snr_spec,
        snr_spec_dl=snr_spec_dl,
        src_spec=src_spec,
        noise_spec=noise_spec,
        obs_spec=obs_spec,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=dl_mas,
        N_pix_ap=N_pix_ap,
        xpix=xpix,
        ypix=ypix,
    )
    return sim


# ── calc_type = 'itime' (IFS) ────────────────────────────────────────────────

def calc_itime_ifs(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
    n_frames: int = 1,
) -> dict:
    """
    Per-wavebin analytical itime for desired per-channel aperture SNR.
    Solves per wavebin: S²t² − snr²(S+B)t − snr²·RN_var = 0
    Returns the worst-case (longest) required itime across the bandpass.
    """
    desired_snr  = float(exposure_params['snr'])
    read_noise   = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput         = instrument_params['tput_tot']
    collarea     = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale  = instrument_params['plate_scale']
    filter_info  = instrument_params['filter_info']
    ifs_mode     = instrument_params['ifs_mode'].lower()

    sky_data = get_sky_data(instrument_params, sky_params)
    wave     = np.array(sky_data['wave'])

    psf = _build_effective_psf(instrument_params, source_params)

    size = instrument_params['size']
    xpix_f = (size[1] - 1) / 2.0
    ypix_f = (size[0] - 1) / 2.0

    template_spec = _get_ifs_spectrum(source_params, wave, filter_info, resolution=instrument_params['resolution'])  # phot/s/m²/wavebin
    unit_cube = np.zeros((len(wave), *size), dtype=np.float32)
    make_point_source_ifs_cube(
        xpix_f, ypix_f,
        wave=wave,
        template=np.ones_like(template_spec, dtype=np.float32),
        psf=psf, cube_out=unit_cube,
    )

    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)
    N_pix_ap = np.pi * r_pix ** 2

    # PSF fraction in aperture per wavebin (spatially summed)
    f_ap_cube = _ifs_aperture_sum_spectrum(unit_cube, r_pix)

    # Source e-/s in aperture per wavebin
    S_arr = template_spec * collarea * tput * np.array(sky_data['sky_trans']) * f_ap_cube

    sky_em_rate_pix = np.array(sky_data['sky_em']) * collarea * tput  # e-/s/pix/wavebin
    B_arr = (sky_em_rate_pix + dark_current) * N_pix_ap

    RN_var = N_pix_ap * (read_noise * np.sqrt(n_frames)) ** 2
    snr2 = desired_snr ** 2

    a = S_arr ** 2
    b = -snr2 * (S_arr + B_arr)
    c = np.full_like(a, -snr2 * RN_var)

    disc = b ** 2 - 4.0 * a * c
    valid = (S_arr > 0) & (disc >= 0)
    T_arr = np.where(valid, (-b + np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.inf)
    itime_arr = T_arr / n_frames

    # Worst-case channel
    worst_idx = int(np.argmax(T_arr))
    T_worst   = float(T_arr[worst_idx])
    itime_worst = T_worst / n_frames

    return dict(
        calc_type='itime',
        wave=wave,
        itime_arr=itime_arr,
        itime=itime_worst,
        n_frames=n_frames,
        total_time=T_worst,
        worst_wave=float(wave[worst_idx]),
        desired_snr=desired_snr,
        S_arr=S_arr,
        B_arr=B_arr,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=aperture_params.get('aperture_rad_diff_lim'),
    )


# ── calc_type = 'flux' (IFS) ─────────────────────────────────────────────────

def calc_flux_ifs(
    instrument_params: dict,
    sky_params: dict,
    source_params: dict,
    exposure_params: dict,
    aperture_params: dict,
    desired_snr: float = 5.0,
) -> dict:
    """
    Per-wavebin limiting flux density for a given itime and per-channel SNR
    threshold.  Solves per wavebin:
        S²T² − snr²·T·S − snr²(BT + RN_var) = 0
    Returns the limiting phot/s/m²/wavebin spectrum and the equivalent
    continuum magnitude (via filter zeropoint).
    """
    itime        = float(exposure_params['input_itime'])
    n_frames     = int(exposure_params['input_n_frames'])
    read_noise   = float(exposure_params['read_noise'])
    dark_current = float(exposure_params['dark_current'])
    tput         = instrument_params['tput_tot']
    collarea     = get_instrument_prop(instrument_params['instrument_name'], 'collarea')
    plate_scale  = instrument_params['plate_scale']
    filter_info  = instrument_params['filter_info']
    ifs_mode     = instrument_params['ifs_mode'].lower()

    T = itime * n_frames

    sky_data = get_sky_data(instrument_params, sky_params)
    wave     = np.array(sky_data['wave'])

    psf = _build_effective_psf(instrument_params, source_params)

    size = instrument_params['size']
    xpix_f = (size[1] - 1) / 2.0
    ypix_f = (size[0] - 1) / 2.0

    unit_cube = np.zeros((len(wave), *size), dtype=np.float32)
    make_point_source_ifs_cube(
        xpix_f, ypix_f,
        wave=wave,
        template=np.ones_like(wave, dtype=np.float32),
        psf=psf, cube_out=unit_cube,
    )

    aperture_rad_mas = float(aperture_params['aperture_rad'])
    r_pix = aperture_rad_mas / (plate_scale * 1000.0)
    N_pix_ap = np.pi * r_pix ** 2

    f_ap_cube = _ifs_aperture_sum_spectrum(unit_cube, r_pix)

    sky_em_rate_pix = np.array(sky_data['sky_em']) * collarea * tput
    sky_trans       = np.array(sky_data['sky_trans'])
    B_arr = (sky_em_rate_pix + dark_current) * N_pix_ap
    RN_var = N_pix_ap * (read_noise * np.sqrt(n_frames)) ** 2

    snr2 = desired_snr ** 2
    a = np.full(len(wave), T ** 2)
    b = np.full(len(wave), -snr2 * T)
    c = -snr2 * (B_arr * T + RN_var)

    disc = b ** 2 - 4.0 * a * c
    S_ap_lim = np.where(disc >= 0, (-b + np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.inf)

    conv = collarea * tput * sky_trans * f_ap_cube
    flux_lim = np.where(conv > 0, S_ap_lim / conv, np.inf)  # phot/s/m²/wavebin

    # Bandpass-integrated continuum equivalent magnitude
    dwave = np.gradient(wave)
    flux_lim_density = np.where(dwave > 0, flux_lim / dwave, np.inf)  # phot/s/m²/μm
    median_flux_density = float(np.nanmedian(flux_lim_density[np.isfinite(flux_lim_density)]))
    bandwidth = filter_info['wavemax'] - filter_info['wavemin']
    photon_flux_lim = median_flux_density * bandwidth
    mag_lim = _photons_to_mag(photon_flux_lim, filter_info) if photon_flux_lim > 0 else np.nan

    return dict(
        calc_type='flux',
        wave=wave,
        flux_lim=flux_lim,
        flux_lim_density=flux_lim_density,
        photon_flux_lim=photon_flux_lim,
        mag_lim=mag_lim,
        itime=itime,
        n_frames=n_frames,
        total_time=T,
        desired_snr=desired_snr,
        B_arr=B_arr,
        aperture_user=aperture_rad_mas,
        aperture_diff_lim=aperture_params.get('aperture_rad_diff_lim'),
    )
