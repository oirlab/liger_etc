"""
etc_analytic_psf.py
-------------------
Semi-analytic AO PSF generator for the Keck / OSIRIS / LIGER ETC.

Originally written by Sanchit Sabhlok for the OSIRIS ETC (OIRLab, UCSD).
Adapted from speckle.py / wavefront.py by Tom Murphy
  (https://tmurphy.physics.ucsd.edu/astr597/exercises/speckle.html).
Keck pupil FITS image provided by Mike Fitzgerald & Pauline Arriaga (UCLA).

This module is an *application layer* on top of the low-level primitives in
``keck_psf.py``.  It adds:

  * Physical plate-scale → overfill mapping (``get_overfill``)
  * Semi-analytic AO PSF model  PSF = (1−S)·G_seeing + S·PSF_diffrac
  * Multi-wavelength iteration with optional stack / band-average output
  * Detector rebinning via ``liger_iris_sim.utils.rebin_image``
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
import streamlit as st

from liger_etc.utils.wavefront import (
    KECK_DIAM_M,
    plane_wave,
    make_psf,
)
from liger_iris_sim.utils import rebin_image

from liger_iris_drp_resources import load_keck_pupil_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARCSEC_PER_RAD = 206_265.0
_CM_PER_M       = 100.0


def round_up_to_even(f: float) -> int:
    """Return the smallest even integer ≥ *f*."""
    return math.ceil(f / 2.0) * 2


def get_overfill(
    wl: float,
    plate_scale: float,
    npix: int = 1800,
    diameter: float = KECK_DIAM_M,
    rebin: float = 2.0,
) -> tuple[float, float]:
    """
    Compute the FFT zero-padding (overfill) factor needed to achieve a given
    output plate scale.

    The FFT pixel scale is λ/D per resolution element. Oversampling by
    *rebin* before binning to *plate_scale* gives::

        overfill = (λ/D) * 206265 / (plate_scale / rebin)

    The result is nudged up to the nearest even multiple of 1/npix so the
    padded array has an even side length.

    Parameters
    ----------
    wl         : float  wavelength [Å]
    plate_scale: float  desired output plate scale [arcsec/pixel]
    npix       : int    pupil array size [pixels]  (default 1800)
    diameter   : float  telescope diameter [m]     (default KECK_DIAM_M)
    rebin      : float  oversampling factor applied before binning (default 2)

    Returns
    -------
    overfill : float  zero-padding factor (≥ 1)
    rebin    : float  oversampling factor (may be doubled if overfill < 1)
    """
    wl_m = wl * 1e-10                          # Å → m
    raw  = wl_m / diameter * _ARCSEC_PER_RAD / (plate_scale / rebin)
    ovf  = (float(round_up_to_even(raw * npix)) + 0.001) / float(npix)

    if ovf < 1.0:
        # Plate scale too coarse; double the internal oversampling and retry
        return get_overfill(wl, plate_scale, npix=npix,
                            diameter=diameter, rebin=rebin * 2.0)

    return float(ovf), float(rebin)


def crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    """
    Return the central *cropy* × *cropx* sub-array of *img*.

    Parameters
    ----------
    img   : 2-D array
    cropx : output width  [pixels]
    cropy : output height [pixels]
    """
    ny, nx = img.shape
    x0 = nx // 2 - cropx // 2
    y0 = ny // 2 - cropy // 2
    return img[y0 : y0 + cropy, x0 : x0 + cropx]


def _gaussian_kernel(
    d: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """
    Normalised circular 2-D Gaussian with the given FWHM.

    Parameters
    ----------
    d    : 2-D array of radial distances [same units as fwhm]
    fwhm : full width at half maximum

    Returns
    -------
    g : 2-D array, sum ≈ 1  (exact once discretised and flux-renormalised)
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    g     = np.exp(-0.5 * (d / sigma) ** 2) / (2.0 * math.pi * sigma ** 2)
    return g / g.sum()


# ---------------------------------------------------------------------------
# Main PSF generator
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def analytic_psf(
    strehl: float,
    lvals: float | list[float] | tuple[float, ...],
    plate_scale: float,
    fried_parameter: float = 20.0,
    diameter: float = KECK_DIAM_M,
    npix_pupil: int = 1800,
    verbose: bool = False,
    stack: bool = False,
) -> np.ndarray:
    """
    Generate a semi-analytic AO PSF for Keck at one or more wavelengths.

    The model follows the standard AO PSF decomposition::

        PSF(λ) = (1 − S) · G_seeing(λ)  +  S · PSF_diffrac(λ)

    where *S* is the Strehl ratio, *G_seeing* is a Gaussian seeing halo
    scaled to the Fried parameter, and *PSF_diffrac* is the diffraction-
    limited PSF computed from the actual Keck pupil FITS image.  The result
    is convolved with a Strehl-broadening kernel that corrects the diffraction-
    limited core width for the finite Strehl, then binned to *plate_scale*
    via :func:`liger_iris_sim.utils.rebin_image`.

    Parameters
    ----------
    strehl           : float  Strehl ratio in (0, 1]
    lvals            : float | list[float]
                        Wavelength(s) in Ångströms.  The output is the mean
                        over all wavelengths unless *stack* is True.
    plate_scale      : float  Output plate scale [arcsec/pixel]
    fried_parameter  : float  Fried parameter r₀ [cm]  (default 20)
    diameter         : float  Telescope diameter [m]    (default KECK_DIAM_M)
    npix_pupil       : int    Pupil array side length   (default 1800)
    verbose          : bool   Print diagnostic messages (default False)
    stack            : bool   Return per-wavelength stack instead of mean
                              (default False)

    Returns
    -------
    ndarray
        If ``stack=False`` : (ny, nx) mean PSF, flux-normalised to unity.
        If ``stack=True``  : (n_λ, ny, nx) stack of per-wavelength PSFs.
    """
    # ------------------------------------------------------------------
    # Load Keck pupil
    # ------------------------------------------------------------------

    # keck_pupil = fits.open(pupil_path)[0].data.astype(np.float64)
    keck_pupil = load_keck_pupil_image()
    npix       = keck_pupil.shape[0]

    pwf = plane_wave(npix=npix)          # aberration-free wavefront

    if verbose:
        print(f"Telescope diameter : {diameter} m")
        print(f"Fried parameter    : {fried_parameter} cm")
        print(f"Strehl ratio       : {strehl}")

    # ------------------------------------------------------------------
    # Normalise wavelength input
    # ------------------------------------------------------------------
    if not isinstance(lvals, (list, tuple)):
        lvals = [lvals]

    r0_m = fried_parameter * 1e-2          # cm → m

    psf_stack: list[np.ndarray] = []

    for wl_ang in lvals:
        wl_m = wl_ang * 1e-10              # Å → m

        if verbose:
            print(f"\n── λ = {wl_ang:.1f} Å ──")

        # --------------------------------------------------------------
        # Overfill factor & internal pixel scale
        # --------------------------------------------------------------
        overfill, rebin = get_overfill(
            wl_ang, plate_scale, npix=npix_pupil, diameter=diameter
        )
        pixel_scale_internal = plate_scale / rebin    # arcsec/pixel (oversampled)

        if verbose:
            print(f"  Overfill factor   : {overfill:.4f}")
            print(f"  Rebin factor      : {rebin:.0f}")
            print(f"  Internal px scale : {pixel_scale_internal:.5f} arcsec/px")

        # --------------------------------------------------------------
        # Diffraction-limited PSF from FITS pupil
        # --------------------------------------------------------------
        psf_diffrac = make_psf(keck_pupil, pwf, overfill=overfill)
        psf_diffrac /= psf_diffrac.sum()

        # --------------------------------------------------------------
        # Radial coordinate grid [arcsec]
        # --------------------------------------------------------------
        n_half = psf_diffrac.shape[0] // 2
        ax     = np.linspace(-n_half, n_half - 1, 2 * n_half) * pixel_scale_internal
        xarr, yarr = np.meshgrid(ax, ax)
        d      = np.hypot(xarr, yarr)       # radial distance [arcsec]

        # --------------------------------------------------------------
        # Seeing halo (Gaussian with FWHM = λ/r₀)
        # --------------------------------------------------------------
        seeing_fwhm_arcsec = (wl_m / r0_m) * _ARCSEC_PER_RAD
        g_seeing           = _gaussian_kernel(d, seeing_fwhm_arcsec)

        # --------------------------------------------------------------
        # Strehl-broadening kernel
        # Diffraction-limited FWHM: 1.03 λ/D
        # Broadened FWHM at Strehl S:  0.92 · (λ/D) / √S
        # Kernel FWHM: quadrature difference of broadened and DL widths
        # --------------------------------------------------------------
        fwhm_dl  = 1.03  * (wl_m / diameter) * _ARCSEC_PER_RAD
        fwhm_ao  = 0.92  * fwhm_dl / math.sqrt(strehl)
        fwhm_ker = math.sqrt(max(fwhm_ao**2 - fwhm_dl**2, 0.0))

        if verbose:
            print(f"  FWHM_DL   : {fwhm_dl*1e3:.3f} mas")
            print(f"  FWHM_AO   : {fwhm_ao*1e3:.3f} mas")
            print(f"  FWHM_ker  : {fwhm_ker*1e3:.3f} mas")

        # --------------------------------------------------------------
        # Combine seeing halo + diffraction-limited core
        # --------------------------------------------------------------
        psf_combined = (1.0 - strehl) * g_seeing + strehl * psf_diffrac
        psf_combined /= psf_combined.sum()

        # --------------------------------------------------------------
        # Convolve with Strehl-broadening kernel (cropped for speed)
        # --------------------------------------------------------------
        if fwhm_ker > 0.0:
            ker_half   = max(4, int(fwhm_ker / pixel_scale_internal * 5))
            ker_ax     = np.linspace(-ker_half, ker_half - 1, 2 * ker_half) \
                         * pixel_scale_internal
            kx, ky     = np.meshgrid(ker_ax, ker_ax)
            d_ker      = np.hypot(kx, ky)
            ker        = _gaussian_kernel(d_ker, fwhm_ker)

            psf_combined = fftconvolve(psf_combined, ker, mode="same")
            psf_combined = np.clip(psf_combined, 0.0, None)
            psf_combined /= psf_combined.sum()

        # --------------------------------------------------------------
        # Rebin to output plate scale
        # --------------------------------------------------------------
        psf_rebinned = rebin_image(
            psf_combined,
            scale_in=pixel_scale_internal,
            scale_out=plate_scale,
        )
        psf_rebinned /= psf_rebinned.sum()

        if verbose:
            print(f"  Rebinned shape: {psf_rebinned.shape}")

        psf_stack.append(psf_rebinned)

    # ------------------------------------------------------------------
    # Align all images to the smallest (shortest λ) size and return
    # ------------------------------------------------------------------
    min_size = min(im.shape[0] for im in psf_stack)
    aligned  = np.stack(
        [crop_center(im, min_size, min_size) for im in psf_stack],
        axis=0,
    )

    if stack:
        return aligned                         # (n_λ, ny, nx)

    result = np.mean(aligned, axis=0)
    result /= result.sum()

    from liger_iris_sim.utils.psf_utils import _recenter_psf_to_odd_shape
    result = _recenter_psf_to_odd_shape(result)

    return result                              # (ny, nx)