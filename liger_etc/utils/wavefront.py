"""
Analytic PSF module

Rewritten with AI from:
https://github.com/oirlab/osiris_etc/blob/master/etc_analytic_psf/wavefront.py

#   Originally written by Tom Murphy
#   Original source code at https://tmurphy.physics.ucsd.edu/astr597/exercises/speckle.html

"""

from __future__ import annotations

import sys
from math import ceil, floor, sqrt, pi

import numpy as np
from numpy.random import default_rng
from scipy.special import gamma
import numba
from numba import njit, prange

# ---------------------------------------------------------------------------
# Physical / instrument constants
# ---------------------------------------------------------------------------

KECK_DIAM_M       = 10.949          # primary mirror outer diameter [m]
KECK_CENT_OBS     = 0.2265          # fractional central obscuration (diameter ratio)
KECK_SPIDER_FRAC  = 0.00457         # spider vane width as fraction of pupil diameter
KECK_N_SEGMENTS   = 36              # hexagonal segment count
KECK_SEG_GAP_FRAC = 0.00264        # inter-segment gap as fraction of pupil diameter

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _noll_to_nm(j: int) -> tuple[int, int]:
    """
    Convert Noll index *j* (1-based) to radial order *n* and azimuthal
    frequency *m* following the Noll (1976) convention.

    Returns
    -------
    n : int  radial order
    m : int  azimuthal frequency (positive = cosine term, negative = sine term)
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = int(ceil((-3 + sqrt(9 + 8 * (j - 1))) / 2))
    j_remaining = j - n * (n + 1) // 2
    # azimuthal index following Noll ordering
    if n % 2 == 0:
        m = 2 * int(floor(j_remaining / 2))
    else:
        m = 1 + 2 * int(floor((j_remaining - 1) / 2))
    if j % 2 == 0:
        m = -m
    return n, m


def _factorial(n: float) -> float:
    """Generalised factorial via the gamma function (supports non-integers)."""
    return float(gamma(n + 1))


# ---------------------------------------------------------------------------
# Numba-accelerated radial polynomial accumulator
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _zernike_radial(r_flat: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Evaluate the Zernike radial polynomial R_n^m(r) at each point in
    *r_flat* (a 1-D view of the radius array).

    Uses the standard sum definition::

        R_n^m(r) = sum_{s=0}^{(n-m)//2}
                     (-1)^s * (n-s)! / (s! * ((n+m)/2-s)! * ((n-m)/2-s)!)
                     * r^{n-2s}

    Factorial values are computed iteratively to avoid calling scipy inside
    a JIT-compiled kernel.

    Parameters
    ----------
    r_flat : 1-D float64 array  – radial coordinates in [0, 1]
    n, m   : ints               – Zernike orders

    Returns
    -------
    R : 1-D float64 array, same length as *r_flat*
    """
    npix = r_flat.shape[0]
    R = np.zeros(npix, dtype=np.float64)
    s_max = (n - m) // 2

    for s in range(s_max + 1):
        # Compute the factorial coefficient iteratively (no scipy in numba)
        ns   = n - s
        ns_f = 1.0
        for k in range(1, ns + 1):
            ns_f *= k
        s_f = 1.0
        for k in range(1, s + 1):
            s_f *= k
        npm2_s = (n + m) // 2 - s
        npm2_f = 1.0
        for k in range(1, npm2_s + 1):
            npm2_f *= k
        nmp2_s = (n - m) // 2 - s
        nmp2_f = 1.0
        for k in range(1, nmp2_s + 1):
            nmp2_f *= k

        sign  = (-1) ** s
        coeff = sign * ns_f / (s_f * npm2_f * nmp2_f)
        power = n - 2 * s

        for i in prange(npix):
            R[i] += coeff * (r_flat[i] ** power)

    return R


# ---------------------------------------------------------------------------
# Zernike polynomial (single term)
# ---------------------------------------------------------------------------

def zernike(j: int, npix: int = 256, phase: float = 0.0) -> np.ndarray:
    """
    Compute the *j*-th Zernike polynomial (Noll ordering, 1-based) on a
    square grid of *npix* × *npix* pixels inscribed in a unit circle.

    Parameters
    ----------
    j     : int    Noll index (1 = piston, 2 = tip, 3 = tilt, …)
    npix  : int    output array size in pixels (default 256)
    phase : float  global rotation of the azimuthal angle [rad] (default 0)

    Returns
    -------
    Z : (npix, npix) float64 array  – polynomial values; 0 outside unit circle
    """
    J_MAX = 820          # supports n < 40
    if j < 1 or j > J_MAX:
        raise ValueError(f"Noll index j must be in [1, {J_MAX}]")

    x = np.linspace(-1.0, 1.0, npix, endpoint=False) + 1.0 / npix
    xarr, yarr = np.meshgrid(x, x)

    r     = np.sqrt(xarr**2 + yarr**2)
    theta = np.arctan2(yarr, xarr) + phase

    inside = r <= 1.0

    n, m = _noll_to_nm(j)
    m_abs = abs(m)

    R_flat = _zernike_radial(r[inside].ravel(), n, m_abs)

    zarr = np.zeros((npix, npix), dtype=np.float64)

    norm = sqrt(n + 1) if m_abs == 0 else sqrt(2 * (n + 1))

    if m_abs == 0:
        zarr[inside] = norm * R_flat
    elif m > 0:                       # Noll even → cosine
        zarr[inside] = norm * R_flat * np.cos(m_abs * theta[inside])
    else:                             # Noll odd  → sine
        zarr[inside] = norm * R_flat * np.sin(m_abs * theta[inside])

    return zarr


# ---------------------------------------------------------------------------
# Aperture functions
# ---------------------------------------------------------------------------

def aperture(
    npix: int = 256,
    cent_obs: float = 0.0,
    spider: int = 0,
) -> np.ndarray:
    """
    Circular aperture with optional central obscuration and cross spiders.

    Parameters
    ----------
    npix      : int   side length of the output array [pixels]
    cent_obs  : float fractional diameter of the central obscuration (0–1)
    spider    : int   width of the spider vanes in pixels (0 = no spider)

    Returns
    -------
    illum : (npix, npix) float64 binary mask
    """
    x = np.linspace(-1.0, 1.0, npix, endpoint=False) + 1.0 / npix
    xarr, yarr = np.meshgrid(x, x)
    r = np.sqrt(xarr**2 + yarr**2)

    illum = (r <= 1.0).astype(np.float64)

    if cent_obs > 0.0:
        illum[r < cent_obs] = 0.0

    if spider > 0:
        half = spider // 2
        ctr  = npix // 2
        illum[ctr - half : ctr + half + 1, :] = 0.0
        illum[:, ctr - half : ctr + half + 1] = 0.0

    return illum


def keck_aperture(npix: int = 256, include_gaps: bool = True) -> np.ndarray:
    """
    Approximate Keck primary mirror pupil.

    Models the outer aperture, central obscuration, and four-vane spider.
    Hexagonal segment gaps are optionally included as thin annular gaps
    (a simple approximation; full segment geometry requires additional
    computation).

    Parameters
    ----------
    npix         : int   pixel size of the output array
    include_gaps : bool  if True, add approximate inter-segment gap structure

    Returns
    -------
    illum : (npix, npix) float64 pupil mask
    """
    spider_pix = max(1, int(KECK_SPIDER_FRAC * npix))
    illum = aperture(npix=npix, cent_obs=KECK_CENT_OBS, spider=spider_pix)

    if include_gaps:
        # Approximate the three sets of radial gaps in the segmented mirror
        x = np.linspace(-1.0, 1.0, npix, endpoint=False) + 1.0 / npix
        xarr, yarr = np.meshgrid(x, x)
        r     = np.sqrt(xarr**2 + yarr**2)
        theta = np.arctan2(yarr, xarr)          # [-pi, pi]

        gap_half = KECK_SEG_GAP_FRAC / 2.0

        # Three gap orientations at 0°, 60°, 120°
        for angle_deg in (0, 60, 120):
            phi = np.deg2rad(angle_deg)
            # Angular distance to the nearest gap axis (mod 60°)
            dtheta = np.abs(np.mod(theta - phi + pi / 6, pi / 3) - pi / 6)
            # Gap width narrows with r (narrower near centre)
            in_gap = (dtheta < gap_half) & (r > KECK_CENT_OBS) & (r <= 1.0)
            illum[in_gap] = 0.0

    return illum


# ---------------------------------------------------------------------------
# Wavefront generators
# ---------------------------------------------------------------------------

def plane_wave(npix: int = 256) -> np.ndarray:
    """Return a flat (aberration-free) wavefront."""
    return np.zeros((npix, npix), dtype=np.float64)


def seeing(
    d_over_r0: float,
    npix: int = 256,
    nterms: int = 15,
    level: float | None = None,
    rng: np.random.Generator | None = None,
    quiet: bool = False,
) -> np.ndarray:
    """
    Simulate a Kolmogorov turbulence wavefront via Zernike decomposition.

    Uses the Noll (1976) residual variance table for the first ten terms and
    the analytic approximation ``0.2944 * (D/r0)^(5/3) * j^{-0.866}`` for
    higher orders.

    Parameters
    ----------
    d_over_r0 : float  telescope diameter / Fried parameter (strength of seeing)
    npix      : int    output wavefront size [pixels]
    nterms    : int    number of Zernike modes to include (≥ 10)
    level     : float  if given, include all modes whose RMS exceeds *level*
                       radians (overrides *nterms*)
    rng       : numpy Generator  random number generator (reproducibility)
    quiet     : bool   suppress progress message

    Returns
    -------
    wf : (npix, npix) float64 wavefront in radians
    """
    if nterms < 10:
        raise ValueError("Need at least 10 Zernike terms for a valid seeing screen.")

    scale = d_over_r0 ** (5.0 / 3.0)

    if level is not None:
        j_arr  = np.arange(1, 401, dtype=float)
        coeffs = np.sqrt(0.2944 * scale * (j_arr[:-1] ** -0.866 - j_arr[1:] ** -0.866))
        below  = np.where(coeffs < level)[0]
        if len(below):
            n_order = int(ceil(sqrt(2 * below[0]) - 0.5))
            nterms  = max(15, int(n_order * (n_order + 1) / 2))

    # Noll residual variance table (first 10 terms, Noll 1976 Table 1)
    resid = np.zeros(nterms, dtype=np.float64)
    resid[:10] = [1.030, 0.582, 0.134, 0.111, 0.088,
                  0.065, 0.059, 0.053, 0.046, 0.040]
    for i in range(10, nterms):
        resid[i] = 0.2944 * (i + 1) ** -0.866

    if rng is None:
        rng = default_rng()

    wf   = np.zeros((npix, npix), dtype=np.float64)
    coef = np.zeros(nterms, dtype=np.float64)
    for j in range(2, nterms + 1):
        coef[j - 1] = sqrt((resid[j - 2] - resid[j - 1]) * scale)
        wf          += coef[j - 1] * rng.standard_normal() * zernike(j, npix=npix)

    if not quiet:
        print(f"Seeing screen: {nterms} Zernike terms, "
              f"highest-order RMS = {coef[nterms-1]:.4f} rad")

    return wf


# ---------------------------------------------------------------------------
# PSF computation
# ---------------------------------------------------------------------------

def make_psf(
    pupil: np.ndarray,
    wavefront: np.ndarray,
    overfill: float = 1.0,
) -> np.ndarray:
    """
    Compute the PSF by Fraunhofer diffraction (pupil-plane FFT).

    The complex pupil field is P(x,y) = A(x,y) * exp(i * W(x,y)), where
    A is the aperture amplitude mask and W is the wavefront in radians.
    The PSF is |FT{P}|², shifted so that zero frequency is at array centre,
    and cropped back to the original pixel count.

    Parameters
    ----------
    pupil     : (npix, npix) float64  aperture amplitude (0/1 mask)
    wavefront : (npix, npix) float64  OPD map [radians]
    overfill  : float  zero-padding factor; >1 increases PSF sampling

    Returns
    -------
    psf_out : (npix, npix) float64  normalised PSF (peak ≈ 1 for diffraction limit)
    """
    npix  = wavefront.shape[0]
    nbig  = int(npix * overfill)
    half  = (nbig - npix) // 2

    # Embed pupil and wavefront in a zero-padded array
    pupil_big = np.zeros((nbig, nbig), dtype=np.float64)
    wf_big    = np.zeros((nbig, nbig), dtype=np.float64)
    pupil_big[half:half + npix, half:half + npix] = pupil
    wf_big   [half:half + npix, half:half + npix] = wavefront

    # Complex pupil field and Fourier transform
    field  = pupil_big * np.exp(1j * wf_big)
    ft     = np.fft.fft2(field)
    power  = np.real(ft * np.conj(ft))

    # FFT-shift (swap quadrants) and crop to original size
    power_shifted = np.fft.fftshift(power)
    psf_out = power_shifted[half:half + npix, half:half + npix].copy()

    # Normalise so peak of diffraction-limited PSF ≈ 1
    psf_out /= psf_out.max() if psf_out.max() > 0 else 1.0

    return psf_out


def strehl(
    pupil: np.ndarray,
    wavefront: np.ndarray,
    overfill: float = 1.0,
) -> float:
    """
    Estimate the Strehl ratio from the PSF peak relative to a diffraction-
    limited reference computed on the same pupil.

    Parameters
    ----------
    pupil, wavefront, overfill : see :func:`make_psf`

    Returns
    -------
    S : float  Strehl ratio in [0, 1]
    """
    psf_abbr = make_psf(pupil, wavefront,              overfill=overfill)
    psf_diff = make_psf(pupil, plane_wave(pupil.shape[0]), overfill=overfill)
    return float(psf_abbr.max() / psf_diff.max())


# ---------------------------------------------------------------------------
# Numba-accelerated aperture photometry
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def flux_in(
    img: np.ndarray,
    ctrx: float,
    ctry: float,
    rad: float,
    n_sub: int = 10,
) -> float:
    """
    Integrate the flux inside a circular aperture using sub-pixel sampling.

    Each pixel that is within one pixel width of the aperture edge is
    sub-sampled on an *n_sub* × *n_sub* grid to obtain a precise fractional
    area weight.  Pixels fully inside the aperture contribute their full flux.

    Compiled with Numba (``@njit``, ``parallel=True``) for performance.

    Parameters
    ----------
    img   : 2-D float64 array  – image to measure
    ctrx  : float  – aperture centre, axis-0 coordinate
    ctry  : float  – aperture centre, axis-1 coordinate
    rad   : float  – aperture radius [pixels]
    n_sub : int    – sub-pixel grid size per axis (default 10)

    Returns
    -------
    flux : float  – integrated flux within the aperture
    """
    nx    = img.shape[0]
    ny    = img.shape[1]
    flux  = 0.0
    inv_n = 1.0 / n_sub
    n2    = n_sub * n_sub

    for x in prange(nx):
        dx   = x - ctrx
        r_cx = sqrt(dx * dx)             # used to skip entire rows quickly
        for y in range(ny):
            dy = y - ctry
            r  = sqrt(dx * dx + dy * dy)

            if r + 0.7072 < rad:         # pixel fully inside (diagonal ≈ 0.707)
                flux += img[x, y]
            elif r - 0.7072 < rad:       # pixel straddles the edge → sub-sample
                count = 0
                for si in range(n_sub):
                    for sj in range(n_sub):
                        sx = dx + (si - n_sub / 2 + 0.5) * inv_n
                        sy = dy + (sj - n_sub / 2 + 0.5) * inv_n
                        if sqrt(sx * sx + sy * sy) < rad:
                            count += 1
                flux += img[x, y] * count / n2

    return flux


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def keck_psf(
    d_over_r0: float = 0.0,
    nterms: int = 22,
    npix: int = 256,
    overfill: float = 2.0,
    include_gaps: bool = True,
    rng: np.random.Generator | None = None,
    quiet: bool = False,
) -> dict:
    """
    High-level convenience function: build a Keck PSF from scratch.

    Parameters
    ----------
    d_over_r0    : float  D/r0 seeing strength; 0 = diffraction limited
    nterms       : int    number of Zernike modes for the seeing screen
    npix         : int    pupil/PSF array size [pixels]
    overfill     : float  FFT zero-padding factor (≥1; 2 gives Nyquist sampling)
    include_gaps : bool   include segment gap structure in pupil
    rng          : numpy Generator  for reproducible seeing screens
    quiet        : bool   suppress console output

    Returns
    -------
    result : dict with keys
        'psf'        – (npix, npix) PSF array
        'pupil'      – (npix, npix) pupil mask
        'wavefront'  – (npix, npix) wavefront [rad]
        'strehl'     – estimated Strehl ratio
    """
    pup = keck_aperture(npix=npix, include_gaps=include_gaps)

    if d_over_r0 > 0:
        wf = seeing(d_over_r0, npix=npix, nterms=nterms, rng=rng, quiet=quiet)
    else:
        wf = plane_wave(npix=npix)

    psf_img = make_psf(pup, wf, overfill=overfill)
    S       = strehl(pup, wf, overfill=overfill)

    if not quiet:
        print(f"Strehl ratio: {S:.4f}")

    return {"psf": psf_img, "pupil": pup, "wavefront": wf, "strehl": S}