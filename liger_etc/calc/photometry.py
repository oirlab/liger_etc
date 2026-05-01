import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry


def _circular_aperture_flux(
    data : np.ndarray,
    x : float, y : float, r : float,
    error : np.ndarray | None = None,
    mask : np.ndarray | None = None
):
    positions = [(x, y)]
    aperture = CircularAperture(positions, r=r)

    if error is not None:
        # TODO: Handle this better.
        bad = np.where(~np.isfinite(error) | ~np.isfinite(data) | (error <= 0) | (data < 0))
        error[bad] = np.inf

    phot = aperture_photometry(
        data,
        aperture,
        error=error,
        mask=mask
    )

    return phot


# def circular_aperture_flux(
#     sim : dict,
#     # data : np.ndarray,
#     # x : float, y : float, r : float,
#     # error : np.ndarray | None = None,
#     # mask : np.ndarray | None = None
# ):
#     positions = [(x, y)]
#     aperture = CircularAperture(positions, r=r)

#     phot = aperture_photometry(
#         data,
#         aperture,
#         error=error,
#         mask=mask
#     )

#     return phot