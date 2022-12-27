"""projectors and reflectors

basic components of the iterative phase retrieval algorithms"""

import numpy as np
import numpy.typing as npt
from numpy.fft import fft2, ifft2


def Ps(
    rho: npt.ArrayLike,
    support: npt.ArrayLike,
    positivity: bool = False
) -> np.ndarray:
    """support projector"""
    rho = rho*support
    if positivity:
        rho[np.real(rho) < 0] = 0
    return rho


def Rs(
    rho: npt.ArrayLike,
    support: npt.ArrayLike,
    positivity: bool = False
) -> np.ndarray:
    """support reflector"""
    rho = 2*Ps(rho, support, positivity) - rho
    return rho


def Pm(rho: npt.ArrayLike, modulus: npt.ArrayLike) -> np.ndarray:
    """modulus projector"""
    rho = fft2(rho)
    rho = modulus*np.exp(1j*np.angle(rho))
    rho = ifft2(rho)
    return rho


def Rm_gen(
    rho: npt.ArrayLike,
    modulus: npt.ArrayLike,
    gamma_gif: float = 1.0
) -> np.ndarray:
    """generalized modulus reflector"""
    rho = ((1+gamma_gif)*Pm(rho, modulus)) - (gamma_gif*rho)
    return rho



