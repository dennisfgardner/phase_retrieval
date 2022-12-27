"""iterative phase retrieval algorithms

written as combinations of projections and reflections"""

import numpy as np
import numpy.typing as npt

from projectors_and_reflectors import Pm, Rm_gen, Ps, Rs


def ER(
    rho: npt.ArrayLike,
    modulus: npt.ArrayLike,
    support: npt.ArrayLike,
    positivity: bool = False
):
    """error reduction"""
    rho = Ps(Pm(rho, modulus), support, positivity)
    return rho


def GIF_RAAR(
    rho: npt.ArrayLike,
    modulus: npt.ArrayLike,
    support: npt.ArrayLike,
    beta: float = 0.9,
    gamma_gif: float = 1.9,
    positivity: bool = False
) -> np.ndarray:
    """generalized interior feedback relaxed averaged alternating reflectors"""
    term1 = 0.5*beta*(Rs(Rm_gen(rho, modulus, gamma_gif), support, positivity) + rho)
    term2 = 0.5*beta*(Rm_gen(rho, modulus, gamma_gif) - Rm_gen(rho, modulus))
    term3 = (1-beta)*Pm(rho, modulus)
    rho = term1 + term2 + term3
    return rho
