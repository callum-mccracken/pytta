"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from de_utilities import triplet_decay_solution

def find_best_fit_params(times: np.ndarray,
                         intensities: np.ndarray,
                         initial_guesses: list[float],
                         epsilon: float=1) -> list[float]:
    """
    Given time and intensity values,
    find parameters in the function: k_tta & such.
    Wrapper for curve_fit.
    """

    result = curve_fit(
        triplet_decay_solution(epsilon), times, intensities,
        p0=initial_guesses,
        bounds=([0]*len(initial_guesses), [np.inf]*len(initial_guesses)))
    return result[0]
