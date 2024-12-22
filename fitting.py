"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from de_utilities import triplet_decay_solution

def find_best_fit_params(times: np.ndarray, intensities: np.ndarray,
                         initial_guesses: list, epsilon: float=1) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    """

    result = curve_fit(
        triplet_decay_solution(epsilon), times, intensities, p0=initial_guesses,
        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    popt = result[0]
    k_2_conc_a, k_3, k_4, k_ph = popt

    return k_2_conc_a, k_3, k_4, k_ph
