"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from de_utilities import triplet_decay_solution,triplet_decay_solution_2

def find_best_fit_params(times: np.ndarray, intensities: np.ndarray,
                         initial_guesses: list) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    """
    result = curve_fit(
        triplet_decay_solution, times, intensities, p0=initial_guesses,
        bounds=((0,0,0,0), (np.inf,np.inf,np.inf,np.inf)))
    k_tet, k_3s, k_3_a, k_tta = result[0]

    #for _ in range(10):
    #    result = curve_fit(
    #        triplet_decay_solution, times, intensities,
    #        p0=[k_tet, k_3s, k_3_a, k_tta],
    #        bounds=((0,0,0,0), (np.inf,np.inf,np.inf,np.inf)))
    #    k_tet, k_3s, k_3_a, k_tta = result[0]
    return k_tet,k_3s,k_3_a,k_tta

def find_best_fit_params_2(times: np.ndarray, intensities: np.ndarray,
                           initial_guesses: list) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    """
    result = curve_fit(
        triplet_decay_solution_2, times, intensities, p0=initial_guesses,
        bounds=((0,0,0,0,0), (np.inf,np.inf,np.inf,np.inf)))
    popt = result[0]
    capital_k, k_tet, k_3s, k_3_a, k_tta = popt

    for _ in range(10):
        result = curve_fit(
            triplet_decay_solution_2, times, intensities, p0=popt,
            bounds=((0,0,0,0,0), (np.inf,np.inf,np.inf,np.inf)))
        popt = result[0]
        capital_k,k_tet,k_3s,k_3_a,k_tta = popt

    return capital_k,k_tet,k_3s,k_3_a,k_tta
