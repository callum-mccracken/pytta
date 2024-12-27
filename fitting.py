"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from de_utilities import triplet_decay_solution,triplet_decay_solution_2
import math

def find_best_fit_params(times: np.ndarray, intensities: np.ndarray,
                         initial_guesses: list) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    """
    #
    result = curve_fit(triplet_decay_solution, times, intensities, p0=initial_guesses,bounds = ((0,0,0,0),(math.inf,math.inf,math.inf,math.inf)))
    popt = result[0]
    k_TET,k_3s,k_3A,k_TTA = popt

    for i in range(10):
        result = curve_fit(triplet_decay_solution, times, intensities, p0=popt,bounds = ((0,0,0,0),(math.inf,math.inf,math.inf,math.inf)))
        popt = result[0]
        k_TET,k_3s,k_3A,k_TTA = popt

    return k_TET,k_3s,k_3A,k_TTA

def find_best_fit_params_2(times: np.ndarray, intensities: np.ndarray,
                         initial_guesses: list) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    """
    #
    result = curve_fit(triplet_decay_solution_2, times, intensities, p0=initial_guesses,bounds = ((0,0,0,0,0),(math.inf,math.inf,math.inf,math.inf,math.inf)))
    popt = result[0]
    K,k_TET,k_3s,k_3A,k_TTA = popt

    for i in range(10):
        result = curve_fit(triplet_decay_solution_2, times, intensities, p0=popt,bounds = ((0,0,0,0,0),(math.inf,math.inf,math.inf,math.inf,math.inf)))
        popt = result[0]
        K,k_TET,k_3s,k_3A,k_TTA = popt

    return K,k_TET,k_3s,k_3A,k_TTA
