"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from de_utilities import equation_to_fit

def find_best_fit_params(times: np.ndarray, intensities: np.ndarray) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    
    We're using scipy.optimize.curve_fit which wants one independent variable.
    We don't really have an independent variable so uhhhhh?
    There's probably a better function to use, this is just a first attempt.
    """
    def scipy_format(x, k_2_conc_a, k_3, k_4, k_ph):
        time = x[:, 0]
        intensity = x[:, 1]
        return equation_to_fit(k_2_conc_a, k_ph, k_3, k_4, time, intensity)

    input_x = np.column_stack((times, intensities))
    result = curve_fit(scipy_format,
                       input_x,
                       intensities,
                       p0=[10, 10, 10, 10])  # TODO: smart initial guesses?
    popt = result[0]
    k_2_conc_a, k_3, k_4, k_ph = popt
    return k_2_conc_a, k_3, k_4, k_ph
