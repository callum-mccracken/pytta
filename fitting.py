"""Module to deal with fitting."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from de_utilities import equation_to_fit

def find_best_fit_params(times: np.ndarray, intensities: np.ndarray) -> tuple[float]:
    """
    Given time and intensity values, find k_2_conc_a, k_3, k_4, and k_ph.
    
    We're using scipy.optimize.curve_fit which wants one independent variable.
    We don't really have an independent variable so uhhhhh?
    There's probably a better function to use, this is just a first attempt.
    """

    def fitfunc(time, k_2_conc_a, k_3, k_4, k_ph):
        'Function that returns Ca computed from an ODE for a k'
        def myode(intensity, time):
            return equation_to_fit(k_2_conc_a, k_ph, k_3, k_4, time, intensity)

        intensity_0 = intensities[0]
        solution = odeint(myode, intensity_0, time)
        return solution[:,0]

    # TODO: how to make smart initial guesses?
    # Currently very dependent on initial guess
    result = curve_fit(fitfunc, times, intensities, p0=[2, 100, 30, 1])

    popt = result[0]
    k_2_conc_a, k_3, k_4, k_ph = popt
    return k_2_conc_a, k_3, k_4, k_ph
