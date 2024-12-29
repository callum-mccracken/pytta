"""
Collection of utility functions for DE-related things like:
- preparing data into the format needed to solve the DE
- actually solving the DE (with RK)
"""

from typing import Callable
import numpy as np
from scipy.signal import savgol_filter


def prepare_data(time: np.ndarray, intensity: np.ndarray,
                 t_cut: float) -> tuple[np.ndarray]:
    """
    Prepare data for further analysis
    - translate intensity down by the mean of the first 20 entries
    - scale time up by 1e6.
    - remove intensity values with a time value
      that is either negative or above t_cut

    Return time, intensity (two arrays, after preparation steps).
    """
    average = np.mean(intensity[:20])
    intensity = intensity-average
    time = time*1e6

    indices_to_keep = (0 < time) & (time < t_cut)

    return time[indices_to_keep], intensity[indices_to_keep]


def smooth_intensity(intensity: np.ndarray,
                     window_length: int=30, poly_filter: int=3) -> np.ndarray:
    """Smooth intensity values with a savgol_filter."""
    return savgol_filter(intensity, window_length, poly_filter)


def equation_to_fit(k_2_conc_a: float, k_ph: float, k_3: float, k_4: float,
                    times: np.ndarray, intensity: np.ndarray) -> float:
    """Form of the equation to fit using RK."""
    return k_2_conc_a*np.exp(-k_ph*times)-k_3*intensity-2*k_4*(intensity**2)


def runge_kutta(diff_equation: Callable,
                k_2_conc_a: float, k_ph: float, k_3: float, k_4: float,
                time_0: float, intensity_0: float,
                times: np.ndarray) -> np.ndarray:
    """Do RK."""
    step = times[1]-times[0]
    intensity = np.zeros(len(times))
    intensity[0] = intensity_0
    times[0] = time_0

    params = [k_2_conc_a,k_ph,k_3,k_4]

    for i in range(len(times)-1):
        time = times[i]
        intens = intensity[i]
        k1 = step*diff_equation(*params, time, intens)
        k2 = step*diff_equation(*params, time+1/2*step, intens+1/2*k1)
        k3 = step*diff_equation(*params, time+1/2*step, intens+1/2*k2)
        k4 = step*diff_equation(*params, time+step, intens+k3)

        intensity[i+1] = intens + k1/6 + k2/3 + k4/6

    return intensity


def triplet_decay_solution(epsilon: float=1,
                           time_0: float=0,
                           intensity_0: float=0) -> np.ndarray:
    """Solve for the solution to the triplet decay, using RK."""
    def to_return(times: np.ndarray,
                  *params: list[float]):
        """Helper function so curve_fit can have a 4-parameter function."""
        return epsilon*(
            runge_kutta(equation_to_fit, *params,
                        time_0, intensity_0, times))**2
    return to_return
