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
    - remove intensity values with a time value that is either negative
      or above t_cut

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


def equation_to_fit(k_tet: float, k_3s: float, k_3_a: float, k_tta: float,
                    times: np.ndarray, intensity: np.ndarray) -> float:
    """Form of the equation to fit using RK."""
    return k_tet*np.exp((-k_3s)*times)-k_3_a*intensity-2*k_tta*(intensity**2)

def equation_to_fit_2(capital_k: float, k_tet: float, k_3s: float,
                      k_3_a: float, k_tta: float, times: np.ndarray,
                      intensity: np.ndarray) -> float:
    """Form of the equation to fit using RK."""
    return capital_k*k_tet*np.exp((-k_3s-k_tet)*times)-k_3_a*intensity-2*k_tta*(intensity**2)


def runge_kutta(diff_equation: Callable,
                k_tet: float, k_3s: float, k_3_a: float, k_tta: float,
                time_0: float, intensity_0: float,
                times: np.ndarray) -> np.ndarray:
    """Do RK."""
    time_step = times[1]-times[0]
    intensity = np.zeros(len(times))
    intensity[0] = intensity_0

    k1 = time_step*diff_equation(
     k_tet,k_3s,k_3_a,k_tta,time_0,intensity_0)
    k2 = time_step*diff_equation(
     k_tet,k_3s,k_3_a,k_tta,time_0+1/2*time_step,intensity_0+1/2*k1)
    k3 = time_step*diff_equation(
     k_tet,k_3s,k_3_a,k_tta,time_0+1/2*time_step,intensity_0+1/2*k2)
    k4 = time_step*diff_equation(
     k_tet,k_3s,k_3_a,k_tta,time_0+time_step,intensity_0+k3)

    intensity[1] = intensity[0] + k1/6 + k2/3 + k3/3 + k4/6

    for i in range(2,len(times)-1):
        k1 = time_step*diff_equation(
             k_tet,k_3s,k_3_a,k_tta,
             times[i-1],intensity[i-1])
        k2 = time_step*diff_equation(
             k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+1/2*time_step,intensity[i-1]+1/2*k1)
        k3 = time_step*diff_equation(
             k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+1/2*time_step,intensity[i-1]+1/2*k2)
        k4 = time_step*diff_equation(
             k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+time_step,intensity[i-1]+k3)

        intensity[i] = intensity[i-1] + k1/6 + k2/3 +k3/3 + k4/6

    return intensity

def runge_kutta_2(diff_equation: Callable,
                capital_k:float, k_tet: float, k_3s: float,
                k_3_a: float, k_tta: float,
                time_0: float, intensity_0: float,
                times: np.ndarray) -> np.ndarray:
    """Do RK."""
    time_step = times[1]-times[0]
    intensity = np.zeros(len(times))
    intensity[0] = intensity_0

    k1 = time_step*diff_equation(
          capital_k,k_tet,k_3s,k_3_a,k_tta,
          time_0,intensity_0)
    k2 = time_step*diff_equation(
          capital_k,k_tet,k_3s,k_3_a,k_tta,
          time_0+1/2*time_step,intensity_0+1/2*k1)
    k3 = time_step*diff_equation(
          capital_k,k_tet,k_3s,k_3_a,k_tta,
          time_0+1/2*time_step,intensity_0+1/2*k2)
    k4 = time_step*diff_equation(
          capital_k,k_tet,k_3s,k_3_a,k_tta,
          time_0+time_step,intensity_0+k3)

    intensity[1] = intensity[0] + k1/6 + k2/3 + k3/3 + k4/6

    for i in range(2,len(times)-1):
        k1 = time_step*diff_equation(
             capital_k,k_tet,k_3s,k_3_a,k_tta,
             times[i-1],intensity[i-1])
        k2 = time_step*diff_equation(
             capital_k,k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+1/2*time_step,intensity[i-1]+1/2*k1)
        k3 = time_step*diff_equation(
             capital_k,k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+1/2*time_step,intensity[i-1]+1/2*k2)
        k4 = time_step*diff_equation(
             capital_k,k_tet,k_3s,k_3_a,k_tta,
             times[i-1]+time_step,intensity[i-1]+k3)

        intensity[i] = intensity[i-1] + k1/6 + k2/3 +k3/3 + k4/6

    return intensity

def triplet_decay_solution(times: np.ndarray,
                           k_tet: float, k_3s: float,
                           k_3_a: float, k_tta: float) -> np.ndarray:
    """Solve for the solution to the triplet decay, using RK."""
    time_0 = 0
    intensity_0 = 0
    epsilon = 1

    return epsilon*(
        runge_kutta(equation_to_fit,
                    k_tet,k_3s,k_3_a,k_tta,time_0,intensity_0,times))**2

def triplet_decay_solution_2(times: np.ndarray,
                             capital_k: float, k_tet: float, k_3s: float,
                             k_3_a: float, k_tta: float) -> np.ndarray:
    """Solve for the solution to the triplet decay, using RK."""
    time_0 = 0
    intensity_0 = 0
    epsilon = 1

    return epsilon*(
        runge_kutta_2(equation_to_fit_2,
                      capital_k,k_tet,k_3s,k_3_a,k_tta,
                      time_0,intensity_0,times))**2
