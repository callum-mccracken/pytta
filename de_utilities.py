"""
Collection of utility functions for DE-related things like:
- preparing data into the format needed to solve the DE
- actually solving the DE (with RK)
"""

import statistics
from typing import Callable
import numpy as np
from scipy.signal import savgol_filter

def prepare_data(time: np.ndarray, intensity: np.ndarray, t_cut: float) -> tuple[np.ndarray]:
    """
    Prepare data for further analysis
    - translate intensity down by the mean of the first 20 entries
    - scale time up by 1e6.
    - remove intensity values with a time value that is either negative or above t_cut

    Return time, intensity (two arrays, after preparation steps).
    """
    average = statistics.mean(intensity[:20])
    intensity = intensity-average
    time = time*1e6

    indices_to_keep = 0 < time < t_cut

    return time[indices_to_keep], intensity[indices_to_keep]

def smooth_intensity(intensity: np.ndarray,
                     window_length: int=30, poly_filter: int=3) -> np.ndarray:
    """Smooth intensity values with a savgol_filter."""
    return savgol_filter(intensity, window_length, poly_filter)

def equation_to_fit(a: float, b: float, c: float, d: float, x: np.ndarray, y: np.ndarray) -> float:
    """Form of the equation to fit using RK."""
    return a*np.exp(-b*x)-c*y-2*d*(y**2)


def runge_kutta(diff_equation: Callable,
                a: float, b: float, c: float, d: float, x0: float, y0: float,
                x: np.ndarray) -> np.ndarray:
    """Do RK."""
    h = x[1]-x[0]
    y = np.zeros(len(x))
    y[0] = y0

    k1 = h*diff_equation(a,b,c,d,x0,y0)
    k2 = h*diff_equation(a,b,c,d,x0+1/2*h,y0+1/2*k1)
    k3 = h*diff_equation(a,b,c,d,x0+1/2*h,y0+1/2*k2)
    k4 = h*diff_equation(a,b,c,d,x0+h,y0+k3)

    y[1] = y[0] + k1/6 + k2/3 + k4/6

    for i in range(2,len(x)-1):
        k1 = h*diff_equation(a,b,c,d,x[i-1],y[i-1])
        k2 = h*diff_equation(a,b,c,d,x[i-1]+1/2*h,y[i-1]+1/2*k1)
        k3 = h*diff_equation(a,b,c,d,x[i-1]+1/2*h,y[i-1]+1/2*k2)
        k4 = h*diff_equation(a,b,c,d,x[i-1]+h,y[i-1]+k3)

        y[i] = y[i-1] + k1/6 + k2/3 + k4/6

    return y

def triplet_decay_solution(x: np.ndarray,
                           a: float, b: float, c: float, d: float) -> np.ndarray:
    """Solve for the solution to the triplet decay, using RK."""
    x0 = 0
    y0 = 0
    e = 1000

    return e*(runge_kutta(equation_to_fit,a,b,c,d,x0,y0,x))**2

#Inputs = np.array([0.0062,2.2950,4.6428,10])*1e-2
##x = np.linspace(0,250,1000)
##test = (RungeKutta(Equation,Inputs,0,0,x))


#time,data = readfile(r'H:\user\v\vbjellan\Documents\SEL_Folder\Kode\Kode\DPA_10uM_Ptx535m435_g10us_d550.csv')
#time,data = PrepareData(time,data,250)
#data = SmoothData(data)

#popt, pcov = curve_fit(TripletDecaySolution,time,data, Inputs)
#print(popt)
#plt.plot(time,data)

#plt.plot(time,TripletDecaySolution(time,popt[0],popt[1],popt[2],popt[3]))
#plt.plot(time,TripletDecaySolution(time,Inputs[0],Inputs[1],Inputs[2],Inputs[3]))
#plt.show()

