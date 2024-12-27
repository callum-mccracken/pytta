"""Script to show how the plots in the supplementary information were made."""
import matplotlib.pyplot as plt
import numpy as np
from de_utilities import triplet_decay_solution,triplet_decay_solution_2
from fitting import  find_best_fit_params,find_best_fit_params_2
def main():
    """
    Make all the plots :)

    Variable name explanations: equation 1 from the paper is
    d[A_T]/dt = Kk_{TET}[A_S]e^{-(k_{3S}+k_{tet}[A_S])t} - k_3[A_T] - 2k_{TTA}[A_T]^2
    * k_ph = k_{3S}+k_{tet}[A_S]
    * k_2_conc_a = Kk_{TET}[A_S]  # k_{TET} = k_4 yes, but we can optimize separately since K is indep.
    * k_3 = k_3
    * k_4 = k_{TTA}
    * intensity = [A_T]
    * time = t

    In the analytically solvable case,
    setting the source term to a constant means
    setting k_ph=0 and k_2_conc_a=a
    dropping the second term means k_3=0
    and if we rewrite the equation as dy/dt = a-by^2
    that means k_4 = b/2.
    """

    # pick arbitrary values for a and b
    a = 10000
    b = 10000

    # define other variables such that dy/dt = a-by^2
    K = 1
    k_TET = a
    k_3s = 0
    k_3A = 0
    k_TTA = b/2
    t_min = 0
    t_max = 0.5e-3


    # get some times for the x axis
    times = np.linspace(t_min, t_max, 1000)
    
    analytical_y = np.sqrt(a/b) * np.tanh(np.sqrt(a*b)*times)
    # find the y values with a fit
    fit_y = find_best_fit_params(times,analytical_y**2,initial_guesses = [10000,0,0,5000])

    print(f'The fit found the following values: K*k_TET = {fit_y[0]}, k_3s = {fit_y[1]}, k_3A = {fit_y[2]}, k_TTA = {fit_y[3]}')
     
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_y, color="red", linestyle="-", label="analytical $y(t)$")
    ax.plot(times, analytical_y**2, color="k", linestyle="-", label="analytical $y^2(t)$")
    ax.plot(times, triplet_decay_solution(times,*fit_y), color="orange", linestyle="--", label="fit $y^2(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(loc="best")
    fig.tight_layout()
    plt.show()

    analytical_y1 = analytical_y
    analytical_y2 = analytical_y*0.5
    analytical_y3 = analytical_y*1.5

    fit1 = (find_best_fit_params(times, analytical_y**2,[10000,0,0,5000]))
    fit2 = (find_best_fit_params(times, analytical_y2**2,[10000,0,0,5000]))
    fit3 = (find_best_fit_params(times, analytical_y3**2,[10000,0,0,5000]))

    fit_y1 = triplet_decay_solution(times, fit1[0]  , fit1[1], fit1[2], fit1[3])
    fit_y2 = triplet_decay_solution(times, fit2[0]  , fit2[1], fit2[2], fit2[3])
    fit_y3 = triplet_decay_solution(times, fit3[0]  , fit3[1], fit3[2], fit3[3])

    print(f'The fits for the three analytical values were: K*k_TET = {fit1[0]},{fit2[0]},{fit3[0]}, k_3s = {fit1[1]},{fit2[1]},{fit3[1]}, k_3A = {fit1[2]},{fit2[2]},{fit3[2]}, k_TTA = {fit1[3]},{fit2[3]},{fit3[3]}')


    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_y**2, color="red", linestyle="-", label="analytical $y^2(t)$")
    ax.plot(times, analytical_y2**2, color="red", linestyle="-", label="analytical $0.5*y^2(t)$")
    ax.plot(times, analytical_y3**2, color="red", linestyle="-", label="analytical $0.5*y^2(t)$")

    ax.plot(times, fit_y1, color="orange", linestyle="--", label="fit $y^2(t)$")
    ax.plot(times, fit_y2, color="orange", linestyle="--", label="fit $0.5*y^2(t)$")
    ax.plot(times, fit_y3, color="orange", linestyle="--", label="fit $1.5*y^2(t)$")

    plt.show()


    times = np.linspace(0,1e-3,1000)
    k_TET       = 1.5*50*1e3
    K           = 1
    k_TET       = k_TET*K
    k_3s        = 15000
    k_3A        = 80000
    k_TTA       = 0

    fit_y = triplet_decay_solution_2(times, K, k_TET, k_3s, k_3A, k_TTA)
    analytical_fit = (K*k_TET/(k_3s+k_TET-k_3A))*(np.exp(-k_3A*times)-np.exp(-(k_3s+k_TET)*times))

    fit1 = (find_best_fit_params_2(times, analytical_fit**2,[1,10000,0,0,5000]))
    print(fit1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_fit**2, color="blue", linestyle="-", label="analytical $y^2(t)$")
    ax.plot(times, fit_y, color="red", linestyle="-", label="analytical $0.5*y^2(t)$")

    plt.show()

if __name__ == "__main__":
    main()
