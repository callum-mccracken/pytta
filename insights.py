"""Script to show how the plots in the supplementary information were made."""
import matplotlib.pyplot as plt
import numpy as np
from de_utilities import triplet_decay_solution

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
    k_2_conc_a = a  # k_2 times concentration of A
    k_3 = 0
    k_4 = b/2
    k_ph = 0
    t_min = 0
    t_max = 0.5e-3

    # get some times for the x axis
    times = np.linspace(t_min, t_max, 1000)

    # find the y values with a fit
    fit_y = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)

    # trim off the final point that gets funky due to numerical nonsense
    times = times[:-1]
    fit_y = fit_y[:-1]

    # use a and b to find the analytical values
    analytical_y = np.sqrt(a/b) * np.tanh(np.sqrt(a*b)*times)

    # plot
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_y, color="k", linestyle="-", label="analytical $y^2(t)$")
    ax.plot(times, fit_y, color="orange", linestyle="--", label="fit $y^2(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(loc="best")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
