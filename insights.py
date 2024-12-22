"""Script to show how the plots in the supplementary information were made."""
import matplotlib.pyplot as plt
import numpy as np
from fitting import find_best_fit_params
from de_utilities import triplet_decay_solution

def main():
    """
    Make all the plots :)

    Variable name explanations: equation 1 from the paper is
    d[A_T]/dt = Kk_{TET}[A_S]e^{-(k_{3S}+k_{TET}[A_S])t} - k_3[A_T] - 2k_{TTA}[A_T]^2
    * k_ph = k_{3S}+k_{TET}[A_S]
    * k_2_conc_a = Kk_{TET}[A_S]
    * k_3 = k_3
    * k_4 = k_{TTA}
    * intensity = epsilon*k_{TTA}*[A_T]^2
    * time = t

    In the first analytically solvable case, dy/dt=a-by^2,
    setting the source term to a constant means
    setting k_ph=0 and k_2_conc_a=a
    dropping the second term means k_3=0
    and if we rewrite the equation as dy/dt = a-by^2
    that means k_4 = b/2.

    In the second, y = Kk_{TET}[A_S] / (k_{3S}+k_{TET}[A_S] - k_3)
                       * (e^{-k_3 t} - e^{-(k_3+k_{TET}[A_S]) t})
                     = k_2_conc_a/(k_ph-k_3) * (e^{-k_3 t} - e^{-k_ph t})

    """

    # pick arbitrary values for a and b
    a = 10000
    b = 10000

    t_min = 0
    t_max = 0.5e-3

    # get some times for the x axis
    times = np.linspace(t_min, t_max, 1000)

    # analytical values using a and b
    analytical_y = np.sqrt(a/b) * np.tanh(np.sqrt(a*b)*times)

    # TODO: think about it
    epsilon=1/5000

    # fit result
    k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
        times, epsilon*b/2*analytical_y**2,
        initial_guesses=[1, 1, 1, 1], epsilon=epsilon)
    # find fit_a and fit_b from the best-fit parameters
    # commented for now since it doesn't actually seem to converge
    fit_a = k_2_conc_a
    fit_b = 2 * k_4
    fit_y = triplet_decay_solution(epsilon=epsilon)(times, k_2_conc_a, k_ph, k_3, k_4)
    # fit_a = a
    # fit_b = b
    # fit_y = analytical_y

    # plot one curve & fit (Figure S1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_y**2, color="k", linestyle="-", label="analytical $y^2(t)$")
    ax.plot(times, fit_y**2, color="orange", linestyle="--",
            label=f"fit $y^2(t)$, (a={int(fit_a)}, b={int(fit_b)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(loc="best")
    fig.tight_layout()
    plt.show()

    # try a couple amplitude factors (Figure S2)
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    factors = [1.5, 1.0, 0.5]
    colors = ["purple", "orange", "red"]
    for factor, color in zip(factors, colors):
        analytical_y_factor = factor * analytical_y
        k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
            times, epsilon*b/2*analytical_y**2,
            initial_guesses=[1, 1, 1, 1], epsilon=epsilon)
        # find fit_a and fit_b from the best-fit parameters
        fit_a = k_2_conc_a
        fit_b = 2 * k_4
        fit_y_factor = triplet_decay_solution(epsilon=epsilon)(times, k_2_conc_a, k_ph, k_3, k_4)
        ax.plot(times, analytical_y_factor**2, color="k", linestyle="-")
        ax.plot(times, fit_y_factor**2, color=color, linestyle="--",
                label=f"fit $y^2(t)$, (a={round(fit_a)}, b={round(fit_b)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(loc="best")
    fig.tight_layout()
    plt.show()


    # sub-case 2 where y = k_2_conc_a/(k_ph-k_3) * (e^{-k_3 t} - e^{-k_ph t})
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    k_2_conc_a = 30000
    k_ph = 10000
    k_3 = 20000
    analytical_y_case2 = k_2_conc_a/(k_ph-k_3) * (np.exp(-k_3 * times) - np.exp(-k_ph * times))
    k_2_conc_a_fit, k_ph_fit, k_3_fit, k_4_fit = find_best_fit_params(
        times, analytical_y_case2,
        initial_guesses=[k_2_conc_a, k_ph, k_3, 0])
    # commented for now since it doesn't actually seem to converge
    fit_y_case2 = triplet_decay_solution(epsilon=10)(
        times, k_2_conc_a_fit, k_ph_fit, k_3_fit, k_4_fit)
    ax.plot(times, analytical_y_case2, color="k", linestyle="-")
    ax.plot(times, fit_y_case2, color="red", marker="x", label="fit $y^2(t)$")
    ax.plot(times, np.abs(fit_y_case2 - analytical_y_case2), color="green", label="|difference|")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend(loc="best")
    fig.tight_layout()
    plt.show()






if __name__ == "__main__":
    main()
