"""Script to show how the plots in the supplementary information were made."""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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
    epsilon=1/5000  # has to be tweaked manually, but this value works for the plots here
    # get some times for the x axis
    times = np.linspace(t_min, t_max, 1000)
    # analytical values using a and b
    analytical_y = np.sqrt(a/b) * np.tanh(np.sqrt(a*b)*times)
    # fit result
    analytical_intensity = analytical_y**2  # well epsilon * k_4 * this but we'll normalize
    k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
        times, analytical_intensity,
        initial_guesses=[a, 0, 0, b/2], epsilon=epsilon)
    # find fit_a and fit_b from the best-fit parameters
    fit_a = k_2_conc_a
    fit_b = 2 * k_4
    fit_intensity = triplet_decay_solution(epsilon=epsilon)(times, k_2_conc_a, k_ph, k_3, k_4)
    # plot one curve & fit (Figure S1)
    # note the normalization since y is only proportional to intensity
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ax.plot(times, analytical_intensity/max(analytical_intensity), color="k", linestyle="-",
            label="analytical")
    ax.plot(times, fit_intensity/max(fit_intensity), color="orange", linestyle="--",
            label=f"fit, (a={int(fit_a)}, b={int(fit_b)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Intensity")
    ax.tick_params(direction='in', length=6, top=True, right=True)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig("images/S1.png")
    ax.clear()

    # try a couple amplitude factors (Figure S2)
    factors = [1.5, 1, 0.5]
    colors = ["purple", "orange", "red"]
    for factor, color in zip(factors, colors):
        analytical_intensity_i = factor * analytical_intensity
        k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
            times, analytical_intensity_i,
            initial_guesses=[a, 0, 0, b/2], epsilon=epsilon)
        # find fit_a and fit_b from the best-fit parameters
        fit_a = k_2_conc_a
        fit_b = 2 * k_4
        print(fit_a * fit_b)
        fit_intensity_i = triplet_decay_solution(epsilon=epsilon)(times, k_2_conc_a, k_ph, k_3, k_4)
        ax.plot(times, analytical_intensity_i, color="k", linestyle="-")
        # normalize the fits so they line up with the analytical things
        ax.plot(times, fit_intensity_i/np.max(fit_intensity_i)*np.max(analytical_intensity_i),
                color=color, linestyle="--",
                label=f"fit, (a={round(fit_a)}, b={round(fit_b)})")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig("images/S2.png")
    ax.clear()

    # is this actually invariant?
    # yes, seems like it was, we didn't include this plot in the paper
    # start = 1
    # end = 100
    # num = 1000
    # factors = np.linspace(start, end, num=num)
    # products = np.zeros_like(factors)
    # for i, factor in tqdm(enumerate(factors)):
    #     analytical_intensity_i = factor * analytical_intensity
    #     k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
    #         times, analytical_intensity_i,
    #         initial_guesses=[a, 0, 0, b/2], epsilon=epsilon)
    #     # find fit_a and fit_b from the best-fit parameters
    #     fit_a = k_2_conc_a
    #     fit_b = 2 * k_4
    #     products[i] = fit_a * fit_b
    # ax.plot(factors, products, "k")
    # ax.set_xlabel("factor")
    # ax.set_ylabel("ab product")
    # fig.tight_layout()
    # plt.savefig(f"images/ab_products_{start}_{end}_{num}.png")
    # ax.clear()


    # sub-case 2 where y = k_2_conc_a/(k_ph-k_3) * (e^{-k_3 t} - e^{-k_ph t}) (Figure S3)
    t_min = 0
    t_max = 1e-3
    times = np.linspace(t_min, t_max, 1000)
    k_2_conc_a = 30000
    k_ph = 10000
    k_3 = 20000
    analytical_y_case2 = k_2_conc_a/(k_ph-k_3) * (np.exp(-k_3 * times) - np.exp(-k_ph * times))
    analytical_intensity_case2 = analytical_y_case2**2
    k_2_conc_a_fit, k_ph_fit, k_3_fit, k_4_fit = find_best_fit_params(
        times, analytical_intensity_case2,  # if k_4 is zero shouldn't intensity = 0?
        initial_guesses=[k_2_conc_a, k_ph, k_3, 1])
    # commented for now since it doesn't actually seem to converge
    fit_intensity_case2 = triplet_decay_solution(epsilon=epsilon)(
        times, k_2_conc_a_fit, k_ph_fit, k_3_fit, k_4_fit)
    ax.plot(times, analytical_intensity_case2/np.max(analytical_intensity_case2),
            color="k", linestyle="-", label="analytical")
    ax.plot(times, fit_intensity_case2/np.max(fit_intensity_case2),
            color="orange", linestyle="--", label="fit")
    ax.plot(times, np.abs(analytical_intensity_case2/np.max(analytical_intensity_case2)
                          - fit_intensity_case2/np.max(fit_intensity_case2)),
            color="green", linestyle="-", label="|difference|")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig("images/S3.png")
    plt.close()






if __name__ == "__main__":
    main()
