
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from fitting import find_best_fit_params
from de_utilities import triplet_decay_solution, smooth_intensity


        # read the data
dataframe = pd.read_csv(r'C:\Users\Victoria Madeleine\Desktop\MikaelPaper\pytta-main\pytta-main\example_data\DPA_10uM_Ptx535m435_g10us_d550.csv', header=None)
times = dataframe[0].to_numpy(dtype=float)
intensities = dataframe[1].to_numpy(dtype=float)
intensities *= -1
intensities -= np.mean(intensities[:20])
intensities = smooth_intensity(intensities)
times *= 1e6
t_min, t_max = 0,200
time_filtered_indices = (t_min < times) & (times < t_max)
times = times[time_filtered_indices]

intensities = intensities[time_filtered_indices]

plt.plot(times,intensities)
plt.show()

k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
                times, intensities,
                initial_guesses=[1, 1, 1, 1])
print(k_2_conc_a, k_ph, k_3, k_4)
best_fit_intensities = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)

plt.plot(times,intensities)
plt.plot(times,best_fit_intensities)
plt.show()


        #total_duration = max(times) - min(times)

        ## invert intensity if applicable
        #invert = st.checkbox("Invert intensity values")
        #if invert:
        #    intensities *= -1

        #translate_i = st.checkbox("Center intensity values (set average of first 20 values = 0)")
        #if translate_i:
        #    intensities -= np.mean(intensities[:20])

        #do_smoothing = st.checkbox("Smooth data savgol(window_length=30, poly_filter=3)")
        #if do_smoothing:
        #    intensities = smooth_intensity(intensities)