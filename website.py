"""
A script to generate the website.
- display a curve given input parameters
- find best-fit parameters given input curve (file upload)

If you're running this yourself, make sure to download all the dependencies (in requirements.txt)
then run `streamlit run website.py`.
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from fitting import find_best_fit_params
from de_utilities import triplet_decay_solution, smooth_intensity

DATA_COLOR = "#086788"  # blue
FIT_COLOR = "#FF4B4B"  # red

# make it prettyyy
mpl.rc('lines', linewidth=2, linestyle='-')
sns.set_style("darkgrid", {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "grid.color": "black",
            "grid.linestyle": ":",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "font": "\"Source Sans Pro\", sans-serif"
            })

# header
st.title("Triplet-Triplet Annihilation")
st.write("(Looks best in light mode)")

explain = st.checkbox("Show explanation text")
if explain:
    st.write(
        """
        This code has two modes:

        ##### Parameter Estimation

        Numerically solves the non-approximative equations for triplet-triplet-annihilation
        in solutions. Gives the four parameters behind the kinetic process
        ($k_2 \\cdot [A]$, $k_{ph}$, $k_3$, $k_4$)
        by fitting against data input by the user.

        The input we expect is CSV format, with time and intensity columns, no headers.

        Both the time and intensity can be scaled before attempting to find the parameters.
        The code also expects I(t=0) = 0, so you should set $t_{min}$=0 and $t_{max}$ as appropriate
        ($t_{max}$ should be the end of the "bump").

        ##### Curve Plotter

        Given $k_2 \\cdot [A]$, $k_{ph}$, $k_3$, $k_4$, plots the non-analytical equation.

        """)

# toggle for different modes
mode = st.radio("Mode:",
                ["Parameter Estimation", "Curve Plotter"])

if mode == "Parameter Estimation":
    uploaded_file = st.file_uploader("Upload a [time, intensity] CSV file, with no header.")

    if uploaded_file is not None:

        # read the data
        dataframe = pd.read_csv(uploaded_file, header=None)
        times = dataframe[0].to_numpy(dtype=float)
        intensities = dataframe[1].to_numpy(dtype=float)

        total_duration = max(times) - min(times)

        # invert intensity if applicable
        invert = st.checkbox("Invert intensity values")
        if invert:
            intensities *= -1

        translate_i = st.checkbox("Center intensity values (set average of first 20 values = 0)")
        if translate_i:
            intensities -= np.mean(intensities[:20])

        do_smoothing = st.checkbox("Smooth data savgol(window_length=30, poly_filter=3)")
        if do_smoothing:
            intensities = smooth_intensity(intensities)


        # scale & translate axes if desired
        col1_t_multipliers, col2_t_multipliers = st.columns([1, 1])
        with col1_t_multipliers:
            multiply_t = st.checkbox("Scale time axis")
        with col2_t_multipliers:
            time_multiplier = st.number_input("t_scale", value=1.0, format="%f", step=1.0)
        if multiply_t:
            times *= time_multiplier

        col1_i_multipliers, col2_i_multipliers = st.columns([1, 1])
        with col1_i_multipliers:
            multiply_i = st.checkbox("Scale intensity axis")
        with col2_i_multipliers:
            intensity_multiplier = st.number_input("i_scale", value=1.0, format="%f", step=1.0)
        if multiply_i:
            intensities *= intensity_multiplier

        # filter time if applicable
        col1_time_filter, col2_time_filter = st.columns([1, 1])
        with col1_time_filter:
            filter_times = st.checkbox("Filter times")
        with col2_time_filter:
            t_min = st.number_input("t_min", value=min(times), format="%f",
                                    step=0.01*total_duration)
            t_max = st.number_input("t_max", value=max(times), format="%f",
                                    step=0.01*total_duration)
        if filter_times:
            if t_max > t_min:
                time_filtered_indices = (t_min < times) & (times < t_max)
                times = times[time_filtered_indices]
                intensities = intensities[time_filtered_indices]

        st.write("Data after applying (checked) filters above:")
        col1_input_data, col2_input_data = st.columns([1, 2])
        with col1_input_data:
            display_dataframe = pd.DataFrame({
                "Time": times,
                "Intensity": intensities
            })
            st.write(display_dataframe)

        with col2_input_data:
            fig = plt.figure(figsize=(5,5))
            ax = fig.gca()
            ax.plot(times, intensities, color=DATA_COLOR)
            if multiply_t:
                ax.set_xlabel(f"Time (scaled by {time_multiplier})")
            else:
                ax.set_xlabel("Time")
            if multiply_i:
                ax.set_ylabel(f"Intensity (scaled by {intensity_multiplier})")
            else:
                ax.set_ylabel("Intensity")

            fig.tight_layout()
            st.pyplot(fig=fig)

        begin = st.checkbox("Begin calculation")

        if begin:
            st.write("Calculating!")
            k_2_conc_a, k_ph, k_3, k_4 = find_best_fit_params(
                times, intensities,
                initial_guesses=[1, 1, 1, 1])
            print(k_2_conc_a, k_ph, k_3, k_4)
            st.write("## Best-Fit Parameters")
            st.text(f"k_2[A]: {k_2_conc_a}")
            st.text(f"k_ph:   {k_ph}")
            st.text(f"k_3:    {k_3}")
            st.text(f"k_4:    {k_4}")

            best_fit_intensities = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)

            st.write("#### Did it work? Best-fit parameter curve:")
            fig = plt.figure(figsize=(5,5))
            ax = fig.gca()
            ax.plot(times, intensities, color=DATA_COLOR)
            ax.plot(times, best_fit_intensities, color=FIT_COLOR)
            ax.set_xlabel("Time")
            ax.set_ylabel("Intensity")
            fig.tight_layout()
            st.pyplot(fig=fig)


else:
    col1, col2 = st.columns([1, 3])

    with col1:
        k_2_conc_a = float(st.text_input("Enter k_2[A]: ", value=10))
        k_3 = float(st.text_input("Enter k_3: ", value=10))
        k_4 = float(st.text_input("Enter k_4: ", value=30))
        k_ph = float(st.text_input("Enter k_ph: ", value=10))
        t_min = float(st.text_input("Enter t_min: ", value=0))
        t_max = float(st.text_input("Enter t_max: ", value=1))

    with col2:
        times = np.linspace(t_min, t_max, 1000)
        intensities = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)

        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.plot(times, intensities, color=FIT_COLOR)
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        fig.tight_layout()
        st.pyplot(fig=fig)
