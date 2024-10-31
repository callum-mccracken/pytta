"""
A module to generate the website.
Currently doesn't do much, but should:
- display a curve given input parameters
- find best-fit parameters given input curve (file upload)
"""
from cycler import cycler
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_theme import st_theme
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from fitting import find_best_fit_params
from de_utilities import triplet_decay_solution

# make it prettyyy
theme = st_theme()
if theme is not None:
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        theme["primaryColor"], 'g', 'b', 'y'])
    mpl.rc('lines', linewidth=1, linestyle='-')
    sns.set_style("darkgrid", {
                "figure.facecolor": theme["backgroundColor"],
                "axes.facecolor": theme["secondaryBackgroundColor"],
                "axes.edgecolor": theme["textColor"],
                "axes.labelcolor": theme["textColor"],
                "grid.color": theme["textColor"],
                "grid.linestyle": ":",
                "xtick.color": theme["textColor"],
                "ytick.color": theme["textColor"],
                "text.color": theme["textColor"],
                "font": theme["font"]
                })

# header
st.title("Triplet-Triplet Annihilation")
st.write("Bla bla bla this is a description of what the website does.")

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
            ax.plot(times, intensities)
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

        k_2_conc_a, k_3, k_4, k_ph = find_best_fit_params(times, intensities)

        st.write("## Best-Fit Parameters")
        st.text(f"k_2[A]: {k_2_conc_a}")
        st.text(f"k_3:    {k_3}")
        st.text(f"k_4:    {k_4}")
        st.text(f"k_ph:   {k_ph}")

        best_fit_intensities = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)

        st.write("#### Did it work? Best-fit parameter curve:")
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.plot(times, best_fit_intensities)
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
        ax.plot(times, intensities)
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        fig.tight_layout()
        st.pyplot(fig=fig)
