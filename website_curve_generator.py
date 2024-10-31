"""
A module to generate the website.
Currently doesn't do much, but should:
- display a curve given input parameters
- find best-fit parameters given input curve (file upload)
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from de_utilities import triplet_decay_solution

st.title("TTA Curve Generator")

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
