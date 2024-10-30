"""
A module to generate the website.
Currently doesn't do much, but should:
- display a curve given input parameters
- find best-fit parameters given input curve (file upload)
"""

import numpy as np
import streamlit as st
from de_utilities import triplet_decay_solution

def make_and_plot_curve(k_2, k_3, k_4, k_ph, t_min, t_max):
    """Given k_2, k_3, k_4, k_ph, make a curve plot."""
    times = np.linspace(t_min, t_max, 1000)
    conc_a = 1
    k_2_conc_a = k_2 * conc_a
    intensities = triplet_decay_solution(times, k_2_conc_a, k_ph, k_3, k_4)
    st.pyplot(times, intensities)


def make_streamlit_thing():
    """Make a little streamlit site"""

    st.title("Curve Generator")
    st.write("""
    Enter your parameters and we'll make you a curve!
    """)
    k_2 = float(st.text_input("Enter k_2: ", value=10))
    k_3 = float(st.text_input("Enter k_3: ", value=10))
    k_4 = float(st.text_input("Enter k_4: ", value=30))
    k_ph = float(st.text_input("Enter k_1: ", value=10))
    t_min = float(st.text_input("Enter t_min: ", value=0))
    t_max = float(st.text_input("Enter t_max: ", value=1))


    st.button(
        "Generate",
        on_click=make_and_plot_curve,
        args=(k_2, k_3, k_4, k_ph, t_min, t_max))



if __name__ == "__main__":
    make_streamlit_thing()
