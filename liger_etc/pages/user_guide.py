

import streamlit as st
import numpy as np


def UserGuidePage():
    
    st.markdown(
        "# Liger Exposure Time Calculator User Guide"
    )
    
    st.markdown(
        "This page provides a user guide to running the Liger Exposure Time Calculator."
        )

    st.markdown("### About")

    st.markdown("""
    The Liger ETC was developed by the UCSD Optical InfraRed Laboratory in collaboration with the Keck All Sky Precision Adaptive Optics (KAPA) Science Tools Team at UC Berkeley.
                
    Contributors:
    - Bryson Cale (UCSD)
    - Shelley Wright (UCSD)
    - Sanchit Sabhlok (UCSD)
    - KAPA Group (UC Berkeley)
    """)

    st.markdown("##### The ETC uses the Liger IRIS Science Data Simulator:")

    st.markdown("""[https://github.com/oirlab/liger_iris_sim](https://github.com/oirlab/liger_iris_sim)""")

    st.markdown("### Running the ETC")

    st.markdown("""*Under development*""")

    st.markdown("### References")

    st.markdown("""

        Several utilities are pulled from existing codes to support the Liger ETC:
            
        #### OSIRIS ETC:
         
        https://oirlab.ucsd.edu/osiris/etc/
                
        #### Analytic PSF code:
            
        https://github.com/oirlab/osiris_etc/blob/master/etc_analytic_psf/etc_analytic_psf.py
            
        Created by Sanchit Sabhlok at UCSD in 2022, updated for the Liger ETC.
        
    """)

    _cols = st.columns([1, 1, 1, 1])
    with _cols[0]:
        st.image("liger_etc/static/oirlab_logo.png", width='content')
    with _cols[1]:
        st.image("liger_etc/static/liger_logo.png", width='content')
    with _cols[2]:
        st.image("liger_etc/static/keck_logo.png", width='content')
    with _cols[3]:
        st.image("liger_etc/static/mulab_logo_short.png", width='content')