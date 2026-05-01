import streamlit as st


def ExposureInputs():

    st.markdown('### Exposure Config')

    calc_options = {
        'snr':  'Acquired SNR',
        'flux': 'Limiting Flux',
    }

    col_calc, col_params, col_detector = st.columns([1.2, 1.4, 1.4])

    # ── What to calculate ─────────────────────────────────────────────────
    with col_calc:
        calc_type = st.radio(
            label='**Calculate:**',
            options=list(calc_options.keys()),
            index=0,
            format_func=lambda s: calc_options[s],
            key='calc_type',
        )

    # ── Integration / SNR inputs ──────────────────────────────────────────
    with col_params:
        if calc_type == 'snr':
            st.number_input(
                label='**Integration Time (s)**',
                value=10.0, step=1.0, min_value=0.0,
                key='input_itime',
                help='Integration time per frame (s)',
            )
            st.number_input(
                label='**Number of Frames**',
                value=1, step=1, min_value=0,
                key='input_n_frames',
                help='Number of frames to coadd',
            )
        elif calc_type == 'flux':
            st.number_input(
                label='**Integration Time (s)**',
                value=1.0, step=1.0, min_value=0.0,
                key='input_itime',
                help='Integration time per frame (s)',
            )
            st.number_input(
                label='**Number of Frames**',
                value=1, step=1, min_value=0,
                key='input_n_frames',
                help='Number of frames to coadd',
            )
            st.number_input(
                label='**Target SNR**',
                value=5.0, step=1.0, min_value=0.1,
                format='%.1f',
                key='desired_snr',
                help='Minimum SNR threshold to solve for limiting flux',
            )

    # ── Detector noise parameters ─────────────────────────────────────────
    with col_detector:
        st.number_input(
            label='**Dark Current (e⁻/s):**',
            value=0.025,
            step=0.005,
            min_value=0.0,
            format='%.4f',
            key='dark_current',
            help='Detector dark current in electrons per second per pixel',
        )
        st.number_input(
            label='**Read Noise (e⁻ RMS):**',
            value=5.0,
            step=0.5,
            min_value=0.0,
            format='%.2f',
            key='read_noise',
            help='Detector read noise in electrons RMS per pixel per read',
        )


def get_exposure_params():
    state = st.session_state
    calc_type = state.get('calc_type', 'snr')

    params = dict(
        calc_type=calc_type,
        dark_current=float(state.get('dark_current', 0.025)),
        read_noise=float(state.get('read_noise', 5.0)),
    )

    params['input_itime']    = state.get('input_itime')
    params['input_n_frames'] = state.get('input_n_frames')

    if calc_type == 'flux':
        params['desired_snr'] = float(state.get('desired_snr', 5.0))

    return params