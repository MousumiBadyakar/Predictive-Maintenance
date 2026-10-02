import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0b0f14;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 16px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-bottom: 6px;
    }

    .section-description {
        color: #9aa4b2;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Prediction card */
    .prediction-box {
        background: linear-gradient(
            135deg,
            #151e28,
            #10161d
        );
        border: 1px solid #354251;
        border-radius: 18px;
        padding: 35px;
        text-align: center;
        margin-bottom: 25px;
    }

    .prediction-label {
        color: #9aa4b2;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .prediction-number {
        font-size: 52px;
        font-weight: 700;
        line-height: 1.1;
        margin: 8px 0;
    }

    .prediction-unit {
        color: #9aa4b2;
        font-size: 14px;
    }

    /* Model info card */
    .model-card {
        background: #121820;
        border: 1px solid #27313c;
        border-radius: 16px;
        padding: 24px;
        margin-top: 25px;
    }

    /* Buttons */
    .stButton > button {
        height: 48px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #121820;
        border: 1px solid #27313c;
        padding: 18px;
        border-radius: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #697481;
        font-size: 13px;
        margin-top: 45px;
        padding-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚙️ Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Remaining Useful Life prediction for aircraft engines '
    'using a time-series LSTM model.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1, 2],
    gap="large"
)


# ============================================================
# LEFT COLUMN — ENGINE ANALYSIS
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-title">Engine Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Select an engine from the C-MAPSS test dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    engine_id = st.number_input(
        "Engine ID",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )

    analyze_button = st.button(
        "🔍 Analyze Engine",
        use_container_width=True
    )


    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    st.markdown(
        '<div class="model-card">',
        unsafe_allow_html=True
    )

    st.markdown("### Model Information")

    st.caption("Current registered model")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.metric(
            "Model",
            "LSTM"
        )

        st.metric(
            "Input Features",
            "15 sensors"
        )

    with model_col2:

        st.metric(
            "Sequence",
            "50 cycles"
        )

        st.metric(
            "RUL Cap",
            "125 cycles"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================

with right_column:

    if not analyze_button:

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            'READY FOR ANALYSIS'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Select an engine"
        )

        st.caption(
            "Choose an Engine ID and click "
            "**Analyze Engine** to generate an RUL prediction."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    else:

        try:

            response = requests.get(
                f"{API_URL}/predict/{engine_id}",
                timeout=60
            )

            # ------------------------------------------------
            # Successful prediction
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_rul = result["predicted_rul"]
                observed_cycles = result["observed_cycles"]
                model_name = result["model"]
                model_version = result["model_version"]


                st.success(
                    "Prediction completed successfully."
                )


                # ------------------------------------------------
                # Main prediction
                # ------------------------------------------------

                st.markdown(
                    '<div class="prediction-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="prediction-label">'
                    'ESTIMATED REMAINING USEFUL LIFE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="prediction-number">'
                    f'{predicted_rul:.2f}'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="prediction-unit">'
                    'operating cycles'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # Engine details
                # ------------------------------------------------

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:

                    st.metric(
                        "Engine ID",
                        engine_id
                    )

                with detail_col2:

                    st.metric(
                        "Observed Cycles",
                        observed_cycles
                    )


                # ------------------------------------------------
                # Model used
                # ------------------------------------------------

                st.markdown("### Model Used")

                st.info(
                    f"**{model_name}**  \n"
                    f"Registered Model Version: **v{model_version}**"
                )


            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            else:

                try:

                    error_message = response.json().get(
                        "detail",
                        "Prediction failed."
                    )

                except Exception:

                    error_message = "Prediction failed."

                st.error(error_message)


        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "⚠️ FastAPI server is not running. "
                "Start it using: "
                "`uvicorn api:app --reload`"
            )


        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The prediction request timed out."
            )


        # ----------------------------------------------------
        # Other errors
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'NASA C-MAPSS FD001 • LSTM • MLflow • FastAPI'
    '</div>',
    unsafe_allow_html=True
)