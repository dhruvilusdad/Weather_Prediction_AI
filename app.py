
import streamlit as st
import joblib
import pandas as pd
import requests

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Weather Prediction AI",
    page_icon="🌤️",
    layout="centered"
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():
    return joblib.load("weather_prediction_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Check your project folder.")
    st.stop()
    raise

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
:root {
    color-scheme: dark;
    --navy-glass: rgba(8, 20, 43, 0.82);
    --navy-deep: rgba(5, 15, 34, 0.92);
    --glass-border: rgba(117, 215, 255, 0.28);
    --cyan: #75D7FF;
    --white: #FFFFFF;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stElementContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: transparent !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 12% 8%, rgba(37, 99, 235, 0.2), transparent 38%),
        radial-gradient(ellipse at 88% 18%, rgba(124, 58, 237, 0.18), transparent 36%),
        linear-gradient(135deg, #071426 0%, #0c1e39 52%, #111a3b 100%) !important;
    background-attachment: fixed !important;
    color: var(--white);
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3, h4, p, label, [data-testid="stCaptionContainer"] {
    color: var(--white) !important;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: var(--white) !important;
    text-shadow: 0 3px 22px rgba(117, 215, 255, 0.2);
}

.subtitle {
    text-align: center;
    color: var(--cyan) !important;
    font-size: 16px;
}

[data-testid="stForm"] {
    background: linear-gradient(
        135deg,
        rgba(10, 28, 57, 0.9),
        rgba(29, 28, 69, 0.78)
    ) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.24);
    backdrop-filter: blur(14px);
}

[data-testid="stTextInput"] > label,
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] p {
    color: var(--white) !important;
}

[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background: transparent !important;
}

[data-testid="stTextInput"] input {
    background: var(--navy-deep) !important;
    color: var(--white) !important;
    border: 1px solid rgba(117, 215, 255, 0.36) !important;
    border-radius: 12px !important;
    caret-color: var(--cyan);
}

[data-testid="stTextInput"] input::placeholder {
    color: rgba(255, 255, 255, 0.58) !important;
    opacity: 1;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(117, 215, 255, 0.2) !important;
}

[data-testid="stAlert"] {
    background: linear-gradient(
        120deg,
        rgba(8, 25, 51, 0.96),
        rgba(27, 29, 67, 0.9)
    ) !important;
    color: var(--white) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: var(--white) !important;
}

[data-testid="stAlert"] svg {
    color: var(--cyan) !important;
    fill: var(--cyan) !important;
}

.result-card {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(37, 99, 235, 0.2), transparent 48%),
        linear-gradient(135deg, rgba(8, 23, 49, 0.94), rgba(27, 27, 66, 0.88)) !important;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid var(--glass-border);
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(14px);
}

.result-card h3,
.result-card p {
    color: var(--white) !important;
}

.result-temp {
    font-size: 48px;
    font-weight: bold;
    color: var(--cyan) !important;
    text-shadow: 0 0 24px rgba(117, 215, 255, 0.22);
}

div[data-testid="stFormSubmitButton"] button,
.stFormSubmitButton button {
    background: linear-gradient(100deg, #2563eb 0%, #6d4ce8 55%, #7c3aed 100%) !important;
    color: var(--white) !important;
    border: 1px solid rgba(185, 210, 255, 0.25) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px rgba(68, 73, 210, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button *,
.stFormSubmitButton button * {
    color: var(--white) !important;
}

div[data-testid="stFormSubmitButton"] button:hover,
.stFormSubmitButton button:hover {
    color: var(--white) !important;
    filter: brightness(1.12);
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(79, 70, 229, 0.42);
}

div[data-testid="stFormSubmitButton"] button:focus,
.stFormSubmitButton button:focus {
    color: var(--white) !important;
    box-shadow: 0 0 0 3px rgba(117, 215, 255, 0.35) !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--cyan) !important;
}

[data-testid="stDivider"] {
    border-color: rgba(117, 215, 255, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    '<div class="main-title">🌤️ Weather Prediction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Temperature Prediction System</p>',
    unsafe_allow_html=True
)

st.divider()

# =====================================
# CITY INPUT
# =====================================

st.subheader("📍 Enter Your City")

with st.form("city_prediction_form"):
    city = st.text_input(
        "City Name",
        placeholder="Example: Rajkot, Ahmedabad, Mumbai..."
    )
    submitted = st.form_submit_button(
        "🌡️ Predict Temperature",
        use_container_width=True,
        type="primary"
    )

st.caption("Enter any supported city name to predict its temperature.")
# =====================================
# GEOCODING
# =====================================

if submitted:
    if not city.strip():
        st.warning("Please enter a city name.")

    else:
        with st.spinner("Finding city location..."):
            try:
                response = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": city.strip(),
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                    timeout=10,
                )
                response.raise_for_status()
                geocoding_data = response.json()
            except requests.Timeout:
                st.error("City lookup timed out. Please try again.")
            except requests.RequestException:
                st.error("City lookup service is unavailable. Please try again.")
            else:
                results = (
                    geocoding_data.get("results", [])
                    if isinstance(geocoding_data, dict)
                    else None
                )

                if not isinstance(results, list):
                    st.error("City lookup returned an invalid response. Please try again.")
                elif not results:
                    st.error("City not found. Please try another city.")
                elif not isinstance(results[0], dict):
                    st.error("City lookup returned an invalid response. Please try again.")
                else:
                    location = results[0]
                    latitude = location.get("latitude")
                    longitude = location.get("longitude")

                    if (
                        not isinstance(latitude, (int, float))
                        or isinstance(latitude, bool)
                        or not isinstance(longitude, (int, float))
                        or isinstance(longitude, bool)
                    ):
                        st.error("The city lookup returned invalid coordinates.")
                    else:
                        location_name = ", ".join(
                            str(part)
                            for part in (
                                location.get("name"),
                                location.get("admin1"),
                                location.get("country"),
                            )
                            if part
                        )

                        # Keep the trained model input and feature order unchanged.
                        input_data = pd.DataFrame(
                            [[latitude, longitude]],
                            columns=["latitude", "longitude"]
                        )
                        prediction = model.predict(input_data)[0]

                        st.success("Prediction Completed Successfully!")

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <h3>🌍 {location_name}</h3>
                                <p>Predicted Temperature</p>
                                <div class="result-temp">
                                    {prediction:.2f} °C
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.write(" ")
                        st.info(f"📍 Location: {location_name}")
                        st.write(
                            f"Latitude: {latitude:.4f} | "
                            f"Longitude: {longitude:.4f}"
                        )

                        if prediction >= 30:
                            st.warning("🔥 Model Prediction: Hot Weather")
                        elif prediction >= 20:
                            st.info("☀️ Model Prediction: Pleasant Weather")
                        else:
                            st.info("❄️ Model Prediction: Cool Weather")

# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown(
    "<p style='text-align:center;'>"
    "Developed using Python | Streamlit | Scikit-learn"
    "</p>",
    unsafe_allow_html=True
)

st.caption(
    "Note: This is an ML-based temperature estimate, "
    "not a live weather forecast."
)
