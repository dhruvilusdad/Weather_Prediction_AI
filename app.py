
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

theme_mode = st.radio(
    "Display theme",
    ("Dark", "Light"),
    horizontal=True,
    key="weather_display_theme",
)

if theme_mode == "Dark":
    theme_tokens = """
    color-scheme: dark;
    --page-background:
        radial-gradient(ellipse at 12% 8%, rgba(34, 211, 238, 0.12), transparent 38%),
        radial-gradient(ellipse at 88% 18%, rgba(124, 58, 237, 0.16), transparent 36%),
        linear-gradient(135deg, #071426 0%, #0c1e39 52%, #111a3b 100%);
    --surface: rgba(8, 24, 48, 0.86);
    --surface-strong: rgba(7, 20, 38, 0.96);
    --text: #FFFFFF;
    --muted-text: #C4D5E8;
    --accent: #22D3EE;
    --border: rgba(117, 215, 255, 0.28);
    --shadow: rgba(0, 0, 0, 0.28);
    --placeholder: rgba(255, 255, 255, 0.58);
    --button-background: linear-gradient(100deg, #0E7490 0%, #4F46E5 58%, #7C3AED 100%);
    --title-glow: rgba(34, 211, 238, 0.2);
    """
else:
    theme_tokens = """
    color-scheme: light;
    --page-background:
        radial-gradient(ellipse at 10% 5%, rgba(37, 99, 235, 0.12), transparent 40%),
        radial-gradient(ellipse at 90% 12%, rgba(124, 58, 237, 0.08), transparent 38%),
        linear-gradient(135deg, #F4FAFF 0%, #EAF4FF 52%, #F8FAFF 100%);
    --surface: rgba(255, 255, 255, 0.78);
    --surface-strong: rgba(255, 255, 255, 0.94);
    --text: #10233F;
    --muted-text: #425875;
    --accent: #2563EB;
    --border: rgba(37, 99, 235, 0.2);
    --shadow: rgba(30, 64, 110, 0.12);
    --placeholder: rgba(16, 35, 63, 0.5);
    --button-background: linear-gradient(100deg, #2563EB 0%, #4F46E5 58%, #7C3AED 100%);
    --title-glow: rgba(37, 99, 235, 0.12);
    """

st.markdown("""
<style>
:root {
    THEME_TOKENS
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stElementContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: transparent !important;
}

.stApp {
    background: var(--page-background) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3, h4, p, label,
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: var(--text) !important;
    text-shadow: 0 3px 22px var(--title-glow);
}

.subtitle {
    text-align: center;
    color: var(--accent) !important;
    font-size: 16px;
}

[data-testid="stRadio"] {
    margin-bottom: 0.35rem;
}

[data-testid="stRadio"] > label,
[data-testid="stRadio"] [role="radio"],
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}

[data-testid="stRadio"] [role="radio"] > div:first-child {
    border-color: var(--border) !important;
}

[data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

[data-testid="stForm"] {
    background: linear-gradient(135deg, var(--surface), var(--surface-strong)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 12px 32px var(--shadow);
    backdrop-filter: blur(14px);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, var(--surface), var(--surface-strong)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 28px var(--shadow);
    backdrop-filter: blur(14px);
}

[data-testid="stTextInput"] > label,
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] p {
    color: var(--text) !important;
}

[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background: var(--surface-strong) !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] input {
    background: var(--surface-strong) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    caret-color: var(--accent);
}

[data-testid="stTextInput"] input::placeholder {
    color: var(--placeholder) !important;
    opacity: 1;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--border) !important;
}

[data-testid="stAlert"] {
    background: transparent !important;
    color: var(--text) !important;
}

[data-testid="stAlert"] [data-testid="stAlertContainer"] {
    background: linear-gradient(120deg, var(--surface-strong), var(--surface)) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px var(--shadow);
    backdrop-filter: blur(12px);
}

[data-testid="stAlert"] [data-testid^="stAlertContent"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    background: transparent !important;
    color: var(--text) !important;
}

[data-testid="stAlert"] [data-testid="stAlertContainer"] svg {
    color: var(--accent) !important;
    fill: var(--accent) !important;
}

.result-card {
    background:
        radial-gradient(ellipse at 10% 0%, var(--border), transparent 48%),
        linear-gradient(135deg, var(--surface-strong), var(--surface)) !important;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 14px 36px var(--shadow);
    backdrop-filter: blur(14px);
}

.result-card h3,
.result-card p {
    color: var(--text) !important;
}

.result-temp {
    font-size: 48px;
    font-weight: bold;
    color: var(--accent) !important;
    text-shadow: 0 0 24px var(--title-glow);
}

div[data-testid="stFormSubmitButton"] button,
.stFormSubmitButton button {
    background: var(--button-background) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button *,
.stFormSubmitButton button * {
    color: #FFFFFF !important;
}

div[data-testid="stFormSubmitButton"] button:hover,
.stFormSubmitButton button:hover {
    color: #FFFFFF !important;
    filter: brightness(1.12);
    transform: translateY(-2px);
    box-shadow: 0 10px 26px var(--shadow);
}

div[data-testid="stFormSubmitButton"] button:focus,
.stFormSubmitButton button:focus {
    color: #FFFFFF !important;
    box-shadow: 0 0 0 3px var(--border) !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--accent) !important;
}

[data-testid="stDivider"] {
    border-color: var(--border) !important;
}

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.25rem 1rem 2rem;
    }

    .main-title {
        font-size: 30px;
    }

    .result-card {
        padding: 20px 14px;
    }

    .result-temp {
        font-size: 42px;
    }
}
</style>
""".replace("THEME_TOKENS", theme_tokens), unsafe_allow_html=True)

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
