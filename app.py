
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
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #d1e8ff !important;
    font-size: 16px;
}

.result-card {
    background: rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}

.result-temp {
    font-size: 48px;
    font-weight: bold;
    color: #ffd166;
}

div[data-testid="stFormSubmitButton"] button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover,
.stFormSubmitButton button:hover {
    color: #ffffff !important;
    filter: brightness(1.12);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
}

div[data-testid="stFormSubmitButton"] button:focus,
.stFormSubmitButton button:focus {
    color: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(147, 197, 253, 0.55) !important;
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
