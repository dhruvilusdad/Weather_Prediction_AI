
import streamlit as st
import joblib
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

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

city = st.text_input(
    "City Name",
    placeholder="Example: Rajkot, Ahmedabad, Mumbai..."
)

st.caption("Enter any supported city name to predict its temperature.")

# =====================================
# GEOCODING
# =====================================

if st.button("🌡️ Predict Temperature", use_container_width=True):

    if not city.strip():
        st.warning("Please enter a city name.")

    else:
        with st.spinner("Finding city location..."):

            try:
                geolocator = Nominatim(
                    user_agent="weather_prediction_ai_app"
                )

                location = geolocator.geocode(
                    city.strip(),
                    timeout=15
                )

                if location is None:
                    st.error("City not found. Please try another city.")

                else:
                    latitude = location.latitude
                    longitude = location.longitude

                    # Prepare model input
                    input_data = pd.DataFrame(
                        [[latitude, longitude]],
                        columns=["latitude", "longitude"]
                    )

                    # Prediction
                    prediction = model.predict(input_data)[0]

                    # Result
                    st.success("Prediction Completed Successfully!")

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h3>🌍 {city.title()}</h3>
                            <p>Predicted Temperature</p>
                            <div class="result-temp">
                                {prediction:.2f} °C
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(" ")

                    st.info(
                        f"📍 Location: {location.address}"
                    )

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

            except (GeocoderTimedOut, GeocoderUnavailable):
                st.error("Location service unavailable. Please try again.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

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
