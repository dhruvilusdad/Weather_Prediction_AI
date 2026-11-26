# Weather Prediction AI

A Python and Streamlit application that estimates a city's temperature from its geographic coordinates using a trained machine-learning model. Enter a city name to look up its location and see the model's temperature estimate.

> This is an ML-based estimate, not a live weather forecast. It does not retrieve current weather conditions.

## Features

- City-name lookup through the Open-Meteo Geocoding API.
- Temperature prediction in Celsius using latitude and longitude.
- Streamlit interface with location details and a simple temperature category.
- Included training notebook, weather dataset, trained model, and model feature names.

## Technologies

- Python
- Streamlit
- scikit-learn and joblib
- pandas and NumPy
- requests / Open-Meteo Geocoding API
- Jupyter, Matplotlib, Seaborn, and XGBoost for the training notebook

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit application entry point |
| `app1.py` | Preserved project application script |
| `weather_prediction_model.pkl` | Trained Random Forest regression model |
| `weather_feature_names.pkl` | Ordered model feature names |
| `weather.csv` | Weather dataset used by the project |
| `weather.ipynb` | Data exploration, model training, and evaluation notebook |
| `requirements.txt` | Python dependencies for the app and notebook |

## Installation

Use Python 3.10 or later. From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

From the project directory, run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. City lookup requires an internet connection to reach the Open-Meteo Geocoding API.

## Model information

The included model is a scikit-learn `RandomForestRegressor`. It takes two features, in this order:

1. `latitude`
2. `longitude`

It predicts `temperature_celsius`. The feature names are also stored in `weather_feature_names.pkl`. The model artifact was serialized with scikit-learn 1.7.2; the dependency file pins that version for compatibility. See `weather.ipynb` for the training and evaluation workflow. No performance metrics are asserted here; consult the notebook outputs for the results recorded during training.

Only load the included pickle/joblib files from a source you trust. Serialized Python objects can execute code when loaded.
