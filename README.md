# Cricket Thermometer: Dolbear's Law

Predict air temperature from cricket chirp rate using linear regression.

## Background

Dolbear's law (1897) states a linear relationship between the number of cricket chirps per minute and the air temperature:

`Temperature (°C) = a + b × (chirps per minute)`

This project cleans real‑world data (from Kaggle), trains a linear regression model, evaluates it on test data, and provides prediction intervals.

## Files

- `dolbear_model.py` – main Python script
- `requirements.txt` – required libraries
- `.gitignore` – ignores temporary files
- `LICENSE` – MIT license (optional)

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt

Libraries: numpy, pandas, matplotlib, scikit‑learn, scipy, joblib.
How to Run
bash

python dolbear_model.py

The script will:

    Load and clean the data

    Split into training (80%) and testing (20%)

    Train a linear regression model
    Print evaluation metrics (R², MAE)

    Show a residual plot

    Compute a 95% prediction interval for a given chirp rate (example: 130 chirps/min)

    Provide a helper function for 15‑second or 1‑minute counts

    Save the trained model as dolbear_cricket_model.pkl



Example Output

Clean data points: 151
Linear regression equation: Temperature = 12.35 + 0.2027 * chirps_per_minute
Test R² = 0.893
MAE = 1.23 °C
Prediction for 130 chirps/min: 38.70 °C [95% PI: 36.52 – 40.88]
20 chirps in 15 seconds -> 28.56 °C
Custom Prediction

Use the saved model later:
python

import joblib
model = joblib.load('dolbear_cricket_model.pkl')
temp = model.predict([[chirps_per_minute]])[0]

Or use the included function:
python

from dolbear_model import predict_temp_from_chirps
temp = predict_temp_from_chirps(chirps=20, time_unit='15sec')

