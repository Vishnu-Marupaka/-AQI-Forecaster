# 🌍 Air Quality Index (AQI) Predictor

## Overview
This project is an end-to-end Machine Learning web application designed to predict the Air Quality Index (AQI) based on various environmental pollutants. It uses a **Random Forest Regressor** to model the complex relationships between air quality features and outputs a continuous AQI score.

The model is served through a **Flask** web application and containerized using **Docker** for easy deployment on platforms like Hugging Face Spaces.

## 🚀 Features
* **Machine Learning Model:** Utilizes `RandomForestRegressor` from Scikit-Learn for high-accuracy predictions.
* **Dynamic Web Interface:** A user-friendly HTML form that automatically adapts to the features the model was trained on.
* **Data Visualization:** Includes a training script that generates Actual vs. Predicted plots and Feature Importance charts to evaluate model performance.
* **Production Ready:** Fully containerized with a `Dockerfile` and served via `gunicorn`.

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **Web Framework:** Flask
* **Machine Learning:** Scikit-Learn, Pandas
* **Deployment:** Docker, Gunicorn, Hugging Face Spaces

## 📁 Repository Structure
```text
├── app.py                 # The Flask web application
├── train_model.py         # Script to train the ML model and generate graphs
├── model.joblib           # The compiled Random Forest model
├── features.joblib        # Saved list of input features to structure the web form
├── Dockerfile             # Container configuration for deployment
├── requirements.txt       # Python dependencies
└── templates/
    └── index.html         # Front-end UI for the web application
