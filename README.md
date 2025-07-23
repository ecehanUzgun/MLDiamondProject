# MLDiamondProject

## Overview
MLDiamondProject is a machine learning web application for predicting diamond prices based on their physical and categorical features. The app uses FastAPI for the backend and a trained regression model for predictions.

## Features
- Predict diamond prices using a trained ML model
- Web interface with HTML form
- REST API endpoint for predictions

## Requirements
- Python 3.8+
- FastAPI
- pandas
- scikit-learn
- Jinja2

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place `diamond_model_complete.pkl` in the project root (contains trained model and preprocessors).
2. Start the FastAPI server:
```bash
uvicorn app:app --reload
```
3. Open your browser and go to [http://localhost:8000](http://localhost:8000) to use the web form.

## API Endpoints

### Home Page
- `GET /` : Returns the HTML form for inputting diamond features.

### Predict Price
- `POST /predict` : Accepts JSON data and returns the predicted price.

#### Example Request
```json
{
  "carat": 1.0,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS2",
  "depth": 61.5,
  "table": 55.0,
  "x": 6.5,
  "y": 6.5,
  "z": 4.0
}
```

#### Example Response
```json
{
  "predicted_price": 6500.0
}
```

## File Structure
- `app.py` : Main FastAPI application
- `diamond_model_complete.pkl` : Trained model and preprocessors
- `templates/index.html` : Web form template
- `requirements.txt` : Python dependencies
- `model_tests.py` : Test scripts
- `testdatascaled.csv` : Example/test data


