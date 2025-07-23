from fastapi import FastAPI, Request # For managing web application and requests
from fastapi.templating import Jinja2Templates # For working with HTML templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

# Set templates to get Jinja2 templates from the `templates` directory
templates = Jinja2Templates(directory="templates")

with open("diamond_model_complete.pkl", "rb") as f:
    saved_data = pickle.load(f) # Load the saved model and preprocessing components
    model = saved_data["model"] # Trained regression model
    encoders = saved_data["encoders"] # LabelEncoders for categorical variables like 'cut', 'color', 'clarity'
    scaler = saved_data["scaler"] # StandardScaler or MinMaxScaler for numerical variables

## GET, POST, PUT, DELETE
#@app.get("/test")
#async def test():
#    return Response(content="Hello World")

# BaseModel is used by FastAPI for automatic validation and parsing.
# This class defines the structure of the JSON data sent to the POST/predict endpoint.
class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float # length
    y: float # width
    z: float # depth

# Home Page – GET endpoint that shows the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request): #request object is required for Jinja2 template engine
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction Endpoint – Accepts POST requests with JSON
@app.post("/predict")
async def predict(features: DiamondFeatures): # Incoming JSON data is validated and converted to Python object by DiamondFeatures class
    # print(features.model_dump)
    input_data =  pd.DataFrame([features.model_dump()])
    # print(input_data)
    for col in ['cut','color','clarity']: # Categorical variables are converted to numeric values using LabelEncoders from training
        input_data[col] = encoders[col].transform(input_data[col])
    # Numerical columns (carat, depth, x, y, z) are scaled using StandardScaler from training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return {"predicted_price": prediction[0]}