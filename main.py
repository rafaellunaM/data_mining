from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
from starlette.responses import HTMLResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_data = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])

    prediction = model.predict(input_data)[0]
    
    class_names = ['setosa', 'versicolor', 'virginica']
    class_name = class_names[prediction]

    return {"prediction": prediction, "class_name": class_name}
