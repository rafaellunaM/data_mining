from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import time
from starlette.responses import HTMLResponse
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.metrics import accuracy_score

app = FastAPI()

instrumentator = Instrumentator()

prediction_error_counter = Counter(
    'model_prediction_errors', 'Number of prediction errors', ['model', 'error_type']
)

request_duration_histogram = Histogram(
    'http_request_duration_seconds', 'Request duration in seconds', ['handler', 'method']
)

prediction_success_counter = Counter(
    'model_predictions_success_total', 
    'Total number of successful predictions', 
    ['model']
)

correct_predictions_counter = Counter(
    'model_correct_predictions', 'Total number of correct predictions', ['model']
)

total_predictions_counter = Counter(
    'model_total_predictions', 'Total number of predictions made', ['model']
)

prediction_rate_counter = Counter(
    'model_prediction_rate_per_second', 'Rate of predictions per second', ['model']
)

class_predictions_counter = Counter(
    'model_class_predictions_total', 'Total number of predictions per class', ['model', 'class']
)

class_errors_counter = Counter(
    'model_class_prediction_errors', 'Total number of prediction errors per class', ['model', 'class']
)

model_performance_drift_gauge = Gauge('model_performance_drift', 'Drift in model performance', ['model'])

error_rate_gauge = Gauge('model_error_rate', 'Rate of prediction errors', ['model'])

accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model', ['model'])

instrumentator.instrument(app).expose(app)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("xgboost_model_pickle.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    X1: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: float
    X11: float
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float
    X24: float
    X2_2: bool
    X3_2: bool
    X4_2: bool

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_data = np.array([[request.X1, request.X5, request.X6, request.X7, request.X8,
                            request.X9, request.X10, request.X11, request.X12, request.X13,
                            request.X14, request.X15, request.X16, request.X17, request.X18,
                            request.X19, request.X20, request.X21, request.X22, request.X23,
                            request.X24, int(request.X2_2), int(request.X3_2), int(request.X4_2)
                            ]])

    start_time = time.time()

    try:
        prediction = model.predict(input_data)[0]
        
        class_names = ['Class 0', 'Class 1']
        class_name = class_names[prediction]

        return PredictionResponse(
            prediction=prediction,
            class_name=class_name,
        )
    except Exception as e:
        return {"error": str(e)}
