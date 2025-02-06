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

initial_accuracy = 0.95

def calculate_drift(predictions, true_labels):
    current_accuracy = accuracy_score(true_labels, predictions)
    drift = abs(current_accuracy - initial_accuracy)
    return drift, current_accuracy


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_data = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])

    true_label = 1

    start_time = time.time()

    try:
        prediction = model.predict(input_data)[0]
        class_names = ['setosa', 'versicolor', 'virginica']
        class_name = class_names[prediction]

        total_predictions_counter.labels(model="iris").inc()

        if prediction == true_label:
            correct_predictions_counter.labels(model="iris").inc()

        correct = correct_predictions_counter.labels(model="iris")._value.get()
        total = total_predictions_counter.labels(model="iris")._value.get()
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0
        accuracy_gauge.labels(model="iris").set(accuracy)

        drift, current_accuracy = calculate_drift([prediction], [true_label])

        model_performance_drift_gauge.labels(model="iris").set(drift)

        prediction_success_counter.labels(model="iris").inc()

        request_duration_histogram.labels(handler="/predict", method="POST").observe(time.time() - start_time)

        return {"prediction": prediction, "class_name": class_name, "current_accuracy": current_accuracy, "drift": drift}
    except Exception as e:
        prediction_error_counter.labels(model="iris", error_type=str(e)).inc()

        request_duration_histogram.labels(handler="/predict", method="POST").observe(time.time() - start_time)

        return {"error": str(e)}
