from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Define request model
class Message(BaseModel):
    message: str

# Define /predict endpoint
@app.post("/predict")
def predict_spam(data: Message):
    prediction = model.predict([data.message])[0]
    result = "spam" if prediction == 1 else "not spam"
    return {"prediction": result}
