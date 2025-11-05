from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Detection API")

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Welcome to the Spam Detection API! Go to /docs to test the model."}

@app.post("/predict")
def predict(data: InputData):
    try:
        message = data.message
        prediction = model.predict([message])
        return {"prediction": prediction[0]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

