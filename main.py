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
def predict(data: Message):
    text = [data.message]
    text_vectorized = vectorizer.transform(text)
    prediction = model.predict(text_vectorized)[0]
    return {"prediction": prediction}
