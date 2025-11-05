from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define the input schema correctly
class Message(BaseModel):
    message: str

# Load model safely
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

@app.post("/predict")
def predict(data: Message):
    try:
        if model is None:
            return {"error": "Model not loaded on server"}
        msg = str(data.message)
        prediction = model.predict([msg])
        return {"prediction": str(prediction[0])}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
