from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class Message(BaseModel):
    message: str

# Load model
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

def to_label(pred):
    """
    Map the model's raw prediction to a human label.
    - If model predicts strings like 'spam'/'ham' -> normalize to 'spam'/'not_spam'
    - If model predicts 0/1 -> 0 => 'not_spam', 1 => 'spam'
    """
    # string labels
    if isinstance(pred, str):
        return "spam" if pred.lower() == "spam" else "not_spam"
    # numeric labels
    try:
        p = int(pred)
        return "spam" if p == 1 else "not_spam"
    except Exception:
        return str(pred)

@app.post("/predict")
def predict(data: Message):
    try:
        if model is None:
            return {"error": "Model not loaded on server"}

        text = str(data.message)
        raw_pred = model.predict([text])[0]

        label = to_label(raw_pred)

        # confidence if the model supports predict_proba
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            # try to pick the 'spam' column if class names are available
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                # normalize class names to choose 'spam' column when present
                spam_idx = None
                for i, c in enumerate(classes):
                    if str(c).lower() == "spam" or str(c) == "1":
                        spam_idx = i
                        break
                if spam_idx is not None:
                    score = float(proba[spam_idx])
                else:
                    # fallback: if label == spam, take max prob, else 1-max
                    score = float(np.max(proba)) if label == "spam" else float(1 - np.max(proba))
            else:
                # no classes_: use max prob aligned with label
                score = float(np.max(proba)) if label == "spam" else float(1 - np.max(proba))

        return {
            "label": label,            # "spam" or "not_spam"
            "raw_prediction": str(raw_pred),  # keeps what the model actually returned
            "score": score             # 0–1 (may be null if model has no predict_proba)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
