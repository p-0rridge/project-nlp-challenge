from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from data_preprocessing import clean_text


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    text: str

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'logistic_regression_model.joblib')
vectorizer_path = os.path.join(BASE_DIR, 'count_vectorizer.joblib')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.post("/predict")
def predict(payload: NewsInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text leer")
    
    cleaned_text = clean_text(payload.text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    
    result = "REAL" if prediction == 1 else "FAKE"
    
    return {
        "prediction": result,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)