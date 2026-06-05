from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# load env
load_dotenv()

# configure gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

app = FastAPI(title="Emotion Predictor (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    emotions: List[str]


def detect_emotions(text: str):

    prompt = f"""
    Detect emotions present in the following text.

    Possible emotions:
    joy, sadness, anger, fear, love, surprise, neutral, suicidal

    Return ONLY a JSON list.

    Example:
    ["joy","love"]

    Text:
    {text}
    """

    response = model.generate_content(prompt)

    try:
        emotions = json.loads(response.text)
    except:
        emotions = ["neutral"]

    return emotions


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):

    emotions = detect_emotions(payload.text)

    if not emotions:
        emotions = ["neutral"]

    return {"emotions": emotions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)