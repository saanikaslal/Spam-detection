from fastapi import FastAPI #type: ignore
from pydantic import BaseModel
from model_loader import predict_message #type: ignore

app = FastAPI()

class MessageInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}

@app.post("/predict")
def predict(input: MessageInput):
    result = predict_message(input.text)
    return result
