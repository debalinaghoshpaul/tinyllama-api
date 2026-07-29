from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TinyLlama Mock API")

class Query(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "API is online and healthy!"}

@app.post("/predict")
def predict(query: Query):
    # Simulating a light model response to bypass free cloud RAM constraints
    return {
        "prompt": query.prompt,
        "response": f"Mock TinyLlama output for prompt: '{query.prompt}'"
    }