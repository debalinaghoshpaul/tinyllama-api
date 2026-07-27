
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="TinyLlama Docker API")

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="cpu"
)

@app.get("/")
def home():
    return {"status": "Docker container running successfully!"}

@app.post("/predict")
def generate_text(request: QueryRequest):
    formatted_prompt = f"Question: {request.prompt}\nAnswer:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=0.7
        )
    
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"prompt": request.prompt, "generated_text": response_text}