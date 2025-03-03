from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import nest_asyncio

# Initialize FastAPI
app = FastAPI()

# Fix event loop issues in Google Colab
nest_asyncio.apply()

# Load model and tokenizer
MODEL_PATH = "/content/gpt2-ml-model"  # Ensure this path is correct

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
except Exception:
    print("⚠️ Model not found locally, downloading from Hugging Face...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure model is in evaluation mode
model.eval()

# Define request model
class PromptInput(BaseModel):
    prompt: str

# API to generate text
@app.post("/generate/")
async def generate_text(data: PromptInput):
    input_text = data.prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():  # Ensure no gradient calculations
        output_ids = model.generate(input_ids, max_length=100)
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"response": generated_text}

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run FastAPI inside Google Colab
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

import threading
thread = threading.Thread(target=run)
thread.start()
