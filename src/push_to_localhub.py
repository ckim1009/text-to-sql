import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()
MODEL_PATH = os.getenv("MODEL_OUTPUT_PATH")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

app = FastAPI()

@app.post("/text2sql")
async def text2sql(query: dict):
    question = query["question"]
    prompt = f"Translate the following question to SQL: {question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"sql": sql}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
