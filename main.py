from fastapi import FastAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Загрузка модели (пример с русскоязычной моделью)
MODEL_NAME = "inkoziev/saiga_llama3_8b"  # или любая другая компактная модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.get("/ask")
async def ask(question: str):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}
