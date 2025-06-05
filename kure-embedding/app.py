from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import os

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/kure-v1")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
model.eval()

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "KURE-v1"

@app.post("/v1/embeddings")
async def create_embedding(req: EmbeddingRequest):
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # [batch, seq, hidden] -> [batch, hidden] (평균 pooling)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
    data = [
        {"embedding": emb, "index": i, "object": "embedding"}
        for i, emb in enumerate(embeddings)
    ]
    return {"object": "list", "data": data, "model": req.model} 