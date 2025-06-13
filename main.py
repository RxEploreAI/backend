import os
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb

load_dotenv()
PERSIST_DIR      = os.getenv("PERSIST_DIR", "./chroma")
OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11435")
EMBED_MODEL      = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
GENERATE_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "tinyllama")

client = chromadb.PersistentClient(path=PERSIST_DIR)
col    = client.get_or_create_collection("rxvigilance")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- ici on autorise tout
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    content: str

@app.get("/search")
def search(q: str = Query(...)):
    qr = col.query(
        query_texts=[q],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    return {
        "ids":        qr["ids"],
        "documents":  qr["documents"],
        "metadatas":  qr["metadatas"],
        "distances":  qr["distances"]
    }

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Récupération des documents depuis Chroma
    qr = col.query(query_texts=[req.content], n_results=1, include=["documents"])
    chunks = qr["documents"][0] if qr["documents"] else []

    if not chunks:
        raise HTTPException(status_code=404, detail="Aucun contexte trouvé pour cette question.")

    context = "\n".join(chunks)

    # 2. Construction du prompt
    prompt = (
        "You are an expert pharmacist assistant. Answer the question below "
        "using only the information provided in the context. "
        "If the answer is not present in the context, reply with: "
        "\"I'm sorry, I could not find any relevant information in the provided context.\"\n\n"
        f"### CONTEXT:\n{context}\n\n"
        f"### QUESTION:\n{req.content}\n\n"
        "### ANSWER:"
    )

    # 3. Appel à Ollama
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model":  GENERATE_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")

    debug = {"status": resp.status_code, "body": resp.text}

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail={"ollama_error": debug})

    data = resp.json()
    answer = data.get("response", "").strip()

    return {
        "messages": [{
            "role": "assistant",
            "content": answer,
        }]
    }

@app.post("/test-prompt")
def test_prompt(req: ChatRequest):
    # Prompt direct sans Chroma
    prompt = f"Réponds simplement à la question suivante : {req.question}"
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": GENERATE_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")

    debug = {"status": resp.status_code, "body": resp.text}

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail={"ollama_error": debug})

    data = resp.json()
    answer = data.get("response", "").strip()

    return {
        "prompt": prompt,
        "answer": answer,
        "debug": debug
    }
