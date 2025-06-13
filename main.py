from fastapi import FastAPI, Query
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = chromadb.PersistentClient(path="./chroma")
col    = client.get_collection("rxvigilance")
app    = FastAPI()

@app.get("/search")
def search(q: str = Query(..., description="Question à rechercher")):
    res = col.query(query_texts=[q], n_results=5)
    return res

@app.post("/chat")
def chat(question: str):
    # récupérer top-chunks puis appeler Ollama via requests
    docs = col.query(query_texts=[question], n_results=3)["documents"][0]
    return {"context": docs, "answer": "Réponse générée…"}
