# test_query.py

from index_db import ollama_embed  # réutilise la fonction que vous avez déjà
import chromadb

# 1) Calculez l'embedding de la question
question = "Quel est l’ingrédient principal de ce document ?"
emb       = ollama_embed(question)  # ceci renvoie déjà une list[float]

# 2) Interrogez Chroma en fournissant directement l'embedding
client = chromadb.PersistentClient(path="./chroma")
col    = client.get_collection("rxvigilance")

print(col)

results = col.query(
    query_embeddings=[emb],
    n_results=2
)

print(results)
