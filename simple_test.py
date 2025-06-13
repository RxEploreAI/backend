# simple_test.py

import os
import chromadb
from chromadb.config import Settings

# 1) Où est votre dossier d'index local ?
PERSIST_DIR = "./chroma"

# 2) Instancier le client en mode embarqué DuckDB+Parquet
client     = chromadb.PersistentClient(path=PERSIST_DIR)
# confirmation rapide
print("✅ Chroma in-process démarré en duckdb+parquet")

# 3) Récupérer la collection
col = client.get_or_create_collection("rxvigilance")
print("✅ Collection ‘rxvigilance’ prête")

# 4) Faire une simple requête (ici query_texts nécessite onnxruntime,
#    ou utilisez query_embeddings si vous préférez)
results = col.query(
    query_texts=["Quel est l’ingrédient principal ?"],
    n_results=2,
    include=["documents","metadatas","distances"]
)

# 5) Afficher les résultats
for doc, meta, dist in zip(results["documents"][0],
                           results["metadatas"][0],
                           results["distances"][0]):
    print(f"\n— Chunk ({dist:.3f}) —")
    print(f"Méta   : {meta}")
    print(f"Texte  : {doc[:200]}…")
