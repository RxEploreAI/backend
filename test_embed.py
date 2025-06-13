# snippet de test rapide, dans un REPL ou un fichier test_embed.py
import chromadb

client = chromadb.PersistentClient(path="./chroma")
col    = client.get_collection("rxvigilance")

# recherche par similarité brute via texte
results = col.query(
    query_texts=["Quel est l’ingrédient principal de ce document ?"],
    n_results=2
)
print(results)
