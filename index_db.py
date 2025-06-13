import os
import glob
import requests
from lxml import etree
import chromadb
from dotenv import load_dotenv
from requests.exceptions import Timeout, ConnectionError, HTTPError

# — 1. Chargement de la config
load_dotenv()
DATA_DIR     = os.getenv("DATA_DIR", "./data")
PERSIST_DIR  = os.getenv("PERSIST_DIR", "./chroma")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
BASE_URL     = os.getenv("OLLAMA_URL", "http://localhost:11435")

# — 2. Parser le NXML
def parse_nxml(path):
    tree  = etree.parse(path)
    title = tree.findtext('.//article-title') or ""
    paras = tree.findall('.//body//p')
    body  = "\n".join(p.text for p in paras if p.text)
    return title.strip(), body.strip()

# — 3. Chunker le texte
def chunk_text(text, chunk_size=500, overlap=50):
    words, chunks = text.split(), []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

# — 4. Récupérer un embedding depuis Ollama
def ollama_embed(text: str) -> list[float]:
    attempts = [
        ("/api/embed",      {"model": OLLAMA_MODEL, "input": text}),
        ("/api/embeddings", {"model": OLLAMA_MODEL, "prompt": text}),
        ("/v1/embeddings",  {"model": OLLAMA_MODEL, "input": [text]}),
    ]
    for ep, payload in attempts:
        url = BASE_URL.rstrip("/") + ep
        print(f"→ Essai {url}")
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 404:
                print("   404 → route absente, essai suivant")
                continue
            resp.raise_for_status()
            data = resp.json()

            if "embeddings" in data:
                emb = data["embeddings"]
                return emb[0] if isinstance(emb, list) and emb else emb
            if "embedding" in data:
                return data["embedding"]
            if "data" in data and isinstance(data["data"], list):
                return data["data"][0].get("embedding")

            raise ValueError(f"Format inattendu {data}")
        except (Timeout, ConnectionError):
            print("   ⏱ Timeout/ConnError, essai suivant")
            continue
        except HTTPError as e:
            raise RuntimeError(f"HTTP {resp.status_code} sur {url}: {resp.text}") from e

    raise RuntimeError(f"Aucun endpoint valable sur {BASE_URL}")

# — 5. Scanner les fichiers et préparer les chunks
documents = []
for fp in glob.glob(os.path.join(DATA_DIR, "*.nxml")):
    title, body = parse_nxml(fp)
    full = f"{title}\n\n{body}"
    for idx, chunk in enumerate(chunk_text(full)):
        documents.append({
            "id":       f"{os.path.basename(fp)}_chunk{idx}",
            "text":     chunk,
            "metadata": {"source": os.path.basename(fp), "title": title}
        })

print(f"⚙️  Prêt à indexer {len(documents)} chunks…")

# — 6. Générer les embeddings
embeddings = [ollama_embed(d["text"]) for d in documents]

# — 7. Indexer dans Chroma + FAISS
client     = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection("rxvigilance")

try:
    collection.upsert(
        ids=[d["id"] for d in documents],
        embeddings=embeddings,
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )
    print("✅ Upsert OK : vos chunks sont désormais idempotents.")
except Exception as e:
    print("❌ Erreur lors de upsert:", e)
# persist() est optionnel avec le nouveau client
count = collection.count()
print(f"📊 Nombre total de vecteurs dans 'rxvigilance' : {count}")

print(f"✅ Indexation Ollama complète : {len(documents)} chunks indexés.")
