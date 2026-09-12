import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="research_papers")
embedder = SentenceTransformer("BAAI/bge-small-en")

test_query = "What is multi-head attention?"
query_embedding = embedder.encode(test_query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"\nScore: {dist:.3f} | {meta['paper_title']} - {meta['section']} (p.{meta['page_number']})")
    print(doc[:150], "...")