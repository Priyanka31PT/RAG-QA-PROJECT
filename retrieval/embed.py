import json
import chromadb
from sentence_transformers import SentenceTransformer


def load_chunks(path="chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def build_vector_store(chunks, db_path="chroma_db", collection_name="research_papers", batch_size=64):
    """
    Embeds all chunks and stores them in a persistent ChromaDB collection.
    """
    print("Loading embedding model (bge-small-en)...")
    embedder = SentenceTransformer("BAAI/bge-small-en")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)

    # if the collection already exists from a previous run, delete it so we start fresh
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — deleting to rebuild fresh.")
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)

    # bge models are trained to expect this prefix on the TEXT being indexed for retrieval
    # (note: queries get a different prefix — we'll handle that in search.py)
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "paper_title": chunk["paper_title"],
            "section": chunk["section"],
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Adding to ChromaDB collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )

    print(f"Done. Collection now has {collection.count()} chunks stored.")
    return collection


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from chunks.json")
    collection = build_vector_store(chunks)

    result = collection.get(limit=1, include=["documents", "metadatas"])
    print("\nSample stored item:")
    print(f"ID: {result['ids'][0]}")
    print(f"Metadata: {result['metadatas'][0]}")
    print(f"Text preview: {result['documents'][0][:150]}...")