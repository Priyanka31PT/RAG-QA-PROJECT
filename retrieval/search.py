import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re


def load_chunks(path="chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    """Simple tokenizer for BM25 — lowercase, split on non-alphanumeric characters."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    def __init__(self, chunks_path="chunks.json", db_path="chroma_db", collection_name="research_papers"):
        print("Loading chunks for BM25 index...")
        self.chunks = load_chunks(chunks_path)
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.id_to_chunk = {c["chunk_id"]: c for c in self.chunks}

        # Build BM25 index over all chunk texts
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(name=collection_name)

        print("Loading embedding model...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en")

        print("HybridRetriever ready.\n")

    def bm25_search(self, query, top_k=10):
        """Returns list of chunk_ids ranked by BM25 score, highest first."""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunk_ids[i] for i in ranked_indices[:top_k]]

    def vector_search(self, query, top_k=10):
        """Returns list of chunk_ids ranked by vector similarity, highest first."""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[]  # we only need ids here, already returned by default
        )
        return results["ids"][0]

    def reciprocal_rank_fusion(self, bm25_ids, vector_ids, k=60):
        """Merge two ranked ID lists into one fused ranking."""
        scores = {}
        for rank, chunk_id in enumerate(bm25_ids):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        for rank, chunk_id in enumerate(vector_ids):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in fused]

    def hybrid_search(self, query, top_k=10):
        """
        Full hybrid retrieval: BM25 + vector search, fused via RRF.
        Returns list of chunk dicts (full data, not just IDs), ranked by fused score.
        """
        bm25_ids = self.bm25_search(query, top_k=top_k)
        vector_ids = self.vector_search(query, top_k=top_k)
        fused_ids = self.reciprocal_rank_fusion(bm25_ids, vector_ids)

        # return top_k full chunk dicts, in fused order
        return [self.id_to_chunk[cid] for cid in fused_ids[:top_k] if cid in self.id_to_chunk]


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_query = "What is multi-head attention?"
    results = retriever.hybrid_search(test_query, top_k=5)

    print(f"--- Hybrid Search Results for: '{test_query}' ---\n")
    for i, chunk in enumerate(results):
        print(f"{i+1}. {chunk['paper_title']} - {chunk['section']} (p.{chunk['page_number']})")
        print(f"   {chunk['text'][:150]}...\n")