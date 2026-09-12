from sentence_transformers import CrossEncoder
from retrieval.search import HybridRetriever


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("Loading cross-encoder reranker model...")
        self.model = CrossEncoder(model_name)
        print("Reranker ready.\n")

    def rerank(self, query, chunks, top_k=5):
        """
        chunks: list of chunk dicts (from hybrid_search, already fused via RRF)
        Returns: top_k chunk dicts, re-ordered by cross-encoder relevance score, highest first.
        """
        if not chunks:
            return []

        # build (query, chunk_text) pairs — this is what the cross-encoder scores together
        pairs = [(query, chunk["text"]) for chunk in chunks]

        scores = self.model.predict(pairs)

        # attach scores to chunks, then sort by score descending
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # return just the chunk dicts (with score attached for visibility), top_k only
        reranked = []
        for chunk, score in scored_chunks[:top_k]:
            chunk_with_score = dict(chunk)  # copy, don't mutate original
            chunk_with_score["rerank_score"] = float(score)
            reranked.append(chunk_with_score)

        return reranked


if __name__ == "__main__":
    retriever = HybridRetriever()
    reranker = Reranker()

    test_query = "What is multi-head attention?"

    # get a wider candidate pool from hybrid search first
    candidates = retriever.hybrid_search(test_query, top_k=10)

    # then rerank down to the best 5
    final_results = reranker.rerank(test_query, candidates, top_k=5)

    print(f"--- Reranked Results for: '{test_query}' ---\n")
    for i, chunk in enumerate(final_results):
        print(f"{i+1}. [score: {chunk['rerank_score']:.3f}] {chunk['paper_title']} - {chunk['section']} (p.{chunk['page_number']})")
        print(f"   {chunk['text'][:150]}...\n")