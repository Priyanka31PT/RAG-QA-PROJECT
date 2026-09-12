import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from your .env file

client = OpenAI()  # automatically picks up OPENAI_API_KEY from environment


def contextualize_query(user_query, chat_history, model="gpt-4.1-nano"):
    """
    Rewrites a follow-up question into a standalone question using conversation history.
    If there's no history yet, returns the query unchanged (no rewriting needed).

    chat_history: list of dicts like [{"question": "...", "answer": "..."}, ...]
    """
    if not chat_history:
        return user_query  # first question in the session — nothing to resolve

    # build a short history string (last 3 turns is usually enough context, keeps prompt small)
    history_text = ""
    for turn in chat_history[-3:]:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    prompt = f"""Given this conversation history:

{history_text}
Rewrite the following follow-up question into a fully standalone question that does not depend on the conversation history. If the question is already standalone, return it unchanged. Only output the rewritten question, nothing else.

Follow-up question: {user_query}

Standalone question:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten


def build_context_string(chunks):
    """
    Formats retrieved chunks into a numbered source block for the prompt,
    and returns a mapping of source number -> metadata for later attribution.
    """
    context = ""
    source_map = {}

    for i, chunk in enumerate(chunks):
        source_num = i + 1
        context += f"[Source {source_num}: {chunk['paper_title']}, {chunk['section']}, p.{chunk['page_number']}]\n{chunk['text']}\n\n"
        source_map[source_num] = {
            "paper_title": chunk["paper_title"],
            "section": chunk["section"],
            "page_number": chunk["page_number"]
        }

    return context, source_map


def generate_answer(query, chunks, model="gpt-4.1-nano"):
    """
    Generates a citation-grounded answer using the retrieved chunks.
    Returns: (answer_text, source_map)
    """
    context, source_map = build_context_string(chunks)

    prompt = f"""You are a research assistant answering questions using only the provided sources.

Sources:
{context}

Question: {query}

Instructions:
- Answer using ONLY the information in the sources above.
- After each claim, cite the source number it came from, like [Source 1].
- If the question has multiple parts, make sure you address each part separately.
- If the sources don't contain enough information to answer, say so explicitly.

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    return answer, source_map


def format_sources(answer_text, source_map):
    """
    Extracts which [Source N] markers were actually used in the answer,
    and returns a clean, human-readable list of those sources.
    """
    import re
    cited_ids = set(int(x) for x in re.findall(r"\[Source (\d+)\]", answer_text))

    sources_list = []
    for cid in sorted(cited_ids):
        if cid in source_map:
            meta = source_map[cid]
            sources_list.append(
                f"[{cid}] {meta['paper_title']} — {meta['section']} (p.{meta['page_number']})"
            )
    return sources_list


if __name__ == "__main__":
    # Test the full generation flow using your existing retrieval pipeline
    from retrieval.search import HybridRetriever
    from retrieval.rerank import Reranker

    retriever = HybridRetriever()
    reranker = Reranker()

    # simulate a two-turn conversation to test contextualize_query
    chat_history = []

    q1 = "What is RAG?"
    print(f"\nUser: {q1}")
    standalone_q1 = contextualize_query(q1, chat_history)
    candidates1 = retriever.hybrid_search(standalone_q1, top_k=10)
    final_chunks1 = reranker.rerank(standalone_q1, candidates1, top_k=5)
    answer1, source_map1 = generate_answer(standalone_q1, final_chunks1)
    sources1 = format_sources(answer1, source_map1)

    print(f"Assistant: {answer1}")
    print("Sources:")
    for s in sources1:
        print(f"  {s}")

    chat_history.append({"question": q1, "answer": answer1})

    q2 = "What are its drawbacks?"
    print(f"\nUser: {q2}")
    standalone_q2 = contextualize_query(q2, chat_history)
    print(f"(Rewritten as: {standalone_q2})")
    candidates2 = retriever.hybrid_search(standalone_q2, top_k=10)
    final_chunks2 = reranker.rerank(standalone_q2, candidates2, top_k=5)
    answer2, source_map2 = generate_answer(standalone_q2, final_chunks2)
    sources2 = format_sources(answer2, source_map2)

    print(f"Assistant: {answer2}")
    print("Sources:")
    for s in sources2:
        print(f"  {s}")