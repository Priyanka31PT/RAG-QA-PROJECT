import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.search import HybridRetriever
from retrieval.rerank import Reranker
from generation import generate_answer, build_context_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def load_eval_questions(path="evaluation/eval_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_faithfulness(question, answer, context, model="gpt-4.1-nano"):
    """
    Uses an LLM to judge whether the answer is faithful to the provided context
    (i.e., not hallucinating information not present in the sources).
    Returns a score 1-5 and a brief reason.
    """
    prompt = f"""You are evaluating a RAG system's answer for faithfulness to its source material.

Context (retrieved sources):
{context}

Question: {question}

Generated Answer: {answer}

Rate how faithful the answer is to the context on a scale of 1-5:
5 = fully grounded, every claim is supported by the context
3 = mostly grounded, but some minor unsupported details
1 = largely hallucinated, makes claims not found in the context

Respond in this exact format:
Score: <number>
Reason: <one sentence>"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    score = None
    reason = ""
    for line in result.split("\n"):
        if line.startswith("Score:"):
            try:
                score = int(line.replace("Score:", "").strip())
            except ValueError:
                score = None
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()

    return score, reason


def judge_chunk_relevance(question, chunk, model="gpt-4.1-nano"):
    """
    Judges a SINGLE chunk as relevant (1) or not relevant (0) to the question.
    Binary judgment — more reliable than a single 1-5 score over multiple chunks at once.
    """
    prompt = f"""Question: {question}

Retrieved passage:
{chunk['text']}

Is this passage relevant and useful for answering the question? Respond with only one word: "yes" or "no"."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()
    return 1 if "yes" in result else 0


def compute_context_precision(question, chunks):
    """
    Computes context precision (as Average Precision) over the ranked retrieved chunks.
    chunks: ranked list (best first) of chunk dicts actually sent to the LLM for generation.
    Returns: (average_precision, list of per-chunk relevance labels)
    """
    relevance_labels = [judge_chunk_relevance(question, chunk) for chunk in chunks]

    relevant_count = 0
    precision_sum = 0.0

    for k, rel in enumerate(relevance_labels, start=1):
        if rel == 1:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k

    if relevant_count == 0:
        return 0.0, relevance_labels

    average_precision = precision_sum / relevant_count
    return average_precision, relevance_labels


def run_evaluation():
    print("Loading evaluation questions...")
    eval_questions = load_eval_questions()

    print("Initializing retriever and reranker...")
    retriever = HybridRetriever()
    reranker = Reranker()

    results = []

    for i, item in enumerate(eval_questions):
        question = item["question"]
        print(f"\n[{i+1}/{len(eval_questions)}] Evaluating: {question}")

        candidates = retriever.hybrid_search(question, top_k=10)
        final_chunks = reranker.rerank(question, candidates, top_k=5)
        answer, source_map = generate_answer(question, final_chunks)
        context, _ = build_context_string(final_chunks)

        faithfulness_score, faithfulness_reason = judge_faithfulness(question, answer, context)
        context_precision, relevance_labels = compute_context_precision(question, final_chunks)

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "context_precision": context_precision,
            "chunk_relevance_labels": relevance_labels
        })

        print(f"  Faithfulness: {faithfulness_score}/5 — {faithfulness_reason}")
        print(f"  Context Precision: {context_precision:.2f}")

    avg_faithfulness = sum(r["faithfulness_score"] for r in results if r["faithfulness_score"]) / len(results)
    avg_context_precision = sum(r["context_precision"] for r in results) / len(results)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Average Faithfulness:        {avg_faithfulness:.2f}/5")
    print(f"Average Context Precision:   {avg_context_precision:.2f}")
    print(f"Total questions evaluated:   {len(results)}")

    with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_faithfulness": avg_faithfulness,
                "avg_context_precision": avg_context_precision,
                "total_questions": len(results)
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print("\nDetailed results saved to evaluation/eval_results.json")


if __name__ == "__main__":
    run_evaluation()