from flask import Flask, render_template, request, session
from retrieval.search import HybridRetriever
from retrieval.rerank import Reranker
from generation import contextualize_query, generate_answer, format_sources
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # needed for session support (conversation history per user)

# Load these ONCE when the server starts — not on every request (expensive to reload each time)
print("Initializing retriever and reranker (this may take a moment)...")
retriever = HybridRetriever()
reranker = Reranker()
print("Ready to serve requests.")


@app.route("/")
def home():
    session.setdefault("chat_history", [])  # start with empty history if none exists yet
    return render_template("index.html", answer=None, sources=None, chat_history=session["chat_history"])


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    chat_history = session.get("chat_history", [])

    # Step 1: rewrite follow-up questions using conversation history
    standalone_question = contextualize_query(question, chat_history)

    # Step 2: hybrid search + rerank
    candidates = retriever.hybrid_search(standalone_question, top_k=10)
    final_chunks = reranker.rerank(standalone_question, candidates, top_k=5)

    # Step 3: generate answer with citations
    answer, source_map = generate_answer(standalone_question, final_chunks)
    sources = format_sources(answer, source_map)

    # Step 4: save this turn to session history
    chat_history.append({"question": question, "answer": answer})
    session["chat_history"] = chat_history

    return render_template("index.html", answer=answer, sources=sources, chat_history=chat_history)


@app.route("/reset")
def reset():
    session.pop("chat_history", None)  # clear conversation history
    return render_template("index.html", answer=None, sources=None, chat_history=[])


if __name__ == "__main__":
    app.run(debug=True)