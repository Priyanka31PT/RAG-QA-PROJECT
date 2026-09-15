# Research Paper QA System — Retrieval-Augmented Generation (RAG)

A RAG-based question-answering system built over 3 foundational AI research papers (*Attention Is All You Need*, *Language Models are Few-Shot Learners*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*), featuring hybrid retrieval, cross-encoder reranking, multi-turn conversational memory, and grounded source attribution.

## Overview

This system lets users ask natural-language questions about a small corpus of research papers and get accurate, source-attributed answers — including follow-up questions that reference earlier context ("What are **its** drawbacks?").

Rather than using a naive "chunk → embed → retrieve → generate" pipeline, this project implements a multi-stage retrieval funnel designed specifically to handle complex, multi-part questions and to make every claim in the answer traceable back to its exact source (paper, section, page).

## Architecture
PDF Papers
↓
[1] Extraction (PyMuPDF) — page-aware, font-size/boldness-based heading detection
↓
[2] Chunking — section-aware splitting, token-based sizing (tiktoken), with overlap
↓
[3] Embedding (bge-small-en) → ChromaDB (persistent vector store with metadata)
↓
User Query
↓
[4] Query Contextualization — resolves pronouns/follow-ups using conversation history (GPT-4.1-nano)
↓
[5] Hybrid Retrieval — BM25 (keyword) + Vector Search (semantic), merged via Reciprocal Rank Fusion
↓
[6] Cross-Encoder Reranking — precise relevance scoring on top candidates
↓
[7] Answer Generation — GPT-4.1-nano, citation-enforcing prompt
↓
[8] Source Attribution — [Source N] markers mapped back to paper/section/page
↓
Flask Web UI (with session-based multi-turn conversation)

## Key Design Decisions

**Why hybrid search instead of plain vector search?**
Testing showed plain cosine-similarity search pulled in irrelevant noise (bibliography entries, figure-visualization dumps) simply because they contained matching keywords. Combining BM25 (exact keyword matching) with dense vector search (semantic similarity), merged via **Reciprocal Rank Fusion**, casts a wider and more accurate net — RRF avoids the problem of normalizing incompatible score scales (BM25 scores vs. cosine similarity) by fusing on rank position rather than raw scores.

**Why cross-encoder reranking on top of that?**
Hybrid search still let some low-quality matches through (both BM25 and vector search can be fooled by keyword overlap without true relevance). A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the query and each candidate chunk *together*, producing a genuinely relevance-aware score — this stage measurably eliminated irrelevant matches that survived hybrid search, confirmed via before/after testing.

**Why font-based section detection instead of keyword matching?**
An initial approach using a fixed list of expected section names ("Introduction", "Methodology", etc.) failed silently on headings it didn't recognize, and broke when heading text was split across lines during PDF extraction. Switching to font-size/boldness-based heading detection (comparing each line's font size against the paper's dominant body-text size) generalizes to any paper's structure, regardless of specific section naming conventions.

**Why multi-turn conversational support?**
Real usage involves follow-up questions ("What are *its* drawbacks?"). A query-contextualization step rewrites follow-ups into standalone questions using recent conversation history before retrieval — tested and confirmed to correctly resolve pronouns across turns.

**Why GPT-4.1-nano over GPT-4o-mini?**
Comparable or better benchmark performance on extractive QA tasks, ~33% cheaper, and faster inference — a better fit for iterative testing and live demos.

## Evaluation

Since a working pipeline doesn't guarantee *correct* answers, this project includes an evaluation layer using an LLM-as-judge approach across 10 test questions (the 5 provided sample questions + 5 additional ones), measuring two standard RAG metrics:

| Metric | Score |
|---|---|
| Faithfulness (answer grounded in retrieved sources) | 5.00 / 5 |
| Context Precision (relevance of retrieved chunks, rank-weighted) | 0.83 |

**Context Precision** is computed as Average Precision: each retrieved chunk is independently judged relevant/not-relevant by an LLM, then precision is calculated at each rank position where a relevant chunk appears, rewarding relevant chunks that rank higher. This is a more rigorous metric than a single "rate these chunks 1-5" judgment, since binary per-chunk relevance judgments are more reliable for an LLM than holistic scoring across multiple chunks at once.

**Observed pattern:** 5 of 10 questions scored perfect context precision (1.00), while narrower, more compact topics (e.g., positional encoding — 0.25) scored lower. This reflects a real precision-recall trade-off inherent to top-k retrieval: when a topic's relevant content is concentrated in only 1-2 chunks, returning a fixed top-5 necessarily pulls in some topically-adjacent-but-not-directly-relevant chunks, which lowers precision without indicating a retrieval failure.

**Methodology note:** an earlier version of this evaluation used a single holistic "rate these 3 chunks together, 1-5" retrieval-relevance judge, which produced an artificially low score (2.40/5) due to a prompt truncation bug — only the first 200 characters of each chunk were shown to the judge. This was caught by cross-referencing against the faithfulness score, which stayed high even when retrieval scores looked poor — a logical inconsistency, since a faithful, accurate answer cannot exist without genuinely relevant retrieved content. This was replaced with the more rigorous per-chunk context precision metric above. Full detailed results in `evaluation/eval_results.json`.

**Note on methodology:** an initial version of the retrieval-relevance judge scored artificially low (2.40/5) due to a prompt truncation bug (only showing the judge the first 200 characters of each chunk). Faithfulness scores stayed consistently high even when retrieval scores looked poor — this inconsistency was the signal that led to catching and fixing the bug, since a high-faithfulness answer cannot exist without genuinely relevant retrieved content. Full detailed results in `evaluation/eval_results.json`.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Extraction | PyMuPDF (fitz) | Page-aware, handles two-column academic layouts, provides font metadata |
| Chunking | Custom (tiktoken) | Section-aware, token-precise sizing — not an off-the-shelf splitter |
| Embeddings | bge-small-en (sentence-transformers) | Strong retrieval benchmark performance, free, local |
| Vector DB | ChromaDB | Native metadata support, zero-setup persistence |
| Keyword Search | rank_bm25 | Catches exact technical terms embeddings can miss |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precise relevance scoring on narrowed candidate set |
| Generation | GPT-4.1-nano (OpenAI API) | Cost-efficient, fast, strong extractive QA performance |
| Interface | Flask | Session-based conversation history, custom UI |

## Project Structure
rag_qa_project/
├── app.py # Flask app + routes
├── generation.py # Query contextualization + answer generation + citation formatting
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── preprocessing/
│ ├── extract.py # PDF extraction + font-based heading detection
│ └── chunk.py # Section-aware, token-based chunking
├── retrieval/
│ ├── embed.py # Embedding + ChromaDB storage
│ ├── search.py # Hybrid search (BM25 + vector) + RRF fusion
│ └── rerank.py # Cross-encoder reranking
├── evaluation/
│ ├── eval_questions.json
│ ├── evaluate.py
│ └── eval_results.json
├── papers/ # Source PDFs
├── chroma_db/ # Persistent vector store
└── requirements.txt

## Setup & Running

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key to a .env file
echo OPENAI_API_KEY=your_key_here > .env

# 4. Place your PDFs in papers/, then build the index
python preprocessing\chunk.py
python retrieval\embed.py

# 5. Run the app
python app.py
```
Visit `http://127.0.0.1:5000`

## Sample Interaction

**Q: What is multi-head attention?**
> Multi-head attention is a mechanism that allows a model to jointly attend to information from different representation subspaces at different positions... [Source 1]

**Follow-up: What are its uses?**
*(Automatically rewritten to: "What are multi-head attention's uses?")*
> Multi-head attention is used in the Transformer model in three different ways: 1. In encoder-decoder attention layers... [Source 1]

## Limitations & Future Work

- Section-level attribution is based on top-level headings only (not sub-sections like 3.1, 3.2) — sufficient for this corpus's QA granularity, but a finer-grained approach could improve citation precision.
- Conversation history is session-based (in-memory), not persisted across server restarts.
- Evaluation uses LLM-as-judge rather than full RAGAS metrics, due to project timeline — a production version would benefit from human-annotated ground truth for retrieval evaluation.