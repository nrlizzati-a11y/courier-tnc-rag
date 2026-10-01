# Courier T&C RAG: Advanced Retrieval-Augmented Generation on Delivery Terms & Conditions

A retrieval-augmented generation (RAG) system that answers natural-language questions about a real-world legal document — Shopee's SPX Express Terms of Service — built on Google Cloud. This project extends Google's "Advanced RAG Techniques" codelab by swapping the original dataset for a domain-specific legal document and running a structured evaluation to compare retrieval strategies, rather than relying on spot-checked examples.

> This is a technical portfolio project, not a legal or compliance tool. SPX's terms may change after the date this data was retrieved; do not rely on this system for an actual delivery dispute.

## Why this project

Legal and policy documents are a common real-world RAG use case — dense, clause-numbered text where users ask questions in everyday language that rarely matches the document's own wording ("my package never arrived" vs. "loss or non-delivery of your shipment"). This project tests, with real measurements, which retrieval techniques actually help close that gap.

## Architecture

```
HTML source (SPX T&C)
    → cleaned & extracted text
    → chunked (4 strategies, compared independently)
    → embedded (Vertex AI gemini-embedding-001)
    → stored in Cloud SQL (PostgreSQL + pgvector)
    → retrieved via vector similarity search
    → optionally reranked (Vertex AI Ranking API)
    → optionally preceded by query transformation (HyDE / step-back)
```

**Stack:** Python, LangChain, Google Cloud (Vertex AI, Cloud SQL), PostgreSQL + pgvector

## Dataset

- **Source:** SPX Express (Shopee) Terms of Service — a single HTML document, ~58,000 characters, retrieved [28-Sep-2026].
- **Note:** The raw document is not included in this repo (see `.gitignore`); `sources.json` records the source URL so it can be re-downloaded. This keeps the repo free of redistributing a third party's copyrighted text while keeping the project fully reproducible.

## Methodology

### 1. Chunking strategies compared

| Strategy | Approach |
|---|---|
| Character | Fixed-size splitting on character count |
| Recursive | Falls back through paragraph → line → word breaks |
| Token | Fixed-size splitting by token count (~4x larger average chunk than character/recursive) |
| Section (custom) | Splits on clause-number boundaries (e.g. "5.4", "12.2") to keep legal clauses intact |

### 2. Retrieval enhancements tested

- **Reranking:** Vertex AI's Ranking API reorders the top 10 vector search results down to the top 3.
- **Query transformation:** HyDE (generates a hypothetical answer passage, then embeds that instead of the raw question) and step-back prompting (rewrites a specific question into a more general one).

### 3. Evaluation

A 16-question test set (13 answerable, 3 deliberately unanswerable) covering direct lookups, paraphrased/natural-language questions, and questions with no answer in the source document. "Hit" = the correct clause's expected keywords appear in the top-3 retrieved chunks. See `eval_questions.json` and `run_eval.py`.

## Results

| Configuration | Hit Rate | Hits/Total |
|---|---|---|
| Character chunking (vector search only) | 85% | 11/13 |
| Recursive chunking (vector search only) | 85% | 11/13 |
| Token chunking (vector search only) | 92% | 12/13 |
| Section chunking (vector search only) | 92% | 12/13 |
| **Token chunking + reranking** | **100%** | **13/13** |
| **Section chunking + reranking** | **100%** | **13/13** |

On this 13-question evaluation set, adding reranking closed the one remaining gap for both token and section chunking. This is a small test set — a larger one would be needed to confirm the result holds generally — but the pattern across every stage was consistent enough to be a reasonable first signal.

## Key findings

**1. Chunking strategy sets a ceiling; reranking closes the remaining gap.**
Character and recursive chunking missed questions where the correct clause (e.g. clause 12.2, on liability after delivery) didn't even appear in the top-3 vector search results — not because a similar clause outranked it, but because the smaller chunks separated the relevant language from the words the query used. Token and section chunking's larger or clause-aware chunks kept enough context intact to surface the right clause. Reranking then fixed the one failure mode chunking couldn't: a vocabulary mismatch between how a user phrases a question and how the document is worded (see finding 2).

**2. Query transformation helps with vocabulary gaps, but can also introduce fabricated specifics.**
For the question "My package never arrived, can I get my money back?", the original query retrieved irrelevant clauses, while both HyDE and step-back prompting correctly surfaced the compensation clause — by rewriting the question into the document's own vocabulary ("Compensation," "Complaint").

However, for "How many days do I have to complain?", HyDE generated a passage stating a flat "seven (7) calendar days." The actual terms (clause 5.8) specify **5 calendar days for damaged shipments** and **30 calendar days for lost shipments** — HyDE's number matched neither. The passage was still useful for retrieval (it pulled in relevant clauses), because the embedding captured the general topic even though the specific figure was wrong. The practical implication: **a HyDE-generated passage should only ever be used internally to drive retrieval, never shown to a user as an answer.**

**3. Reranking achieved the same fix as query transformation, without the hallucination risk.**
Both techniques solved the same vocabulary-gap failure (the "never arrived" question). Reranking did it by reordering real, existing chunks — it can't invent facts, only mis-rank them. This makes it the lower-risk fix for this specific failure mode, though query transformation remains useful for cases where no retrieved chunk is close enough for reranking to work with.

**4. Token and section chunking perform identically on this document, with different trade-offs.**
Both reached the same hit rate at every stage. Section (clause-aware) chunks are more interpretable for human review, since they align with the document's own numbering — useful if a result ever needs auditing. Token chunking required no custom logic to implement.

## Limitations

- Single document, single carrier — results may not generalize to other legal documents or writing styles.
- 13 answerable questions is a small evaluation set; hit rate differences of 1 question (85% vs. 92%) should be read cautiously.
- Evaluation measures keyword presence in retrieved chunks, not generated-answer correctness — it tests retrieval quality, not the final LLM response end-to-end.
- The reranked 100% result has not been tested with a held-out question set; the same questions were used to design and evaluate the pipeline.

## Project structure

```
.
├── sources.json              # Source document URL(s) — no copyrighted text stored
├── download_docs.py           # Downloads and cleans source HTML/PDF
├── ingest_data.py              # Chunks and embeds documents into Cloud SQL (4 strategies)
├── query_chunking.py           # Compares raw vector search across chunking strategies
├── query_reranking.py          # Compares vector search with/without reranking
├── query_transformation.py     # Compares original query vs. HyDE vs. step-back
├── eval_questions.json         # 16-question evaluation set with expected keywords
├── run_eval.py                  # Runs the evaluation across all chunking strategies
├── run_eval_rerank.py           # Runs the evaluation with reranking added
├── enable_pgvector.py           # One-time setup: enables the pgvector extension
├── requirements.txt
└── .env.example                 # Template for required environment variables
```

## Setup

1. Create a Google Cloud project with the Vertex AI, Cloud SQL, and Discovery Engine APIs enabled.
2. Copy `.env.example` to `.env` and fill in your project ID, region, and a database password.
3. Create the Cloud SQL instance and enable pgvector:
   ```bash
   python enable_pgvector.py
   ```
4. Download and ingest the source document:
   ```bash
   python download_docs.py
   python ingest_data.py
   ```
5. Run comparisons and evaluation:
   ```bash
   python query_chunking.py
   python query_reranking.py
   python query_transformation.py section
   python run_eval.py
   python run_eval_rerank.py
   ```

## Acknowledgments

Built on top of Google's [Advanced RAG Techniques codelab](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/8-advanced-rag-methods/advanced-rag-methods) (Production-Ready AI with Google Cloud path), licensed under Apache 2.0 (code) and CC BY 4.0 (text). This project replaces the original dataset and ingestion logic, adds a clause-aware chunking strategy, and adds a structured evaluation framework not present in the original lab.