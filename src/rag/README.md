
# Customer Complaint Analysis Toolkit — `/src/rag`

This module contains the **core Retrieval-Augmented Generation (RAG) components** used to retrieve, generate, and evaluate answers grounded in CFPB consumer complaint data.

It provides a lightweight, modular implementation built around:

* Vector embeddings
* FAISS similarity search
* Instruction-tuned language models
* Explicit source attribution and evaluation

All components are designed to be **composable, inspectable, and reusable** across notebooks, scripts, and the interactive application.

---

## Module Overview

The `/rag` package includes the following components:

1. **`ComplaintVectorStore`**
   Loads complaint embeddings, builds or restores a FAISS index, and manages alignment between vectors, text, and metadata.

2. **`ComplaintRetriever`**
   Performs semantic search over the vector store and returns ranked complaint chunks with similarity scores and metadata.

3. **`ComplaintGenerator`**
   Generates grounded natural-language answers using an instruction-tuned HuggingFace model and retrieved complaint context.

4. **`ComplaintRAGPipeline`**
   Orchestrates retrieval and generation into a single question-answering interface.

5. **`evaluate`**
   Runs a predefined evaluation set and returns structured results for qualitative analysis.

6. **Predefined Evaluation Questions**
   A small, representative set of domain-specific questions covering credit cards, loans, fraud, money transfers, and savings accounts.

---

## Design Principles

* **Separation of concerns** — retrieval, generation, and orchestration are independent.
* **Persistence-first** — embeddings and FAISS indexes are loaded from disk when available.
* **Transparency** — generated answers always retain access to their source complaints.
* **Evaluation-ready** — outputs are structured for inspection, reporting, and comparison.
* **CPU-friendly by default** — works without requiring GPU acceleration.

---

## Component Responsibilities

### `vector_store.py` — Vector Store Management

* Loads embeddings and metadata from Parquet files.
* Builds or loads a FAISS index.
* Maintains alignment between:

  * Embedded vectors
  * Original complaint text
  * Associated metadata (product, company, date, complaint ID)

This component acts as the single source of truth for all retrieval operations.

---

### `retriever.py` — Semantic Retrieval

* Accepts a natural-language query.
* Encodes the query into the same embedding space.
* Performs top-k similarity search using FAISS.
* Returns a ranked list of complaint chunks, each including:

  * Text
  * Metadata
  * Similarity score

No generation or prompt logic lives here — retrieval is intentionally isolated.

---

### `generator.py` — Answer Generation

* Wraps an instruction-tuned HuggingFace model.
* Formats retrieved complaint text into a controlled prompt.
* Generates concise, grounded answers based strictly on retrieved context.

Key characteristics:

* Deterministic defaults (low temperature)
* CPU/GPU compatible
* Model-agnostic (can be swapped via configuration)

---

### `pipeline.py` — RAG Orchestration

`ComplaintRAGPipeline` connects retrieval and generation into a single interface.

Given a question, it:

1. Retrieves the most relevant complaint chunks.
2. Passes them to the generator as context.
3. Returns:

   * The generated answer
   * The original question
   * The retrieved sources and metadata

This is the primary interface used by notebooks and the Gradio app.

---

### `evaluate.py` — Pipeline Evaluation

* Runs a predefined set of representative questions.
* Executes the full RAG pipeline for each question.
* Collects:

  * Generated answers
  * Retrieved sources
  * Source metadata
* Returns results as a Pandas DataFrame for inspection or reporting.

Evaluation is **qualitative and source-focused**, emphasizing grounding and relevance rather than numeric benchmarks.

---

## Typical Workflow

1. Load or build a FAISS-backed `ComplaintVectorStore`.
2. Initialize a `ComplaintRetriever` using the vector store.
3. Initialize a `ComplaintGenerator`.
4. Combine both using `ComplaintRAGPipeline`.
5. Ask questions and inspect answers with their sources.
6. Run `evaluate` to assess performance across predefined questions.

This workflow is exercised in notebooks and exposed interactively via the Gradio app.

---

## Notes

* This module contains **no data preprocessing or embedding generation logic** — those live in `EDA.py` and `rag_prep.py`.
* Vector store artifacts are intentionally excluded from version control.
* All components are safe to reuse across scripts, notebooks, and applications.
* The design favors clarity and debuggability over abstraction-heavy frameworks.

---
