# Tests

This folder contains **unit tests** for the core data processing, embedding, and retrieval-augmented generation (RAG) components of the project.
The tests are designed to validate **logic, data flow, and component wiring** without depending on large datasets, external services, or brittle model behavior.

The goal is to ensure the pipeline behaves correctly while keeping tests **fast, deterministic, and easy to run**.

---

## What Is Tested

### 1. EDA and Text Preprocessing (Task 1)

Tests for the EDA pipeline focus on **text cleaning and normalization**, including:

* Removal of boilerplate phrases and URLs
* Lowercasing and punctuation removal
* Stopword filtering and lemmatization
* Producing valid, non-empty normalized text

These tests avoid loading large Parquet files and instead validate the core text-processing logic directly using small, in-memory inputs.

---

### 2. Text Chunking and Embedding Pipeline (Task 2)

Tests for the embedding pipeline validate the **end-to-end flow** up to embedding generation:

* Loading a cleaned CSV file
* Stratified sampling by product category
* Text chunking using fixed chunk sizes
* Embedding generation with a mocked embedding model

The embedding model is **mocked** so that:

* No model is downloaded
* No GPU is required
* Tests run quickly and consistently

An optional smoke test verifies that FAISS index files and metadata artifacts are written correctly.

---

### 3. Retrieval-Augmented Generation Pipeline (Task 3)

Tests for the RAG pipeline validate **component integration and data flow**, not language model quality.

These tests ensure that:

* The FAISS-backed vector store loads successfully
* Semantic retrieval returns ranked complaint chunks
* The RAG pipeline returns a structured response containing:

  * The original question
  * A generated answer
  * Retrieved source documents with metadata
* The evaluation utility executes and returns a non-empty results table

The RAG tests act as **sanity and wiring checks**, confirming that retrieval, generation, and orchestration work together correctly without asserting on exact model output.

---

## Testing Strategy

* **Unit-focused** — Tests validate logic and structure, not semantic correctness.
* **Small inputs** — Tiny datasets or fixtures are used wherever possible.
* **Mocked dependencies** — External models are mocked when appropriate to avoid downloads.
* **Deterministic assertions** — Tests check shapes, presence of fields, and successful execution.
* **Fast execution** — All tests are expected to complete in seconds on CPU.

This strategy reflects best practices for testing data pipelines and ML-adjacent systems.

---

## Running the Tests

From the project root, run:

```bash
pytest -v
```

To run only embedding-related tests:

```bash
pytest tests/test_embedding_processor.py -v
```

To run only RAG pipeline tests:

```bash
pytest tests/test_rag_pipeline.py -v
```

---

## Notes

* Progress bars (e.g., `tqdm`) and logging are left enabled for visibility but do not affect test outcomes.
* Vector store artifacts must exist locally for RAG tests to run.
* These tests are intended to catch regressions in preprocessing, sampling, retrieval, and pipeline wiring — not to evaluate model performance or answer quality.

