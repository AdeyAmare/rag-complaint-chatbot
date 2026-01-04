
# Tests

This folder contains **unit tests** for the core data processing and embedding components of the project.
The tests are designed to validate **logic and data flow** without relying on large datasets, external services, or heavy model downloads.

The goal is to ensure the pipeline behaves correctly while keeping tests **fast, deterministic, and easy to run**.

---

## What Is Tested

### 1. EDA and Text Preprocessing (Task 1)

Tests for the EDA pipeline focus on **text cleaning and normalization**, including:

* Removal of boilerplate phrases and URLs
* Lowercasing and punctuation removal
* Stopword filtering and lemmatization
* Producing valid, non-empty normalized text

These tests avoid loading large Parquet files and instead validate the core text-processing logic directly.

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

An optional smoke test checks that FAISS index and metadata files are written correctly.

---

## Testing Strategy

* **Unit-focused**: Tests validate logic, not model quality or performance.
* **Small inputs**: Tiny in-memory datasets are used.
* **Mocked dependencies**: External libraries like sentence-transformers are mocked where appropriate.
* **Fast execution**: All tests should complete in seconds.

This approach reflects best practices for testing data and ML pipelines.

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

---

## Notes

* Progress bars (e.g., `tqdm`) and logging are left enabled for visibility but do not affect test outcomes.
* These tests are intended to catch regressions in preprocessing, sampling, and chunking logic rather than validate semantic correctness.

