# Notebooks

This directory contains Jupyter notebooks that **run, validate, and demonstrate the data pipelines and RAG system** implemented in `src/`.
Notebooks are used for execution, inspection, evaluation, and visualization — **not core logic**.

---

## Task 1 Notebook – EDA & Preprocessing

* Executes `EDAProcessor` on the raw CFPB complaints dataset.
* Processes the data in large batches.
* Writes the cleaned and filtered complaints to CSV.
* Generates and saves EDA figures:

  * Product distribution
  * Narrative availability
  * Raw vs cleaned narrative length distributions

**Output**

* `data/processed/filtered_complaints.csv`
* Figures saved to `reports/figures/`

---

## Task 2 Notebook – Embedding & Vector Store Preparation

* Executes `EmbeddingProcessor` from `rag_prep.py`.
* Performs stratified sampling on the cleaned dataset.
* Chunks complaint narratives and generates embeddings.
* Builds and saves a FAISS vector index and associated metadata.
* Produces diagnostic visualizations:

  * Sampled complaints per product
  * Chunks per complaint
  * Chunk length distribution
  * PCA projections of embeddings (overall and by product)

**Output**

* Embeddings and metadata stored as Parquet
* FAISS index files saved to disk
* Figures saved to `reports/figures/`

---

## Task 3 Notebook – RAG Pipeline Execution & Evaluation (`rag_pipeline.ipynb`)

* Initializes and loads the FAISS-backed `ComplaintVectorStore`.

* Runs semantic retrieval using `ComplaintRetriever`.

* Generates natural language answers with `ComplaintGenerator`.

* Connects retrieval and generation via `ComplaintRAGPipeline`.

* Executes multiple example queries to inspect:

  * Retrieved complaint text
  * Source metadata
  * Similarity scores

* Runs an automated evaluation suite using predefined questions via `evaluate()`:

  * Collects answers
  * Tracks source grounding
  * Outputs results as a DataFrame
  * Converts evaluation results to Markdown for reporting

**Key Capabilities Demonstrated**

* End-to-end RAG flow (retrieve → generate → cite sources)
* Multi-question querying across different financial products
* Transparent source attribution for generated answers
* Lightweight evaluation for qualitative validation

**Output**

* Console inspection of retrieved documents and answers
* Evaluation results as a Pandas DataFrame
* Markdown-formatted evaluation table for easy reporting

---

## Notes

* All heavy processing and business logic lives in `src/`.
* Notebooks are orchestration and validation layers only.
* Notebooks are safe to rerun and do not modify raw data.
* Figures and artifacts are saved automatically and also displayed inline for inspection.
