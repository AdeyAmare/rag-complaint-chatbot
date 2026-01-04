# Notebooks

This directory contains Jupyter notebooks that **run and validate the data pipelines** implemented in `src/EDA.py` and `src/rag_prep.py`.
They are responsible for execution, inspection, and visualization — not core logic.

---

## Task 1 Notebook – EDA & Preprocessing

* Executes `EDAProcessor` on the raw CFPB complaints dataset.
* Processes the data in large batches.
* Writes the cleaned, filtered complaints to CSV.
* Generates and saves EDA figures:

  * Product distribution
  * Narrative availability
  * Raw vs cleaned narrative length distributions

**Output**

* `data/processed/filtered_complaints.csv`
* Figures saved to `reports/figures/`

---

## Task 2 Notebook – Embedding & Vector Store Prep

* Executes `EmbeddingProcessor` from `rag_prep.py`.
* Performs stratified sampling on the cleaned dataset.
* Chunks complaint narratives and generates embeddings.
* Builds and saves a FAISS vector index and metadata.
* Produces diagnostic visualizations:

  * Sampled complaints per product
  * Chunks per complaint
  * Chunk length distribution
  * PCA projections of embeddings (overall and by product)

**Output**

* FAISS index and metadata in `vector_store/`
* Figures saved to `reports/figures/`

---

## Notes

* All heavy processing logic lives in `src/`.
* Notebooks are safe to rerun and do not modify raw data.
* Figures are saved automatically; plots are also displayed inline for inspection.

