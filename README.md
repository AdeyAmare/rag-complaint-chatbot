# AI-Powered Consumer Complaint Analysis

### Retrieval-Augmented Generation (RAG) Data Pipeline

This project implements an end-to-end data preparation pipeline for building retrieval-augmented AI systems over CFPB consumer complaint data. It transforms raw regulatory complaint records into a structured, embedded, and indexed corpus suitable for semantic search and downstream RAG applications.

The repository cleanly separates **core processing logic**, **execution and analysis**, and **generated artifacts**.

---

## Repository Structure

```
.
├── src/                    # Core, reusable pipeline logic
│   ├── EDA.py               # EDA + preprocessing
│   ├── rag_prep.py          # Chunking, embedding, vector indexing
│   └── README.md            # Source code documentation
│
├── notebooks/               # Executable notebooks
│   ├── EDA.ipynb
│   ├── rag_prep.ipynb
│   └── README.md            # Notebook documentation
│
├── data/
│   ├── raw/                 # Original CFPB data (Parquet)
│   └── processed/           # Cleaned and filtered CSV
│
├── vector_store/            # FAISS index + metadata (generated)
├── reports/
│   └── figures/             # EDA & embedding diagnostics
└── README.md
```

---

## Pipeline Overview

### 1. Exploratory Data Analysis & Preprocessing

Raw CFPB complaints are processed in large batches to:

* Analyze product coverage and narrative availability
* Clean and normalize complaint narratives
* Filter data to relevant financial products
* Produce a reusable, model-ready dataset

### 2. Embedding & Vector Indexing

The cleaned dataset is:

* Stratified sampled by product
* Chunked into overlapping text segments
* Embedded using a sentence-transformer model
* Indexed using FAISS for fast similarity search

### 3. RAG-Ready Outputs (Artifacts)

The final artifacts enable:

* Semantic retrieval
* Context injection for LLMs
* Exploratory analysis of embedding space

---

## Usage

### Prerequisites

* Python 3.9+
* Sufficient memory for batch processing and embedding
* CFPB complaints dataset in Parquet format

Install dependencies (example):

```bash
pip install -r requirements.txt
```

---

### Step 1: Run EDA & Preprocessing (Task 1)

Run the **Task 1 notebook** located in `notebooks/`.

This notebook executes the `EDAProcessor` from `src/EDA.py`.

**What happens**

* Loads raw complaint data from Parquet in large batches
* Computes dataset-level statistics (products, narratives, lengths)
* Cleans complaint narratives using NLP preprocessing
* Filters complaints to required product categories
* Saves the cleaned dataset
* Generates and saves EDA visualizations

**Inputs**

* `data/raw/complaints.parquet`

**Outputs**

* `data/processed/filtered_complaints.csv`
* Figures saved to `reports/figures/`

---

### Step 2: Build Embeddings & Vector Store (Task 2)

Run the **Task 2 notebook** in `notebooks/`.

This notebook executes `EmbeddingProcessor` from `src/rag_prep.py`.

**What happens**

* Loads the cleaned complaints CSV
* Performs stratified sampling by product
* Splits complaint narratives into overlapping chunks
* Generates dense embeddings using a transformer model
* Builds a FAISS vector index
* Saves index and metadata to disk
* Generates diagnostic and embedding visualizations

**Inputs**

* `data/processed/filtered_complaints.csv`

**Outputs**

* `vector_store/`

  * FAISS index (`.idx`)
  * Chunk metadata (`.pkl`)
* Figures saved to `reports/figures/`, including:

  * Sample distribution by product
  * Chunk count per complaint
  * Chunk length distribution
  * PCA projections of embedding space

---

### Step 3: Downstream RAG or Retrieval Use

The contents of `vector_store/` can be loaded by any FAISS-compatible system to support:

* Semantic complaint search
* Context retrieval for LLM prompts
* Topic or product-specific complaint analysis

No additional preprocessing is required.

---

## Documentation

* **Core pipeline logic:**
  👉 [`src/README.md`](src/README.md)

* **Notebook execution & figures:**
  👉 [`notebooks/README.md`](notebooks/README.md)

---

## Notes & Design Decisions

* Raw data is treated as immutable.
* All heavy processing logic lives in `src`; notebooks only orchestrate execution and visualization.
* The vector store is excluded from version control due to size.
* Batch processing is used to support large-scale datasets without exhausting memory.

