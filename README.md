# AI-Powered Consumer Complaint Analysis

### Retrieval-Augmented Generation (RAG) Data Pipeline

This project implements an end-to-end pipeline for building **retrieval-augmented AI systems** over CFPB consumer complaint data. It transforms raw regulatory complaint records into a structured, embedded, and indexed corpus suitable for semantic search, question answering, and downstream RAG applications.

The repository cleanly separates **core processing logic**, **execution and analysis**, and **generated artifacts**, making the system easy to inspect, evaluate, and extend.

---

## Repository Structure

```
.
├── src/                     # Core, reusable pipeline logic
│   ├── EDA.py                # EDA and preprocessing
│   ├── rag_prep.py           # Chunking, embedding, vector indexing
│   ├── rag/
│   │   ├── vector_store.py   # FAISS index wrapper
│   │   ├── retriever.py      # Semantic retrieval
│   │   ├── generator.py      # Answer generation
│   │   ├── pipeline.py       # RAG pipeline orchestration
│   │   └── evaluate.py       # Evaluation utilities
│   └── README.md             # Source code documentation
│
├── notebooks/                # Executable notebooks
│   ├── EDA.ipynb
│   ├── rag_prep.ipynb
│   ├── rag_pipeline.ipynb
│   └── README.md
│
├── data/
│   ├── raw/                  # Original CFPB data (Parquet)
│   └── processed/            # Cleaned and filtered CSV
│
├── vector_store/             # FAISS index and metadata (generated)
├── reports/
│   └── figures/              # EDA and embedding diagnostics
├── app.py                    # Interactive Gradio complaint assistant
└── README.md
```

---

## Pipeline Overview

### 1. Exploratory Data Analysis & Preprocessing

Raw CFPB complaints are processed in large batches to:

* Analyze product coverage and narrative availability
* Clean and normalize complaint narratives
* Filter data to a focused set of financial products
* Produce a reusable, model-ready dataset

This step ensures the downstream retrieval system operates on high-quality, consistent text data.

---

### 2. Embedding & Vector Indexing

The cleaned complaint dataset is:

* Stratified sampled by product
* Split into overlapping text segments
* Embedded using a transformer-based sentence embedding model
* Indexed using FAISS for efficient similarity search

All embeddings, metadata, and indexes are saved to disk and reused across experiments.

---

### 3. RAG Pipeline Execution & Evaluation

Once the vector store is built, the full **retrieval-augmented generation workflow** is exercised:

1. **Vector Store**
   Loads or builds a FAISS index from persisted embeddings.

2. **Retriever**
   Performs semantic search to retrieve the most relevant complaint chunks for a given question.

3. **Generator**
   Uses retrieved complaints as context to generate grounded natural-language answers.

4. **RAG Pipeline**
   Orchestrates retrieval and generation into a single question-answering interface with source attribution.

5. **Evaluation**
   Runs a predefined set of representative questions and outputs answers alongside their source metadata for inspection.

This setup enables systematic testing of answer quality, relevance, and grounding.

---

### 4. Interactive Complaint Assistant (Gradio)

The project includes an optional **interactive web application** (`app.py`) built with Gradio:

* Users can ask natural-language questions about consumer complaints.
* The system retrieves relevant complaint narratives and generates answers in real time.
* Generated responses are displayed alongside the most relevant source complaints.
* Designed for local use or lightweight deployment to support demos and stakeholder review.

This interface provides a practical way to explore and validate the RAG pipeline without interacting directly with notebooks or scripts.

---

## Usage Summary

1. **Run EDA & preprocessing**
   Clean and filter raw CFPB complaint data.

2. **Build embeddings and vector store**
   Generate embeddings, metadata, and FAISS indexes.

3. **Test the RAG pipeline**
   Retrieve complaints, generate answers, and evaluate results.

4. **Optional interactive exploration**
   Launch the Gradio app to query the system and inspect answers with sources.

---

## Notes & Design Decisions

* Raw data is treated as immutable.
* All heavy processing logic lives in `src`; notebooks are orchestration and validation layers only.
* Vector store artifacts are excluded from version control due to size.
* Batch processing is used to support large datasets without exhausting memory.
* Evaluation focuses on qualitative correctness and source grounding rather than benchmark scores.
