# `/src` Overview

This directory contains the **core implementation** for preprocessing CFPB customer complaint data and building a Retrieval-Augmented Generation (RAG) system.
All heavy data processing, embedding, retrieval, generation, and evaluation logic lives here.
Notebooks only orchestrate and validate these components.

---

## 1. `EDA.py` — Exploratory Data Analysis & Preprocessing

**Purpose**
Analyze, clean, and filter CFPB complaint narratives to produce a high-quality dataset suitable for downstream retrieval and modeling tasks.

**Key Responsibilities**

* Batch processing of large Parquet datasets.
* Counts complaints by product and checks narrative availability.
* Cleans complaint text:

  * Removes URLs, boilerplate phrases, numbers, and special characters.
  * Normalizes text via tokenization, stopword removal, and lemmatization.
* Filters complaints to a focused set of products:

  * Credit cards
  * Personal loans
  * Savings accounts
  * Money transfers
* Writes cleaned and filtered complaints to CSV for reuse.
* Generates EDA visualizations:

  * Product distribution
  * Narrative availability
  * Raw vs cleaned word count distributions

**Logging**

* Tracks batch progress, cleaning steps, filtering decisions, and plot generation.
* Designed to be rerunnable and fault-tolerant for large datasets.

---

## 2. `rag_prep.py` — Embedding & Vector Store Preparation

**Purpose**
Transform cleaned complaint narratives into embedding-ready chunks and persist them for retrieval-based workflows.

**Key Responsibilities**

* Loads cleaned complaint data produced by `EDA.py`.
* Performs stratified sampling across products.
* Chunks complaint narratives into retrieval-friendly segments.
* Generates embeddings for each chunk.
* Saves embeddings and metadata in Parquet format.
* Builds and persists a FAISS vector index.
* Produces diagnostic visualizations:

  * Sampled complaints per product
  * Chunks per complaint
  * Chunk length distribution
  * PCA projections of embeddings (overall and by product)

**Logging & Error Handling**

* Tracks each pipeline stage.
* Logs sampling behavior, embedding progress, and index creation.
* Captures and reports failures without corrupting outputs.

---

## 3. `rag/` — Retrieval-Augmented Generation Core

This submodule contains the full RAG implementation used by the notebooks.

### Key Components

* **`vector_store.py`**

  * Implements `ComplaintVectorStore`
  * Loads embeddings from Parquet
  * Builds or loads a FAISS index
  * Manages document text and metadata alignment

* **`retriever.py`**

  * Implements `ComplaintRetriever`
  * Performs semantic search over the FAISS index
  * Returns ranked complaint chunks with similarity scores and metadata

* **`generator.py`**

  * Implements `ComplaintGenerator`
  * Generates natural language answers grounded in retrieved complaints

* **`pipeline.py`**

  * Implements `ComplaintRAGPipeline`
  * Orchestrates retrieval → generation
  * Returns answers with explicit source attribution

* **`evaluate.py`**

  * Runs a predefined evaluation set
  * Collects answers, sources, and metadata
  * Outputs results as a Pandas DataFrame for inspection or reporting

---

## Quick Start (Core API Usage)

### 1. Run EDA & Preprocessing

```python
from EDA import EDAProcessor

eda = EDAProcessor(
    parquet_path="data/complaints.parquet",
    output_csv_path="output/filtered_complaints.csv",
    figures_dir="figures/"
)

results = eda.run()
```

---

### 2. Prepare Embeddings & Vector Store

```python
from rag_prep import EmbeddingProcessor

rag = EmbeddingProcessor(vector_store_dir="vector_store/")
rag.run_pipeline()
```

---

### 3. Build Vector Store, Retrieve, Generate Answers, and Evaluate

This demonstrates the **full RAG workflow**, exactly as implemented in the project.

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.rag.vector_store import ComplaintVectorStore
from src.rag.retriever import ComplaintRetriever
from src.rag.generator import ComplaintGenerator
from src.rag.pipeline import ComplaintRAGPipeline
from src.rag.evaluate import evaluate

# Initialize vector store
vector_store = ComplaintVectorStore(
    parquet_path="../data/complaint_embeddings.parquet"
)

vector_store.load_or_build()
print("Total complaints in index:", vector_store.index.ntotal)

# Initialize retriever
retriever = ComplaintRetriever(vector_store=vector_store)

# Example retrieval
results = retriever.retrieve(
    "What are the common complaints about credit cards?",
    k=5
)

for i, res in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"TEXT: {res['text'][:200]}...")
    print(f"METADATA: {res['metadata']}")
    print(f"SCORE: {res['score']}\n")

# Initialize generator
generator = ComplaintGenerator()

# Connect retriever + generator
rag_pipeline = ComplaintRAGPipeline(
    retriever=retriever,
    generator=generator
)

# Example questions
questions = [
    "What problems do customers have with credit cards?",
    "Are there delays with money transfers?",
    "What issues do customers report about personal loans?",
    "Do customers complain about unauthorized transactions?",
    "Are customers unhappy with savings account fees?"
]

for question in questions:
    result = rag_pipeline.answer(question)
    print("Question:", result["question"])
    print("Answer:\n", result["answer"])
    print("Sample Source Metadata:", result["sources"][0]["metadata"])
    print()
```

---

### 4. Run Evaluation

```python
eval_df = evaluate(rag_pipeline)
print(eval_df)
```

Optional Markdown export:

```python
def df_to_markdown(df):
    md = "| " + " | ".join(df.columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(str(x) for x in row.values) + " |\n"
    return md

print(df_to_markdown(eval_df))
```

---

## Workflow Summary

1. Clean and filter complaint narratives (`EDA.py`).
2. Chunk complaints and generate embeddings (`rag_prep.py`).
3. Build or load FAISS vector store.
4. Retrieve relevant complaint chunks.
5. Generate grounded answers with source attribution.
6. Evaluate RAG performance across predefined questions.

