
# `/src` Overview

This folder contains scripts for preprocessing and preparing customer complaint data for AI-powered retrieval tasks.

## 1. `EDA.py` – Exploratory Data Analysis & Preprocessing

* **Purpose:** Analyze and clean CFPB complaint narratives.
* **Key Features:**

  * Batch processing of Parquet files.
  * Counts complaints by product and checks narrative availability.
  * Cleans text (removes URLs, boilerplate, numbers, special characters).
  * Normalizes text (tokenization, stopwords removal, lemmatization).
  * Filters by key products: credit card, personal loan, savings account, money transfer.
  * Saves cleaned complaints to CSV.
  * Generates visualizations for product distribution, narrative availability, and word counts.
* **Logging:** Tracks batch processing, cleaning steps, and plot generation.

## 2. `rag_prep.py` – Retrieval-Augmented Generation Preparation

* **Purpose:** Prepare cleaned complaints for retrieval-based AI tasks.
* **Key Features:**

  * Loads cleaned complaint data.
  * Prepares data for text retrieval or question answering pipelines.
  * (Optional) Connects to vector stores or embedding indexes for RAG workflows.
* **Logging & Error Handling:** Tracks pipeline execution and exceptions.

---

### Quick Start

1. **Run EDA & preprocessing**

```python
from EDA import EDAProcessor

eda = EDAProcessor(
    parquet_path="data/complaints.parquet",
    output_csv_path="output/filtered_complaints.csv",
    figures_dir="figures/"
)
results = eda.run()
```

2. **Prepare RAG data**

```python
from rag_prep import RAGPrep

rag = RAGPrep(vector_store_dir="vector_store/")
rag.load_index()
```

