import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm

class ComplaintVectorStore:
    def __init__(self, parquet_path: str, batch_size: int = 50_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.index = None
        self.metadata = []
        self.documents = []

    def build_faiss_index(self):
        parquet = pd.read_parquet(self.parquet_path, engine="pyarrow")

        dim = len(parquet.iloc[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)

        for start in tqdm(range(0, len(parquet), self.batch_size), desc="Building FAISS index"):
            batch = parquet.iloc[start:start+self.batch_size]

            embeddings = np.vstack(batch["embedding"].values).astype("float32")
            self.index.add(embeddings)

            self.documents.extend(batch["document"].tolist())
            self.metadata.extend(batch["metadata"].tolist())

        return self.index
