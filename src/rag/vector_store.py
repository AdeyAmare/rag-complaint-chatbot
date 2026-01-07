import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle

class ComplaintVectorStore:
    def __init__(
        self,
        parquet_path: str,
        index_dir: str = "vector_store",
        batch_size: int = 50_000
    ):
        self.parquet_path = parquet_path
        self.batch_size = batch_size

        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.store_path = self.index_dir / "store.pkl"

        self.index = None
        self.documents = []
        self.metadata = []

    # ------------------------------------------------------------------
    # Build + persist index
    # ------------------------------------------------------------------
    def build_and_save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        parquet = pd.read_parquet(self.parquet_path, engine="pyarrow")

        dim = len(parquet.iloc[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)

        for start in tqdm(
            range(0, len(parquet), self.batch_size),
            desc="Building FAISS index"
        ):
            batch = parquet.iloc[start:start + self.batch_size]

            embeddings = np.vstack(batch["embedding"].values).astype("float32")
            self.index.add(embeddings)

            self.documents.extend(batch["document"].tolist())
            self.metadata.extend(batch["metadata"].tolist())

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save documents + metadata
        with open(self.store_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                },
                f
            )

        return self.index

    # ------------------------------------------------------------------
    # Load persisted index
    # ------------------------------------------------------------------
    def load(self):
        if not self.index_path.exists() or not self.store_path.exists():
            raise FileNotFoundError("Persisted vector store not found")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.store_path, "rb") as f:
            store = pickle.load(f)
            self.documents = store["documents"]
            self.metadata = store["metadata"]

        return self.index

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------
    def load_or_build(self):
        if self.index_path.exists():
            print("Loading existing FAISS index...")
            return self.load()
        else:
            print("Building FAISS index from scratch...")
            return self.build_and_save()
