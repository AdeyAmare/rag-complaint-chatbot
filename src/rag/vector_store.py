import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import List, Dict, Any, Optional


class ComplaintVectorStore:
    """
    A class to manage a FAISS vector store for complaint embeddings.
    
    This includes building an index from parquet data, persisting it,
    and loading it back for similarity searches.

    Attributes:
        parquet_path (str): Path to the parquet file containing embeddings, documents, and metadata.
        index_dir (Path): Directory where FAISS index and store pickle are saved.
        batch_size (int): Number of rows to process per batch when building the index.
        index (Optional[faiss.Index]): The FAISS index object.
        documents (List[str]): List of documents corresponding to embeddings.
        metadata (List[Dict[str, Any]]): List of metadata corresponding to documents.
    """

    def __init__(
        self,
        parquet_path: str,
        index_dir: str = "vector_store",
        batch_size: int = 50_000
    ) -> None:
        """
        Initialize the ComplaintVectorStore.

        Args:
            parquet_path (str): Path to the parquet file.
            index_dir (str, optional): Directory to save FAISS index and store. Defaults to "vector_store".
            batch_size (int, optional): Number of rows to process per batch. Defaults to 50_000.
        """
        self.parquet_path: str = parquet_path
        self.batch_size: int = batch_size

        self.index_dir: Path = Path(index_dir)
        self.index_path: Path = self.index_dir / "faiss.index"
        self.store_path: Path = self.index_dir / "store.pkl"

        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build + persist index
    # ------------------------------------------------------------------
    def build_and_save(self) -> faiss.Index:
        """
        Build the FAISS index from parquet file and save both index and data.

        Returns:
            faiss.Index: The built FAISS index.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the parquet file is empty or has no embeddings.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if not Path(self.parquet_path).exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        parquet = pd.read_parquet(self.parquet_path, engine="pyarrow")

        if parquet.empty:
            raise ValueError("Parquet file is empty.")

        if "embedding" not in parquet.columns or "document" not in parquet.columns or "metadata" not in parquet.columns:
            raise ValueError("Parquet file must contain 'embedding', 'document', and 'metadata' columns.")

        dim: int = len(parquet.iloc[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)

        for start in tqdm(
            range(0, len(parquet), self.batch_size),
            desc="Building FAISS index"
        ):
            batch = parquet.iloc[start:start + self.batch_size]

            try:
                embeddings: np.ndarray = np.vstack(batch["embedding"].values).astype("float32")
            except Exception as e:
                raise ValueError(f"Error processing embeddings at batch starting index {start}: {e}")

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
    def load(self) -> faiss.Index:
        """
        Load a persisted FAISS index along with documents and metadata.

        Returns:
            faiss.Index: The loaded FAISS index.

        Raises:
            FileNotFoundError: If the index or store files do not exist.
        """
        if not self.index_path.exists() or not self.store_path.exists():
            raise FileNotFoundError("Persisted vector store not found. Build it first using 'build_and_save()'.")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.store_path, "rb") as f:
            store = pickle.load(f)
            self.documents = store["documents"]
            self.metadata = store["metadata"]

        return self.index

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------
    def load_or_build(self) -> faiss.Index:
        """
        Load an existing FAISS index if available, otherwise build and save a new one.

        Returns:
            faiss.Index: The loaded or newly built FAISS index.
        """
        if self.index_path.exists() and self.store_path.exists():
            print("Loading existing FAISS index...")
            return self.load()
        else:
            print("Building FAISS index from scratch...")
            return self.build_and_save()
