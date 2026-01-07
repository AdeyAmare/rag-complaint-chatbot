from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any


class ComplaintRetriever:
    """
    A class to retrieve relevant complaints from a vector store using a
    SentenceTransformer embedding model.

    Attributes:
        vector_store: An instance of ComplaintVectorStore containing embeddings, documents, and metadata.
        model: A SentenceTransformer model for embedding queries.
    """

    def __init__(self, vector_store, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the ComplaintRetriever.

        Args:
            vector_store: A ComplaintVectorStore instance with a loaded FAISS index.
            model_name (str, optional): Name of the SentenceTransformer model. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

        Raises:
            ValueError: If the provided vector_store has no index loaded.
        """
        if not hasattr(vector_store, "index") or vector_store.index is None:
            raise ValueError("The provided vector_store must have a loaded FAISS index.")

        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar complaints to a query.

        Args:
            query (str): The input query string.
            k (int, optional): Number of top results to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - "text": The complaint text.
                - "metadata": Associated metadata for the complaint.
                - "score": The similarity score (distance) from the query.

        Raises:
            ValueError: If the query is empty.
        """
        if not query.strip():
            raise ValueError("Query string is empty.")

        # Encode the query to a vector
        query_embedding = self.model.encode([query]).astype("float32")

        # Search the vector store
        distances, indices = self.vector_store.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.vector_store.documents[idx],
                "metadata": self.vector_store.metadata[idx],
                "score": float(distances[0][i])
            })

        return results
