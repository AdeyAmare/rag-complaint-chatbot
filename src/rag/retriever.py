from sentence_transformers import SentenceTransformer
import numpy as np

class ComplaintRetriever:
    def __init__(self, vector_store, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, k: int = 5):
        query_embedding = self.model.encode([query]).astype("float32")

        distances, indices = self.vector_store.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.vector_store.documents[idx],
                "metadata": self.vector_store.metadata[idx],
                "score": float(distances[0][i])
            })


        return results
