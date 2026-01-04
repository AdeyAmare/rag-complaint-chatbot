import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from src.rag_prep import EmbeddingProcessor


@pytest.fixture
def sample_csv(tmp_path):
    """Create a tiny cleaned CSV for testing."""
    data = {
        "Complaint ID": [1, 2, 3, 4],
        "Product": [
            "Credit card",
            "Credit card",
            "Personal loan",
            "Personal loan"
        ],
        "Consumer complaint narrative": [
            "My credit card was charged incorrectly.",
            "Late fee applied unfairly.",
            "Loan interest is too high.",
            "Issues with loan repayment."
        ]
    }

    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def processor(sample_csv, tmp_path, monkeypatch):
    """Create processor with mocked embedding model."""
    proc = EmbeddingProcessor(
        cleaned_csv_path=sample_csv,
        vector_store_dir=tmp_path / "vector_store",
        sample_size=4,
        chunk_size=10,
        chunk_overlap=0
    )

    # Mock the SentenceTransformer.encode method
    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(len(texts), 384)
    proc.model = fake_model

    return proc


def test_pipeline_steps_up_to_embedding(processor):
    # Load data
    df = processor.load_data()
    assert len(df) == 4

    # Stratified sample
    sample_df = processor.stratified_sample()
    assert len(sample_df) == 4
    assert set(sample_df["Product"].unique()) == {"Credit card", "Personal loan"}

    # Chunk texts
    chunks, metadata = processor.chunk_texts()
    assert len(chunks) > 0
    assert len(chunks) == len(metadata)

    # Embed chunks
    embeddings = processor.embed_chunks(batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384
