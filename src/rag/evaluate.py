import pandas as pd
from typing import Any

QUESTIONS = [
    "What problems do customers have with credit cards?",
    "Do customers complain about credit card billing?",
    "Are there issues with credit card fees?",

    "Are there delays with money transfers?",
    "Do customers report failed money transfers?",
    "Are international transfers a problem?",

    "What issues do customers report about personal loans?",
    "Do customers complain about loan payments?",
    "Are loan interest rates a common complaint?",

    "Do customers complain about unauthorized transactions?",
    "Are there fraud-related complaints?",
    "Do customers report account security issues?",

    "Are customers unhappy with savings account fees?",
    "Do customers complain about account charges?",
    "Are there problems with savings account access?"
]


def evaluate(pipeline: Any) -> pd.DataFrame:
    """
    Runs a RAG pipeline against a set of representative questions
    and prepares an evaluation table for reporting.

    The function also exports the results to 'rag_evaluation_table.csv'.

    Args:
        pipeline (Any): An object with an 'answer' method that accepts
                        a question string and returns a dictionary with
                        'answer' and 'sources' keys.

    Returns:
        pd.DataFrame: A DataFrame containing questions, generated answers,
                      retrieved sources, and placeholders for qualitative analysis.

    Raises:
        AttributeError: If the pipeline does not have an 'answer' method.
        ValueError: If a question returns an invalid result structure.
    """
    if not hasattr(pipeline, "answer"):
        raise AttributeError("The pipeline object must have an 'answer' method.")

    rows = []
    print(f"Starting evaluation on {len(QUESTIONS)} questions...")

    for q in QUESTIONS:
        result = pipeline.answer(q)
        if not isinstance(result, dict) or "answer" not in result or "sources" not in result:
            raise ValueError(f"Invalid result structure for question: {q}")

        answer = result["answer"]

        # Format sources: Show the first 1-2 sources for scannability
        sources_list = result["sources"][:2]
        formatted_sources = " | ".join(
            [f"Source {i+1}: {s['text'][:150]}..." for i, s in enumerate(sources_list)]
        )

        rows.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": formatted_sources,
            "Quality Score (1-5)": "",  # Left blank for manual qualitative analysis
            "Comments/Analysis": ""     # Left blank for manual qualitative analysis
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Export to CSV
    df.to_csv("rag_evaluation_table.csv", index=False)
    print("✓ Evaluation complete. Results saved to 'rag_evaluation_table.csv'.")

    return df
