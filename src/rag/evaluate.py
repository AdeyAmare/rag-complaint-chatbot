import pandas as pd

QUESTIONS = [
    "What issues do customers report with credit card billing?",
    "Are there common problems with money transfer delays?",
    "What complaints are frequent for personal loans?",
    "Do customers report unauthorized transactions?",
    "Are savings account fees a common concern?"
]

def evaluate(pipeline):
    rows = []

    for q in QUESTIONS:
        result = pipeline.answer(q)

        rows.append({
            "Question": q,
            "Generated Answer": result["answer"],
            "Retrieved Source 1": result["sources"][0]["text"][:200],
            "Quality Score (1-5)": "",
            "Comments": ""
        })

    return pd.DataFrame(rows)
