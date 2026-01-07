import pandas as pd

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

def evaluate(pipeline):
    rows = []

    for q in QUESTIONS:
        result = pipeline.answer(q)
        answer = result["answer"]
        
        # Automated Diagnostic Scoring
        # 1: Refusal (No info)
        # 2: Nonsense/Fragmented
        # 4-5: Coherent & Sourced
        if "do not have enough information" in answer.lower():
            score = 1
            comments = "Model Refusal: Context was likely too fragmented or truncated."
        elif len(answer.split()) < 10:
            score = 2
            comments = "Output too brief or fragmented. Likely model capacity issue."
        else:
            score = 4
            comments = "Coherent answer generated from context."

        rows.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": " | ".join([s["text"][:100] + "..." for s in result["sources"]]),
            "Quality Score (1-5)": score,
            "Comments/Analysis": comments
        })

    return pd.DataFrame(rows)