# =============================================================================
# generator.py - Complaint Generator using HuggingFace text2text-generation
# =============================================================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
from typing import Optional

PROMPT_TEMPLATE = """
Context:
{context}
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Based on the complaints above, please answer the following question: {question}
(If the context is truly unrelated to the question, state that you don't have enough info.)

Answer:
"""

class ComplaintGenerator:
    """
    Wrapper for a HuggingFace instruction-tuned LLM.
    Uses text2text-generation pipeline for instruction-following.
    """

    def __init__(
        self,
        model_name: Optional[str] = "google/flan-t5-base",
        max_new_tokens: int = 300,
        temperature: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.llm = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            device=-1  # CPU
        )
        print(f"✓ ComplaintGenerator loaded: {model_name}")

    def generate(self, prompt: str) -> str:
        """
        Generate a response given a full prompt string.
        Args:
            prompt: Full prompt string (context + question)
        Returns:
            Generated answer
        """
        result = self.llm(prompt)[0]["generated_text"]
        answer = result.strip()
        return answer
