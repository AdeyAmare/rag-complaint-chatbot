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
    Uses a text2text-generation pipeline for instruction-following.

    Attributes:
        tokenizer: HuggingFace tokenizer for the model.
        model: HuggingFace Seq2Seq model.
        llm: Pipeline object for text2text-generation.
        cache_dir: Optional cache directory for model downloads.
    """

    def __init__(
        self,
        model_name: Optional[str] = "google/flan-t5-base",
        max_new_tokens: int = 300,
        temperature: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ComplaintGenerator.

        Args:
            model_name (Optional[str]): HuggingFace model name. Defaults to "google/flan-t5-base".
            max_new_tokens (int): Maximum tokens to generate. Defaults to 300.
            temperature (float): Sampling temperature. Defaults to 0.1.
            cache_dir (Optional[str]): Optional cache directory for model files.
        
        Raises:
            ValueError: If the model_name is empty or None.
        """
        if not model_name:
            raise ValueError("model_name must be provided.")

        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )

        # Setup text2text-generation pipeline
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
            prompt (str): Full prompt string containing context and question.

        Returns:
            str: Generated answer text.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        result = self.llm(prompt)[0]["generated_text"]
        answer = result.strip()
        return answer
