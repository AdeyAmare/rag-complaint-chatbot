from src.rag.generator import PROMPT_TEMPLATE
from transformers import AutoTokenizer

class ComplaintRAGPipeline:
    """
    CPU-friendly RAG pipeline:
    - Retrieves relevant complaint chunks
    - Combines chunks into a single context
    - Generates a single coherent answer
    """

    def __init__(self, retriever, generator, model_name="google/flan-t5-small"):
        self.retriever = retriever
        self.generator = generator
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        """Safely truncate text to max_tokens using the tokenizer."""
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded.rsplit(" ", 1)[0]

    def answer(self, question: str, k: int = 3, max_total_tokens: int = 512):
        """
        Generate an answer by combining top-k retrieved chunks into one context.

        Args:
            question: User question string
            k: Number of top chunks to retrieve (reduced for small model efficiency)
            max_total_tokens: Max tokens for the combined context

        Returns:
            dict with keys: question, answer, sources
        """
        # Step 1: Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, k)
        
        # Step 2: Combine chunks into a single context block
        # This prevents the 'nonsense' fragmentation caused by iterative summarizing
        per_chunk_tokens = max_total_tokens // k

        chunks = [
            self._truncate_text(r["text"], max_tokens=per_chunk_tokens)
            for r in retrieved
        ]

        context = "\n\n".join(chunks)


        # Step 4: Generate a single answer from the full context
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        final_answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": final_answer,
            "sources": retrieved[:2]  # Keep first 2 sources for reference
        }