from src.rag.generator import PROMPT_TEMPLATE
from transformers import AutoTokenizer

class ComplaintRAGPipeline:
    def __init__(self, retriever, generator, model_name="google/flan-t5-base"):
        self.retriever = retriever
        self.generator = generator
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        # Proper truncation that doesn't split words
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
        return self.tokenizer.decode(tokens)

    def answer(self, question: str, k: int = 3, max_input_limit: int = 512):
        # 1. Retrieve raw chunks
        retrieved = self.retriever.retrieve(question, k)
        if not retrieved:
            return {"question": question, "answer": "No relevant info found.", "sources": []}

        # 2. Measure overhead
        empty_prompt = PROMPT_TEMPLATE.format(context="", question=question)
        prompt_overhead = len(self.tokenizer.encode(empty_prompt))
        
        # 3. Space for context (leaving a small buffer)
        available_tokens = max_input_limit - prompt_overhead - 10 

        # 4. GLOBAL TRUNCATION (Stops fragmentation)
        # We join all text first, then truncate once.
        full_context_text = "\n\n".join([r["text"] for r in retrieved])
        context = self._truncate_text(full_context_text, max_tokens=available_tokens)

        # 5. Generate with CrediTrust Persona
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        final_answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": final_answer,
            "sources": retrieved
        }