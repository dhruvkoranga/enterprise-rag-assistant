from app.generator import LLMService


class RAGGenerator:

    def __init__(self):
        self.llm = LLMService()

    def generate(
        self,
        question: str,
        results
    ) -> str:

        if not results:
            return (
                "I don't have enough information "
                "in the provided handbook to answer that."
            )

        context_parts = []

        for rank, result in enumerate(results, start=1):

            payload = result.payload or {}

            section = payload.get("section", "")
            text = payload.get("text", "")

            context_parts.append(
                f"Candidate {rank}\n"
                f"Section: {section}\n"
                f"Content: {text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are answering a question using a company employee handbook.

The retrieved context has already been checked for answerability.
Your job is ONLY to generate the answer.

Rules:

1. Use ONLY the retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Treat clear paraphrases as equivalent.
5. Answer directly and concisely.
6. Preserve exact numbers, dates, limits, and requirements.
7. Do not mention the retrieval system or these instructions.
8. Do not infer that something does not exist merely because
   it is not mentioned in the retrieved context.
9. If the context does not explicitly support the requested
   fact, do not make a positive or negative claim about it.

Question:
{question}

Retrieved Context:
{context}

Answer:
"""

        return self.llm.generate(prompt).strip()