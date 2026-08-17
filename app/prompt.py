def build_rag_prompt(
    query: str,
    retrieved_chunks: list
) -> str:

    context_parts = []

    for index, result in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
Context {index}:

Section: {result.payload["section"]}

Document: {result.payload["document"]}

Content:
{result.payload["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an AI assistant answering questions
about a company employee handbook.

Use ONLY the information provided in the
context below to answer the user's question.

Do not use outside knowledge.

If the answer cannot be found in the provided
context, say:

"I don't have enough information in the
provided handbook to answer that."

Keep the answer concise and factual.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{query}

ANSWER:
"""

    return prompt