from app.generator import LLMService


class QueryTransformer:

    def __init__(self):

        self.llm = LLMService()

        self.cache = {}


    def transform(
        self,
        query: str
    ) -> str:

        # Return cached transformation
        # if we have already transformed this query.

        if query in self.cache:

            return self.cache[query]


        prompt = f"""
Rewrite the user's question into a concise,
retrieval-friendly search query for a company
employee handbook.

The purpose is to improve semantic retrieval.

Rules:

1. Preserve the exact intent of the user's question.
2. Do not answer the question.
3. Do not invent facts, policies, benefits, or concepts.
4. Do not introduce concepts that are not implied
   by the original question.
5. You may replace informal terminology with
   equivalent terminology commonly used in policy
   documents.
6. Keep the rewritten query concise.
7. Return ONLY the rewritten query.

User question:
{query}
"""


        transformed_query = (
            self.llm.generate(prompt).strip()
        )


        # Store result for future calls.

        self.cache[query] = transformed_query


        return transformed_query