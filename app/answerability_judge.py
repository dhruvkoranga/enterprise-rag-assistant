import re


class AnswerabilityJudge:

    def __init__(self, llm):
        self.llm = llm

    def is_answerable(self, question, results):

        if not results:
            return False

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
You are a strict answerability classifier for a company employee handbook.

Your ONLY job is to decide whether the retrieved candidates contain
enough explicit information to answer the user's question.

Return exactly:
YES
or
NO

Rules:

1. Return YES only when at least one candidate explicitly supports
   the requested answer.

2. Clear paraphrases are allowed.

3. Do NOT infer missing information.

4. If the question asks about a specific benefit, policy, allowance,
   number, limit, date, or approval requirement, that specific
   information must be supported by the context.

5. A broad category does NOT imply a specific category.

   Example:
   "health insurance" does NOT imply "dental insurance".

6. Absence of information means NO.

7. Do NOT interpret "not mentioned" as proof that something does not
   exist. NO means only that the handbook does not provide enough
   information to answer the question.

Examples:

Question:
How many vacation days do employees get?

Context:
Employees are entitled to 24 paid vacation days per calendar year.

Answer:
YES


Question:
How often can employees work remotely?

Context:
Employees may work remotely up to three days per week.

Answer:
YES


Question:
Who needs to approve an extended vacation?

Context:
Vacation requests longer than five consecutive working days
require manager approval.

Answer:
YES


Question:
How many sick leaves can employees take?

Context:
Employees are entitled to 24 paid vacation days per calendar year.

Answer:
NO


Question:
Does the company provide dental insurance?

Context:
All full-time employees are eligible for company health insurance
after completing 30 days of employment.

Answer:
NO

Now classify this case.

Question:
{question}

Retrieved Candidates:
{context}

Return only YES or NO.
"""

        response = self.llm.generate(prompt)

        match = re.search(
            r"\b(YES|NO)\b",
            response.upper()
        )

        if match:
            return match.group(1) == "YES"

        return False