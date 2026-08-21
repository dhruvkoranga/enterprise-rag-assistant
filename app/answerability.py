from app.answerability_judge import AnswerabilityJudge


class AnswerabilityDetector:

    def __init__(
        self,
        llm,
        high_confidence_score=0.6,
    ):
        self.high_confidence_score = high_confidence_score
        self.judge = AnswerabilityJudge(llm)

    def is_answerable(
        self,
        question: str,
        results,
    ) -> bool:

        if not results:
            return False

        top_score = results[0].score

        # High-confidence retrieval:
        # skip the LLM judge
        if top_score >= self.high_confidence_score:
            return True

        # Lower-confidence retrieval:
        # ask the LLM whether the context answers the question
        return self.judge.is_answerable(
            question,
            results,
        )