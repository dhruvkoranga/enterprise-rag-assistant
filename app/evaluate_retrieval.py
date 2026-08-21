import json

from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever


def load_evaluation_questions():

    with open(
        "data/evaluation_questions.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_retrieval(
    top_k: int = 3,
    min_score: float | None = None
):

    questions = load_evaluation_questions()

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    positive_questions = [
        item
        for item in questions
        if item["expected_section"] is not None
    ]

    negative_questions = [
        item
        for item in questions
        if item["expected_section"] is None
    ]


    # ==========================================
    # POSITIVE QUERY EVALUATION
    # ==========================================

    positive_recall_at_k = 0

    reciprocal_rank_sum = 0


    print("\n")
    print("======================================")
    print(f"Positive Query Evaluation - Top {top_k}")
    print("======================================")


    for item in positive_questions:

        question = item["question"]

        expected_section = item["expected_section"]


        results = retriever.retrieve(
            question,
            top_k=top_k,
            min_score=min_score
        )


        retrieved_sections = [
            result.payload["section"]
            for result in results
        ]


        retrieved_scores = [
            result.score
            for result in results
        ]
        if len(retrieved_scores) >= 2:
            margin = retrieved_scores[0] - retrieved_scores[1]
        else:
            margin = None


        # --------------------------------------
        # Recall@K
        # --------------------------------------

        if expected_section in retrieved_sections:

            positive_recall_at_k += 1


        # --------------------------------------
        # MRR
        # --------------------------------------

        reciprocal_rank = 0

        for rank, section in enumerate(
            retrieved_sections,
            start=1
        ):

            if section == expected_section:

                reciprocal_rank = 1 / rank

                break


        reciprocal_rank_sum += reciprocal_rank


        print("\nQuestion:")
        print(question)

        print("Expected:")
        print(expected_section)

        print("Retrieved:")
        print(retrieved_sections)

        print("Scores:")
        print(retrieved_scores)

        print("Margin:")
        print(margin)

        print(
            "Result:",
            "PASS"
            if expected_section in retrieved_sections
            else "FAIL"
        )


    positive_count = len(
        positive_questions
    )


    recall_at_k = (
        positive_recall_at_k
        / positive_count
        if positive_count
        else 0
    )


    mrr = (
        reciprocal_rank_sum
        / positive_count
        if positive_count
        else 0
    )


    # ==========================================
    # NEGATIVE QUERY EVALUATION
    # ==========================================

    rejected_correctly = 0


    print("\n")
    print("======================================")
    print("Negative Query Evaluation")
    print("======================================")


    for item in negative_questions:

        question = item["question"]


        results = retriever.retrieve(
            question,
            top_k=top_k,
            min_score=min_score
        )


        retrieved_sections = [
            result.payload["section"]
            for result in results
        ]


        retrieved_scores = [
            result.score
            for result in results
        ]
        if len(retrieved_scores) >= 2:
            margin = retrieved_scores[0] - retrieved_scores[1]
        else:
            margin = None


        print("\nQuestion:")
        print(question)

        print("Retrieved:")
        print(retrieved_sections)

        print("Scores:")
        print(retrieved_scores)

        print("Margin:")
        print(margin)


        # --------------------------------------
        # Currently our retriever ALWAYS returns
        # results, so this will initially be False.
        # --------------------------------------

        if len(results) == 0:

            rejected_correctly += 1

            print("Result: PASS")

        else:

            print("Result: FAIL")


    negative_count = len(
        negative_questions
    )


    rejection_rate = (
        rejected_correctly
        / negative_count
        if negative_count
        else 0
    )


    # ==========================================
    # FINAL METRICS
    # ==========================================

    print("\n")
    print("======================================")
    print("FINAL RETRIEVAL METRICS")
    print("======================================")

    print(
        f"Recall@{top_k}: {recall_at_k:.2%}"
    )

    print(
        f"MRR: {mrr:.2%}"
    )

    print(
        f"Negative Rejection Rate: "
        f"{rejection_rate:.2%}"
    )


    vector_store.close()


if __name__ == "__main__":

    print("\n")
    print("******** TOP-K = 1 ********")

    evaluate_retrieval(
        top_k=1
    )


    print("\n")
    print("******** TOP-K = 3 ********")

    evaluate_retrieval(
        top_k=3
    )

if __name__ == "__main__":

    print("\n******** TOP-K = 3, THRESHOLD = 0.40 ********")

    evaluate_retrieval(
        top_k=3,
        min_score=0.50
    )