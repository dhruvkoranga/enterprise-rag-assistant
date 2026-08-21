from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer


def evaluate():

    questions = [
        {
            "question": "How many vacation days do employees get?",
            "expected_section": "Leave Policy"
        },
        {
            "question": "What is the annual paid time-off allowance?",
            "expected_section": "Leave Policy"
        },
        {
            "question": "How often can employees work remotely?",
            "expected_section": "Remote Work Policy"
        },
        {
            "question": "What is the company's work-from-home allowance?",
            "expected_section": "Remote Work Policy"
        },
        {
            "question": "When do employees become eligible for health insurance?",
            "expected_section": "Health Insurance Policy"
        },
        {
            "question": "How soon after joining does medical coverage begin?",
            "expected_section": "Health Insurance Policy"
        },
        {
            "question": "How long do employees have to submit an expense claim?",
            "expected_section": "Expense Policy"
        },
        {
            "question": "What is the deadline for claiming business expenses?",
            "expected_section": "Expense Policy"
        },
        {
            "question": "What is the approval requirement for vacation longer than five days?",
            "expected_section": "Leave Policy"
        },
        {
            "question": "Who needs to approve an extended vacation?",
            "expected_section": "Leave Policy"
        },
        {
            "question": "How many sick leaves can employees take?",
            "expected_section": None
        },
        {
            "question": "Does the company provide dental insurance?",
            "expected_section": None
        }
    ]


    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    transformer = QueryTransformer()


    # ==================================================
    # BASELINE
    # ==================================================

    baseline_reciprocal_rank_sum = 0

    baseline_recall_at_1 = 0

    baseline_recall_at_3 = 0


    print("\n")
    print("======================================")
    print("BASELINE RETRIEVAL")
    print("======================================")


    positive_questions = [
        item
        for item in questions
        if item["expected_section"] is not None
    ]


    for item in positive_questions:

        question = item["question"]

        expected = item["expected_section"]


        results = retriever.retrieve(
            question,
            top_k=3
        )


        sections = [
            result.payload["section"]
            for result in results
        ]


        # Recall@1

        if (
            len(sections) >= 1
            and sections[0] == expected
        ):

            baseline_recall_at_1 += 1


        # Recall@3

        if expected in sections:

            baseline_recall_at_3 += 1


        # MRR

        reciprocal_rank = 0

        for rank, section in enumerate(
            sections,
            start=1
        ):

            if section == expected:

                reciprocal_rank = 1 / rank

                break


        baseline_reciprocal_rank_sum += (
            reciprocal_rank
        )


    positive_count = len(
        positive_questions
    )


    baseline_recall_at_1 /= positive_count

    baseline_recall_at_3 /= positive_count

    baseline_mrr = (
        baseline_reciprocal_rank_sum
        / positive_count
    )


    # ==================================================
    # QUERY TRANSFORMATION
    # ==================================================

    transformed_reciprocal_rank_sum = 0

    transformed_recall_at_1 = 0

    transformed_recall_at_3 = 0


    print("\n")
    print("======================================")
    print("QUERY TRANSFORMATION RETRIEVAL")
    print("======================================")


    for item in positive_questions:

        question = item["question"]

        expected = item["expected_section"]


        transformed_query = (
            transformer.transform(
                question
            )
        )


        results = retriever.retrieve(
            transformed_query,
            top_k=3
        )


        sections = [
            result.payload["section"]
            for result in results
        ]


        print("\nOriginal:")
        print(question)

        print("Transformed:")
        print(transformed_query)

        print("Retrieved:")
        print(sections)


        # Recall@1

        if (
            len(sections) >= 1
            and sections[0] == expected
        ):

            transformed_recall_at_1 += 1


        # Recall@3

        if expected in sections:

            transformed_recall_at_3 += 1


        # MRR

        reciprocal_rank = 0

        for rank, section in enumerate(
            sections,
            start=1
        ):

            if section == expected:

                reciprocal_rank = 1 / rank

                break


        transformed_reciprocal_rank_sum += (
            reciprocal_rank
        )


    transformed_recall_at_1 /= positive_count

    transformed_recall_at_3 /= positive_count

    transformed_mrr = (
        transformed_reciprocal_rank_sum
        / positive_count
    )


    # ==================================================
    # FINAL COMPARISON
    # ==================================================

    print("\n")
    print("======================================")
    print("FINAL COMPARISON")
    print("======================================")


    print("\nMetric                 Baseline     Transformed")

    print(
        f"Recall@1              "
        f"{baseline_recall_at_1:.2%}"
        f"        "
        f"{transformed_recall_at_1:.2%}"
    )

    print(
        f"Recall@3              "
        f"{baseline_recall_at_3:.2%}"
        f"        "
        f"{transformed_recall_at_3:.2%}"
    )

    print(
        f"MRR                   "
        f"{baseline_mrr:.2%}"
        f"        "
        f"{transformed_mrr:.2%}"
    )


    vector_store.close()


if __name__ == "__main__":

    evaluate()