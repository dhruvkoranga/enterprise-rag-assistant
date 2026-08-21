from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer
from app.adaptive_retriever import AdaptiveRetriever


POSITIVE_QUERIES = [
    {
        "question": "How many vacation days do employees get?",
        "expected": "Leave Policy",
    },
    {
        "question": "What is the annual paid time-off allowance?",
        "expected": "Leave Policy",
    },
    {
        "question": "How often can employees work remotely?",
        "expected": "Remote Work Policy",
    },
    {
        "question": "What is the company's work-from-home allowance?",
        "expected": "Remote Work Policy",
    },
    {
        "question": "When do employees become eligible for health insurance?",
        "expected": "Health Insurance Policy",
    },
    {
        "question": "How soon after joining does medical coverage begin?",
        "expected": "Health Insurance Policy",
    },
    {
        "question": "How long do employees have to submit an expense claim?",
        "expected": "Expense Policy",
    },
    {
        "question": "What is the deadline for claiming business expenses?",
        "expected": "Expense Policy",
    },
    {
        "question": "What is the approval requirement for vacation longer than five days?",
        "expected": "Leave Policy",
    },
    {
        "question": "Who needs to approve an extended vacation?",
        "expected": "Leave Policy",
    },
]


NEGATIVE_QUERIES = [
    "How many sick leaves can employees take?",
    "Does the company provide dental insurance?",
]


def get_section_name(result):
    """
    Extract the section name from a retrieval result.

    Supports both:
    1. Normal retrieval result objects
    2. RRF fused dictionaries containing a 'result' key
    """

    if isinstance(result, dict):
        result = result["result"]

    return result.payload.get("section")


def main():

    # --------------------------------------------------
    # Initialize services
    # --------------------------------------------------

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    query_transformer = QueryTransformer()

    adaptive_retriever = AdaptiveRetriever(
        retriever=retriever,
        query_transformer=query_transformer,
        margin_threshold=0.05,
        rrf_k=60,
        original_weight=1.0,
        transformed_weight=1.5,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    recall_at_1_hits = 0
    recall_at_3_hits = 0
    reciprocal_rank_sum = 0

    positive_transformed_count = 0
    negative_transformed_count = 0

    # --------------------------------------------------
    # Positive Query Evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 38)
    print("POSITIVE QUERY EVALUATION")
    print("=" * 38)

    for test_case in POSITIVE_QUERIES:

        question = test_case["question"]
        expected = test_case["expected"]

        retrieval = adaptive_retriever.retrieve(
            query=question,
            top_k=3,
        )

        # AdaptiveRetriever now returns a dictionary
        wrapped_results = retrieval["results"]

        transformed = retrieval["transformed"]
        initial_margin = retrieval["initial_margin"]

        if transformed:
            positive_transformed_count += 1

        # Extract actual retrieval objects.
        # Handles both normal vector results and RRF results.
        results = [
            item["result"] if isinstance(item, dict) else item
            for item in wrapped_results
        ]

        retrieved_sections = [
            get_section_name(result)
            for result in results
        ]

        # -----------------------------
        # Recall@1
        # -----------------------------

        if len(retrieved_sections) > 0:
            if retrieved_sections[0] == expected:
                recall_at_1_hits += 1

        # -----------------------------
        # Recall@3
        # -----------------------------

        if expected in retrieved_sections:
            recall_at_3_hits += 1

        # -----------------------------
        # MRR
        # -----------------------------

        for rank, section in enumerate(
            retrieved_sections,
            start=1,
        ):
            if section == expected:
                reciprocal_rank_sum += 1 / rank
                break

        # -----------------------------
        # Result
        # -----------------------------

        passed = (
            len(retrieved_sections) > 0
            and retrieved_sections[0] == expected
        )

        print("\nQuestion:")
        print(question)

        print("Expected:")
        print(expected)

        print("Retrieved:")
        print(retrieved_sections)

        print(f"Transformed: {transformed}")
        print(f"Initial Margin: {initial_margin:.4f}")

        print(
            "Result:",
            "PASS" if passed else "FAIL",
        )

    # --------------------------------------------------
    # Negative Query Evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 38)
    print("NEGATIVE QUERY EVALUATION")
    print("=" * 38)

    negative_rejected = 0

    for question in NEGATIVE_QUERIES:

        retrieval = adaptive_retriever.retrieve(
            query=question,
            top_k=3,
        )

        wrapped_results = retrieval["results"]

        transformed = retrieval["transformed"]
        initial_margin = retrieval["initial_margin"]

        if transformed:
            negative_transformed_count += 1

        results = [
            item["result"] if isinstance(item, dict) else item
            for item in wrapped_results
        ]

        retrieved_sections = [
            get_section_name(result)
            for result in results
        ]

        # Vector retrieval always returns something.
        # Therefore, retrieval alone cannot truly reject
        # unsupported queries.
        #
        # This is intentionally marked as FAIL here because
        # negative rejection is handled by the Answerability
        # Detector in the full pipeline.

        rejected = False

        if rejected:
            negative_rejected += 1

        print("\nQuestion:")
        print(question)

        print("Retrieved:")
        print(retrieved_sections)

        print(f"Transformed: {transformed}")
        print(f"Initial Margin: {initial_margin:.4f}")

        print(
            "Result:",
            "PASS" if rejected else "FAIL",
        )

    # --------------------------------------------------
    # Final Metrics
    # --------------------------------------------------

    total_positive = len(POSITIVE_QUERIES)
    total_negative = len(NEGATIVE_QUERIES)
    total_queries = total_positive + total_negative

    recall_at_1 = (
        recall_at_1_hits / total_positive
        if total_positive > 0
        else 0
    )

    recall_at_3 = (
        recall_at_3_hits / total_positive
        if total_positive > 0
        else 0
    )

    mrr = (
        reciprocal_rank_sum / total_positive
        if total_positive > 0
        else 0
    )

    negative_rejection_rate = (
        negative_rejected / total_negative
        if total_negative > 0
        else 0
    )

    total_transformed = (
        positive_transformed_count
        + negative_transformed_count
    )

    print("\n")
    print("=" * 38)
    print("FINAL ADAPTIVE RETRIEVAL METRICS")
    print("=" * 38)

    print(f"Recall@1: {recall_at_1 * 100:.2f}%")
    print(f"Recall@3: {recall_at_3 * 100:.2f}%")
    print(f"MRR: {mrr * 100:.2f}%")
    print(
        f"Negative Rejection Rate: "
        f"{negative_rejection_rate * 100:.2f}%"
    )
    print(
        f"Positive Queries Transformed: "
        f"{positive_transformed_count}/{total_positive}"
    )
    print(
        f"Negative Queries Transformed: "
        f"{negative_transformed_count}/{total_negative}"
    )
    print(
        f"Total Queries Transformed: "
        f"{total_transformed}/{total_queries}"
    )


if __name__ == "__main__":
    main()