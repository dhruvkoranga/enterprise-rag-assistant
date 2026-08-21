from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer
from app.adaptive_retriever import AdaptiveRetriever
from app.answerability import AnswerabilityDetector
from app.generator import LLMService


POSITIVE_TEST_CASES = [
    {
        "question": "How many vacation days do employees get?",
        "expected_section": "Leave Policy",
    },
    {
        "question": "What is the annual paid time-off allowance?",
        "expected_section": "Leave Policy",
    },
    {
        "question": "How often can employees work remotely?",
        "expected_section": "Remote Work Policy",
    },
    {
        "question": "What is the company's work-from-home allowance?",
        "expected_section": "Remote Work Policy",
    },
    {
        "question": "When do employees become eligible for health insurance?",
        "expected_section": "Health Insurance Policy",
    },
    {
        "question": "How soon after joining does medical coverage begin?",
        "expected_section": "Health Insurance Policy",
    },
    {
        "question": "How long do employees have to submit an expense claim?",
        "expected_section": "Expense Policy",
    },
    {
        "question": "What is the deadline for claiming business expenses?",
        "expected_section": "Expense Policy",
    },
    {
        "question": "What is the approval requirement for vacation longer than five days?",
        "expected_section": "Leave Policy",
    },
    {
        "question": "Who needs to approve an extended vacation?",
        "expected_section": "Leave Policy",
    },
]


NEGATIVE_TEST_CASES = [
    {
        "question": "How many sick leaves can employees take?",
    },
    {
        "question": "Does the company provide dental insurance?",
    },
]


def unwrap_results(retrieval):
    """
    Extract actual Qdrant results from AdaptiveRetriever output.

    Handles:
    1. Normal vector results
    2. RRF wrapped results:
       {
           "result": QdrantResult,
           "rrf_score": float
       }
    """

    wrapped_results = retrieval["results"]

    return [
        item["result"]
        if isinstance(item, dict)
        else item
        for item in wrapped_results
    ]


def get_section(result):
    """
    Extract section name safely.
    """

    return result.payload.get("section", "Unknown")


def build_context(results):
    """
    Combine retrieved document chunks into one context string
    for the AnswerabilityDetector.
    """

    context_parts = []

    for result in results:
        text = result.payload.get("text", "")

        if text:
            context_parts.append(text)

    return "\n\n".join(context_parts)


def main():

    # ==================================================
    # INITIALIZATION
    # ==================================================

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
    
    llm = LLMService()

    answerability_detector = AnswerabilityDetector(
        llm=llm
    )

    top_k = 3

    # ==================================================
    # METRICS
    # ==================================================

    positive_total = len(POSITIVE_TEST_CASES)
    negative_total = len(NEGATIVE_TEST_CASES)

    retrieval_recall_at_1_hits = 0

    answerable_true_positives = 0
    answerable_false_negatives = 0

    negative_true_negatives = 0
    negative_false_positives = 0

    total_transformed = 0

    # ==================================================
    # HEADER
    # ==================================================

    print("\n")
    print("=" * 38)
    print("FULL RAG PIPELINE EVALUATION")
    print("=" * 38)

    # ==================================================
    # POSITIVE QUERIES
    # ==================================================

    print("\n")
    print("POSITIVE QUERIES")
    print("-" * 38)

    for test_case in POSITIVE_TEST_CASES:

        question = test_case["question"]
        expected_section = test_case["expected_section"]

        # ----------------------------------------------
        # STEP 1: ADAPTIVE RETRIEVAL
        # ----------------------------------------------

        retrieval = adaptive_retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        results = unwrap_results(retrieval)

        transformed = retrieval["transformed"]
        initial_margin = retrieval["initial_margin"]

        if transformed:
            total_transformed += 1

        retrieved_sections = [
            get_section(result)
            for result in results
        ]

        # ----------------------------------------------
        # STEP 2: RETRIEVAL RECALL@1
        # ----------------------------------------------

        retrieval_correct = (
            len(retrieved_sections) > 0
            and retrieved_sections[0] == expected_section
        )

        if retrieval_correct:
            retrieval_recall_at_1_hits += 1

        # ----------------------------------------------
        # STEP 3: BUILD CONTEXT
        # ----------------------------------------------

        context = build_context(results)

        # ----------------------------------------------
        # STEP 4: ANSWERABILITY DETECTION
        # ----------------------------------------------

        answerable = answerability_detector.is_answerable(
            question=question,
            results=results,
        )

        # ----------------------------------------------
        # STEP 5: POSITIVE METRICS
        # ----------------------------------------------

        if answerable:
            answerable_true_positives += 1
        else:
            answerable_false_negatives += 1

        # ----------------------------------------------
        # OUTPUT
        # ----------------------------------------------

        print("\nQuestion:")
        print(question)

        print("Expected Section:")
        print(expected_section)

        print("Retrieved:")
        print(retrieved_sections)

        print(f"Transformed: {transformed}")
        print(f"Initial Margin: {initial_margin:.4f}")
        print(f"Answerable: {answerable}")

    # ==================================================
    # NEGATIVE QUERIES
    # ==================================================

    print("\n")
    print("NEGATIVE QUERIES")
    print("-" * 38)

    for test_case in NEGATIVE_TEST_CASES:

        question = test_case["question"]

        # ----------------------------------------------
        # STEP 1: ADAPTIVE RETRIEVAL
        # ----------------------------------------------

        retrieval = adaptive_retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        results = unwrap_results(retrieval)

        transformed = retrieval["transformed"]
        initial_margin = retrieval["initial_margin"]

        if transformed:
            total_transformed += 1

        retrieved_sections = [
            get_section(result)
            for result in results
        ]

        # ----------------------------------------------
        # STEP 2: BUILD CONTEXT
        # ----------------------------------------------

        context = build_context(results)

        # ----------------------------------------------
        # STEP 3: ANSWERABILITY DETECTION
        # ----------------------------------------------

        answerable = answerability_detector.is_answerable(
            question=question,
            results=results,
        )

        # ----------------------------------------------
        # STEP 4: NEGATIVE METRICS
        # ----------------------------------------------

        if not answerable:
            negative_true_negatives += 1
        else:
            negative_false_positives += 1

        # ----------------------------------------------
        # OUTPUT
        # ----------------------------------------------

        print("\nQuestion:")
        print(question)

        print("Retrieved:")
        print(retrieved_sections)

        print(f"Transformed: {transformed}")
        print(f"Initial Margin: {initial_margin:.4f}")
        print(f"Answerable: {answerable}")

    # ==================================================
    # CALCULATE METRICS
    # ==================================================

    retrieval_recall_at_1 = (
        retrieval_recall_at_1_hits / positive_total
        if positive_total > 0
        else 0
    )

    answerable_detection_rate = (
        answerable_true_positives / positive_total
        if positive_total > 0
        else 0
    )

    negative_rejection_rate = (
        negative_true_negatives / negative_total
        if negative_total > 0
        else 0
    )

    false_positive_rate = (
        negative_false_positives / negative_total
        if negative_total > 0
        else 0
    )

    false_negative_rate = (
        answerable_false_negatives / positive_total
        if positive_total > 0
        else 0
    )

    total_queries = positive_total + negative_total

    # ==================================================
    # FINAL METRICS
    # ==================================================

    print("\n")
    print("=" * 38)
    print("FINAL PIPELINE METRICS")
    print("=" * 38)

    print(
        f"Retrieval Recall@1: "
        f"{retrieval_recall_at_1 * 100:.2f}%"
    )

    print(
        f"Answerable Detection Rate: "
        f"{answerable_detection_rate * 100:.2f}%"
    )

    print(
        f"Negative Rejection Rate: "
        f"{negative_rejection_rate * 100:.2f}%"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate * 100:.2f}%"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative_rate * 100:.2f}%"
    )

    print(
        f"Total Queries Transformed: "
        f"{total_transformed}/{total_queries}"
    )


if __name__ == "__main__":
    main()