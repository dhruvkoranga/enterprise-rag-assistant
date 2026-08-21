from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.answerability import AnswerabilityDetector


TEST_CASES = [
    {
        "question": "How many vacation days do employees get?",
        "answerable": True
    },
    {
        "question": "What is the annual paid time-off allowance?",
        "answerable": True
    },
    {
        "question": "How often can employees work remotely?",
        "answerable": True
    },
    {
        "question": "What is the company's work-from-home allowance?",
        "answerable": True
    },
    {
        "question": "When do employees become eligible for health insurance?",
        "answerable": True
    },
    {
        "question": "How soon after joining does medical coverage begin?",
        "answerable": True
    },
    {
        "question": "How long do employees have to submit an expense claim?",
        "answerable": True
    },
    {
        "question": "What is the deadline for claiming business expenses?",
        "answerable": True
    },
    {
        "question": "What is the approval requirement for vacation longer than five days?",
        "answerable": True
    },
    {
        "question": "Who needs to approve an extended vacation?",
        "answerable": True
    },
    {
        "question": "How many sick leaves can employees take?",
        "answerable": False
    },
    {
        "question": "Does the company provide dental insurance?",
        "answerable": False
    }
]


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    detector = AnswerabilityDetector(
        high_confidence_score=0.60
    )


    correct = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0


    print("\n")
    print("======================================")
    print("ANSWERABILITY EVALUATION")
    print("======================================")


    for item in TEST_CASES:

        question = item["question"]

        expected = item["answerable"]


        results = retriever.retrieve(
            question,
            top_k=3
        )


        predicted = detector.is_answerable(
            question,
            results
        )


        top_score = (
            results[0].score
            if results
            else 0
        )


        margin = 0

        if len(results) >= 2:

            margin = (
                results[0].score
                - results[1].score
            )


        if predicted == expected:

            correct += 1


        if expected and predicted:

            true_positive += 1

        elif not expected and not predicted:

            true_negative += 1

        elif not expected and predicted:

            false_positive += 1

        elif expected and not predicted:

            false_negative += 1


        print("\nQuestion:")
        print(question)

        print(
            f"Expected: "
            f"{expected}"
        )

        print(
            f"Predicted: "
            f"{predicted}"
        )

        print(
            f"Top Score: "
            f"{top_score:.4f}"
        )

        print(
            f"Margin: "
            f"{margin:.4f}"
        )

        print(
            f"Result: "
            f"{'PASS' if predicted == expected else 'FAIL'}"
        )


    total = len(TEST_CASES)


    accuracy = correct / total


    positive_count = sum(
        1
        for item in TEST_CASES
        if item["answerable"]
    )

    negative_count = sum(
        1
        for item in TEST_CASES
        if not item["answerable"]
    )


    print("\n")
    print("======================================")
    print("FINAL ANSWERABILITY METRICS")
    print("======================================")


    print(
        f"Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Answerable Detection Rate: "
        f"{true_positive / positive_count:.2%}"
    )

    print(
        f"Negative Rejection Rate: "
        f"{true_negative / negative_count:.2%}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive / negative_count:.2%}"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative / positive_count:.2%}"
    )


    vector_store.close()


if __name__ == "__main__":

    main()