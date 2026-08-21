from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer
from app.adaptive_retriever import AdaptiveRetriever
from app.answerability import AnswerabilityDetector
from app.generator import LLMService
from app.rag_generator import RAGGenerator


def main():

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

    generator = RAGGenerator()


    questions = [
        "How many vacation days do I get?",
        "How often can employees work remotely?",
        "How many sick leaves can employees take?",
        "Does the company provide dental insurance?",
    ]


    for question in questions:

        print("\n")
        print("=" * 60)

        print("Question:")
        print(question)


        retrieval = adaptive_retriever.retrieve(
            query=question,
            top_k=3,
        )


        wrapped_results = retrieval["results"]

        results = [
            item["result"]
            if isinstance(item, dict)
            else item
            for item in wrapped_results
        ]


        print("\nRetrieved:")
        print([
            result.payload["section"]
            for result in results
        ])


        answerable = (
            answerability_detector.is_answerable(
                question=question,
                results=results,
            )
        )


        print(
            f"\nAnswerable: "
            f"{answerable}"
        )


        if not answerable:

            print("\nAnswer:")
            print(
                "I don't have enough information "
                "in the provided handbook to answer that."
            )

            continue


        answer = generator.generate(
            question,
            results
        )


        print("\nAnswer:")
        print(answer)


    vector_store.close()


if __name__ == "__main__":
    main()