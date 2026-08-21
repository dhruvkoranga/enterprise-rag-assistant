from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer
from app.adaptive_retriever import AdaptiveRetriever


def main():
    # Initialize embedding service
    embedding_service = EmbeddingService()

    # Initialize vector store
    vector_store = VectorStore()

    # Initialize retriever with required dependencies
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Initialize query transformer
    query_transformer = QueryTransformer()

    # Initialize adaptive retriever
    adaptive_retriever = AdaptiveRetriever(
        retriever=retriever,
        query_transformer=query_transformer,
        margin_threshold=0.05,
        rrf_k=60,
        original_weight=1.0,
        transformed_weight=1.5,
    )

    # Test queries
    queries = [
        "How often can employees work remotely?",
        "What is the company's work-from-home allowance?",
        "How many vacation days do I get?",
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("Query:")
        print(query)

        # Run adaptive retrieval
        retrieval = adaptive_retriever.retrieve(
            query=query,
            top_k=3,
        )

        # Extract values from dictionary
        results = retrieval["results"]
        transformed = retrieval["transformed"]
        initial_margin = retrieval["initial_margin"]
        transformed_query = retrieval["transformed_query"]

        # Print metadata
        print(f"\nTransformed: {transformed}")
        print(f"Initial Margin: {initial_margin:.4f}")

        if transformed:
            print(f"Transformed Query: {transformed_query}")

        # Print final ranking
        print("\nFinal Results:")

        for index, item in enumerate(results, start=1):

            result = item["result"]
            rrf_score = item["rrf_score"]

            section = result.payload.get(
            "section",
            "Unknown Section",
            )

            if rrf_score is not None:

                print(
                    f"{index}. {section} "
                    f"(RRF Score: {rrf_score:.6f})"
                )

            else:

                print(
                    f"{index}. {section} "
                    f"(Vector Score: {result.score:.4f})"
                )


if __name__ == "__main__":
    main()