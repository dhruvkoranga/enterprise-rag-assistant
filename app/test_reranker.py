from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.reranker import Reranker


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service,
        vector_store
    )

    reranker = Reranker()


    queries = [
        {
            "query": "How often can employees work remotely?",
            "expected": "Remote Work Policy"
        },
        {
            "query": "What is the company's work-from-home allowance?",
            "expected": "Remote Work Policy"
        },
        {
            "query": "Does the company provide dental insurance?",
            "expected": None
        }
    ]


    for item in queries:

        query = item["query"]

        expected = item["expected"]


        print("\n")
        print("====================================")
        print("Query:")
        print(query)

        print("\nExpected:")
        print(expected)


        # -------------------------------
        # Vector retrieval
        # -------------------------------

        results = retriever.retrieve(
            query,
            top_k=3
        )


        print("\nOriginal Vector Ranking")
        print("-----------------------")


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result.payload['section']} "
                f"({result.score:.4f})"
            )


        # -------------------------------
        # Reranking
        # -------------------------------

        reranked_results = reranker.rerank(
            query,
            results,
            top_k=3
        )


        print("\nReranked")
        print("--------")


        for rank, result in enumerate(
            reranked_results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result.payload['section']} "
                f"({result.score:.4f})"
            )


    vector_store.close()


if __name__ == "__main__":

    main()