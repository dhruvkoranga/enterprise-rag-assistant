from app.embeddings import EmbeddingService
from app.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None
    ):

        query_embedding = (
            self.embedding_service
            .generate_embedding(query)
        )

        results = self.vector_store.search(
            query_embedding,
            limit=top_k
        )

        if min_score is not None:

            results = [
                result
                for result in results
                if result.score >= min_score
                ]

        return results