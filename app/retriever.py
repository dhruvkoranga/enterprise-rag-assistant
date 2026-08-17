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
        top_k: int = 3
    ):

        query_embedding = (
            self.embedding_service
            .generate_embedding(query)
        )

        results = self.vector_store.search(
            query_embedding,
            limit=top_k
        )

        return results