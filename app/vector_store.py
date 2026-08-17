from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:

    def __init__(
        self,
        collection_name: str = "company_handbook"
    ):

        self.collection_name = collection_name

        self.client = QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    def add_documents(
        self,
        embeddings,
        chunks: list[dict]
    ):

        points = []

        for index, (embedding, chunk) in enumerate(
            zip(embeddings, chunks)
        ):

            points.append(
                PointStruct(
                    id=index,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_embedding,
        limit: int = 3
    ):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit
        ).points

        return results