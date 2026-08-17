from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever


embedding_service = EmbeddingService()

vector_store = VectorStore()

retriever = Retriever(
    embedding_service,
    vector_store
)


query = "What is my annual leave entitlement?"


results = retriever.retrieve(
    query,
    top_k=3
)


print("\nQuery:")
print(query)


print("\nRetrieved Results:")

for result in results:

    print("\n-------------------------")

    print("Score:", result.score)

    print("Text:", result.payload["text"])

    print(
        "Document:",
        result.payload["document"]
    )

    print(
        "Section:",
        result.payload["section"]
    )


vector_store.close()