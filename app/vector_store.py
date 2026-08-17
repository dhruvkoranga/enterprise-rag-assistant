from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 2. Create in-memory Qdrant
# --------------------------------------------------

client = QdrantClient(":memory:")

collection_name = "company_handbook"


# --------------------------------------------------
# 3. Create collection
# --------------------------------------------------

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


# --------------------------------------------------
# 4. Our document chunk
# --------------------------------------------------

text = (
    "Employees are entitled to 24 paid vacation days "
    "per calendar year."
)


# --------------------------------------------------
# 5. Generate embedding
# --------------------------------------------------

embedding = model.encode(text)


# --------------------------------------------------
# 6. Store vector in Qdrant
# --------------------------------------------------

client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=1,
            vector=embedding.tolist(),
            payload={
                "text": text,
                "document": "company_handbook.md",
                "section": "Leave Policy"
            }
        )
    ]
)


print("Vector stored successfully.")

# --------------------------------------------------
# 7. Search Qdrant
# --------------------------------------------------

query = "How many annual leave days do employees get?"

query_embedding = model.encode(query)


search_results = client.query_points(
    collection_name=collection_name,
    query=query_embedding.tolist(),
    limit=3
).points


print("\nSearch Results:")

for result in search_results:

    print("\nScore:", result.score)

    print("Text:", result.payload["text"])

    print("Document:", result.payload["document"])

    print("Section:", result.payload["section"])