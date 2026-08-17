from chunking import load_document, parse_markdown_sections, create_chunks
from embeddings import EmbeddingService
from vector_store import VectorStore


DOCUMENT_PATH = "data/company_handbook.md"


def ingest_document():

    # ----------------------------------------
    # 1. Load document
    # ----------------------------------------

    document = load_document(
        DOCUMENT_PATH
    )


    # ----------------------------------------
    # 2. Parse sections
    # ----------------------------------------

    sections = parse_markdown_sections(
        document
    )


    # ----------------------------------------
    # 3. Create chunks
    # ----------------------------------------

    chunks = create_chunks(
        sections,
        chunk_size=300,
        overlap_sentences=1
    )


    print(
        f"Created {len(chunks)} chunks."
    )


    # ----------------------------------------
    # 4. Create embedding service
    # ----------------------------------------

    embedding_service = EmbeddingService()


    # ----------------------------------------
    # 5. Generate embeddings
    # ----------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_service
        .generate_embeddings(texts)
    )


    print(
        f"Generated {len(embeddings)} embeddings."
    )


    # ----------------------------------------
    # 6. Create vector store
    # ----------------------------------------

    vector_store = VectorStore()


    # ----------------------------------------
    # 7. Store vectors
    # ----------------------------------------

    vector_store.add_documents(
        embeddings,
        chunks
    )


    print(
        "Documents successfully stored in Qdrant."
    )


    return (
        vector_store,
        embedding_service
    )


if __name__ == "__main__":

    vector_store, embedding_service = (
        ingest_document()
    )