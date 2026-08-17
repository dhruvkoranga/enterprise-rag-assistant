from app.rag import RAGPipeline


rag = RAGPipeline()


query = "How many sick leaves am I entitled to?"


answer = rag.ask(
    query,
    top_k=3
)


print("\nQuestion:")
print(query)


print("\nAnswer:")
print(answer)


rag.close()