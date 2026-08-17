from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.generator import LLMService
from app.prompt import build_rag_prompt


class RAGPipeline:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

        self.vector_store = (
            VectorStore()
        )

        self.retriever = Retriever(
            self.embedding_service,
            self.vector_store
        )

        self.llm = LLMService()

    def ask(
        self,
        query: str,
        top_k: int = 3
    ):

        # ----------------------------------
        # 1. Retrieve relevant chunks
        # ----------------------------------

        results = self.retriever.retrieve(
            query,
            top_k=top_k
        )


        # ----------------------------------
        # 2. Build prompt
        # ----------------------------------

        prompt = build_rag_prompt(
            query,
            results
        )


        # ----------------------------------
        # 3. Generate answer
        # ----------------------------------

        answer = self.llm.generate(
            prompt
        )


        return answer

    def close(self):

        self.vector_store.close()