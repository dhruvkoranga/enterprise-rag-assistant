from app.embeddings import EmbeddingService
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.query_transformer import QueryTransformer
from app.adaptive_retriever import AdaptiveRetriever
from app.answerability import AnswerabilityDetector
from app.generator import LLMService
from app.prompt import build_rag_prompt


ABSTAIN_MESSAGE = (
    "I don't have enough information in the provided handbook "
    "to answer that."
)


class RAGPipeline:

    def __init__(self):

        # ----------------------------------
        # 1. Embedding + Vector Store
        # ----------------------------------

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore()

        # ----------------------------------
        # 2. Base Retriever
        # ----------------------------------

        self.retriever = Retriever(
            self.embedding_service,
            self.vector_store
        )

        # ----------------------------------
        # 3. Query Transformer
        # ----------------------------------

        self.query_transformer = QueryTransformer()

        # ----------------------------------
        # 4. Adaptive Retriever
        # ----------------------------------

        self.adaptive_retriever = AdaptiveRetriever(
            retriever=self.retriever,
            query_transformer=self.query_transformer,
            margin_threshold=0.05,
            rrf_k=60,
            original_weight=1.0,
            transformed_weight=1.5,
        )

        # ----------------------------------
        # 5. Local LLM
        # ----------------------------------

        self.llm = LLMService()

        # ----------------------------------
        # 6. Answerability Detector
        # ----------------------------------

        self.answerability_detector = (
            AnswerabilityDetector(
                llm=self.llm
            )
        )


    def _unwrap_results(self, retrieval):

        wrapped_results = retrieval["results"]

        return [
            item["result"]
            if isinstance(item, dict)
            else item
            for item in wrapped_results
        ]


    def ask(
        self,
        query: str,
        top_k: int = 3
    ):

        # ----------------------------------
        # 1. Adaptive retrieval
        # ----------------------------------

        retrieval = self.adaptive_retriever.retrieve(
            query,
            top_k=top_k
        )

        results = self._unwrap_results(
            retrieval
        )


        # ----------------------------------
        # 2. Answerability check
        # ----------------------------------

        answerable = (
            self.answerability_detector.is_answerable(
                question=query,
                results=results,
            )
        )


        # ----------------------------------
        # 3. Abstain if unsupported
        # ----------------------------------

        if not answerable:
            return ABSTAIN_MESSAGE


        # ----------------------------------
        # 4. Build grounded prompt
        # ----------------------------------

        prompt = build_rag_prompt(
            query,
            results
        )


        # ----------------------------------
        # 5. Generate answer
        # ----------------------------------

        answer = self.llm.generate(
            prompt
        )


        return answer.strip()


    def close(self):

        self.vector_store.close()
        
def main():

        pipeline = RAGPipeline()

        print("Enterprise RAG Assistant")
        print("Type 'exit' to quit.")

        try:
            while True:

                query = input("\nQuestion: ").strip()

                if not query:
                    continue

                if query.lower() == "exit":
                    break

                answer = pipeline.ask(query)

                print("\nAnswer:")
                print(answer)

        finally:
            pipeline.close()


if __name__ == "__main__":
    main()