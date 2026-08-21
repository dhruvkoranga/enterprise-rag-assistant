from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 3
    ):

        if not results:
            return []

        pairs = [
            (
                query,
                result.payload["text"]
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        scored_results = list(
            zip(results, scores)
        )

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True
        )

        reranked_results = []

        for result, score in scored_results[:top_k]:

            result.score = float(score)

            reranked_results.append(
                result
            )

        return reranked_results