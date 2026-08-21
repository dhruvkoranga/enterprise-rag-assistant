from app.query_transformer import QueryTransformer


class AdaptiveRetriever:

    def __init__(
        self,
        retriever,
        query_transformer,
        margin_threshold=0.05,
        rrf_k=60,
        original_weight=1.0,
        transformed_weight=1.5,
    ):
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.margin_threshold = margin_threshold
        self.rrf_k = rrf_k
        self.original_weight = original_weight
        self.transformed_weight = transformed_weight

    def _calculate_margin(self, results):
        """
        Calculate the difference between the top two
        retrieval scores.
        """
        if len(results) < 2:
            return float("inf")

        return results[0].score - results[1].score

    def _rrf_fusion(self, original_results, transformed_results, top_k):

        fused_scores = {}
        result_map = {}

        # Original query results
        for rank, result in enumerate(original_results, start=1):

            result_id = str(result.id)

            result_map[result_id] = result

            fused_scores[result_id] = (
                fused_scores.get(result_id, 0)
                + self.original_weight / (self.rrf_k + rank)
            )

        # Transformed query results
        for rank, result in enumerate(transformed_results, start=1):

            result_id = str(result.id)

            result_map[result_id] = result

            fused_scores[result_id] = (
                fused_scores.get(result_id, 0)
                + self.transformed_weight / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fused_scores,
            key=lambda result_id: fused_scores[result_id],
            reverse=True,
        )

        fused_results = []

        for result_id in ranked_ids[:top_k]:

            fused_results.append({
                "result": result_map[result_id],
                "rrf_score": fused_scores[result_id],
            })

        return fused_results

    def retrieve(self, query, top_k=3):
        """
        Retrieve documents using adaptive query transformation.

        If the original retrieval has a low confidence margin,
        transform the query and fuse the original and transformed
        retrieval rankings using RRF.
        """

        # Step 1: Original retrieval
        original_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        initial_margin = self._calculate_margin(original_results)

        # If retrieval is confident, use original results
        if initial_margin >= self.margin_threshold:

            return {
                "results": [
                    {
                        "result": result,
                        "rrf_score": None,
                    }
                    for result in original_results
                ],
                "transformed": False,
                "original_query": query,
                "transformed_query": None,
                "initial_margin": initial_margin,
            }

        # Step 2: Transform ambiguous query
        transformed_query = self.query_transformer.transform(query)

        print("\n[Adaptive Retrieval]")
        print(f"Original query: {query}")
        print(f"Initial margin: {initial_margin:.4f}")
        print(f"Transformed query: {transformed_query}")

        # Step 3: Retrieve using transformed query
        transformed_results = self.retriever.retrieve(
            query=transformed_query,
            top_k=top_k,
        )

        # Step 4: Fuse both rankings using RRF
        fused_results = self._rrf_fusion(
            original_results=original_results,
            transformed_results=transformed_results,
            top_k=top_k,
        )

        return {
            "results": fused_results,
            "transformed": True,
            "original_query": query,
            "transformed_query": transformed_query,
            "initial_margin": initial_margin,
        }