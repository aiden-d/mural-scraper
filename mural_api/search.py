#!/usr/bin/env python3
"""
Semantic search utilities for finding similar items based on embeddings.
"""

from typing import List, Tuple


class SemanticSearch:
    """Class to handle semantic search functionality."""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_vec1 = sum(a * a for a in vec1) ** 0.5
        norm_vec2 = sum(b * b for b in vec2) ** 0.5

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0

        return dot_product / (norm_vec1 * norm_vec2)

    @staticmethod
    def search(
        query_embedding: List[float],
        item_embeddings: List[List[float]],
        items: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """Search for the most similar items to the query."""
        similarities = [
            (
                item,
                SemanticSearch.cosine_similarity(query_embedding, item_embedding),
                idx,
            )
            for idx, (item, item_embedding) in enumerate(zip(items, item_embeddings))
        ]

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:top_k]
