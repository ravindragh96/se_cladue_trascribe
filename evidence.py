"""
==============================================================================
Track A - Service Experts Revenue Opportunity Detection

Evidence Extractor

Responsibilities
----------------
1. Split transcript into semantic chunks
2. Generate embeddings
3. Compute similarity
4. Return top supporting evidence

This module has NO business logic.
==============================================================================
"""

import logging
from typing import List, Dict

import numpy as np

from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
# ==============================================================================
# Evidence Extractor
# ==============================================================================

class EvidenceExtractor:
    """
    Retrieves the most relevant transcript segments supporting the
    predicted revenue opportunity.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        top_k: int = 5
    ):

        self.registry = registry

        self.embedding_model = registry.load_embedding_model()

        self.top_k = top_k
        # ------------------------------------------------------------------
    # Chunk Transcript
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_transcript(
        transcript: str,
        min_length: int = 20
    ) -> List[str]:
        """
        Split transcript into meaningful conversational chunks.
        """

        if not transcript:
            return []

        transcript = transcript.replace("\r", "\n")

        chunks = []

        for line in transcript.split("\n"):

            line = line.strip()

            if len(line) < min_length:
                continue

            chunks.append(line)

        return chunks
        # ------------------------------------------------------------------
    # Generate Embeddings
    # ------------------------------------------------------------------

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Generates embeddings for transcript chunks.
        """

        if not texts:
            return np.empty((0, 384))

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings

    # ------------------------------------------------------------------
    # Cosine Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Computes cosine similarity between the query embedding
        and all transcript chunk embeddings.
        """

        if chunk_embeddings.size == 0:
            return np.array([])

        similarities = np.dot(
            chunk_embeddings,
            query_embedding
        )

        return similarities

    # ------------------------------------------------------------------
    # Rank Chunks
    # ------------------------------------------------------------------

    def rank_chunks(
        self,
        query: str,
        transcript_chunks: List[str]
    ) -> List[Dict]:
        """
        Returns transcript chunks ranked by semantic similarity.
        """

        if not transcript_chunks:
            return []

        chunk_embeddings = self.generate_embeddings(
            transcript_chunks
        )

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = self.cosine_similarity(
            query_embedding,
            chunk_embeddings
        )

        ranked = []

        for idx, score in enumerate(similarities):

            ranked.append({

                "text": transcript_chunks[idx],

                "score": float(score)

            })

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked
        # ------------------------------------------------------------------
    # Extract Evidence
    # ------------------------------------------------------------------

    def extract(
        self,
        transcript: str,
        analysis_result: Dict,
        top_k: int = None
    ) -> List[Dict]:
        """
        Extract the top supporting evidence from the transcript.

        Parameters
        ----------
        transcript : Full conversation transcript

        analysis_result : Output of RevenueOpportunityAnalyzer

        top_k : Number of evidence chunks to return

        Returns
        -------
        [
            {
                "text": "...",
                "score": 0.92
            }
        ]
        """

        if top_k is None:
            top_k = self.top_k

        chunks = self.chunk_transcript(transcript)

        if not chunks:
            return []

        # --------------------------------------------------------------
        # Build Semantic Search Query
        # --------------------------------------------------------------

        # Preserve compatibility with legacy detector fields while supporting
        # the new client 14-field classification schema. This is *semantic*
        # retrieval, not a replacement for exact transcript evidence quotes.
        query_fields = (
            "summary", "refusal_paraphrase", "call_regarding",
            "trade", "service_category", "core_intent",
            "customer_problem", "customer_goal", "requested_service",
        )
        query_parts = [
            str(analysis_result[field]).strip()
            for field in query_fields
            if isinstance(analysis_result.get(field), str)
            and analysis_result[field].strip()
            and analysis_result[field].strip().lower() not in {"none", "other", "na"}
        ]
        query = " ".join(query_parts)

        if not query.strip():

            query = transcript[:500]

        ranked_chunks = self.rank_chunks(
            query=query,
            transcript_chunks=chunks
        )

        return ranked_chunks[:top_k]

    # ------------------------------------------------------------------
    # Attach Evidence
    # ------------------------------------------------------------------

    def enrich(
        self,
        transcript: str,
        analysis_result: Dict
    ) -> Dict:
        """
        Appends semantic evidence under a separate key for auxiliary QA.

        IMPORTANT: The returned dictionary has an EXTRA key and is not itself
        the client's required 14-field JSON. Use a separate operational field
        when persisting semantic_evidence.

        """

        result = analysis_result.copy()

        # DO NOT overwrite the client's ``evidence`` field. That is a map of
        # verbatim quotes tied to specific classification fields. Similarity
        # results have a different meaning and stay in a separate QA field.
        result["semantic_evidence"] = self.extract(
            transcript=transcript,
            analysis_result=analysis_result
        )

        return result
        # ------------------------------------------------------------------
    # Callable Interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        transcript: str,
        analysis_result: Dict
    ) -> Dict:
        """
        Allows the extractor to be called like a function.

        Example
        -------
        enriched = extractor(
            transcript,
            detector_output
        )
        """

        return self.enrich(
            transcript=transcript,
            analysis_result=analysis_result
        )


# ==============================================================================
# Factory
# ==============================================================================

def create_evidence_extractor(
    registry: ModelRegistry,
    top_k: int = 5
) -> EvidenceExtractor:
    """
    Factory function for creating the EvidenceExtractor.

    Parameters
    ----------
    registry : ModelRegistry
        Shared model registry instance.

    top_k : int
        Number of evidence chunks to return.

    Returns
    -------
    EvidenceExtractor
    """

    return EvidenceExtractor(
        registry=registry,
        top_k=top_k
    )
