"""Semantic evidence retrieval for Service Experts transcripts.

No revenue classification is performed here. The detector's verbatim
`evidence` field is preserved; embeddings are stored as `semantic_evidence`.
"""
import logging
import re
from typing import Dict, List, Optional

import numpy as np
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class EvidenceExtractor:
    def __init__(self, registry: ModelRegistry, top_k: int = 5):
        self.registry = registry
        self.embedding_model = registry.load_embedding_model()
        self.top_k = top_k

    @staticmethod
    def chunk_transcript(transcript: str, min_length: int = 20) -> List[str]:
        if not transcript:
            return []
        text = str(transcript).replace("\r", "\n")
        # Diarized turns can be concatenated on one line.
        text = re.sub(
            r"(?<!^)\s+(?=(?:CSR|CUSTOMER)\s*:)",
            "\n", text, flags=re.IGNORECASE
        )
        return [line.strip() for line in text.split("\n")
                if len(line.strip()) >= min_length]

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        return np.asarray(self.embedding_model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        ))

    @staticmethod
    def cosine_similarity(query_embedding: np.ndarray,
                          chunk_embeddings: np.ndarray) -> np.ndarray:
        if chunk_embeddings.size == 0:
            return np.array([])
        return np.dot(chunk_embeddings, query_embedding)

    def rank_chunks(self, query: str, transcript_chunks: List[str]) -> List[Dict]:
        if not transcript_chunks:
            return []
        embeddings = self.generate_embeddings(transcript_chunks)
        query_embedding = np.asarray(self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ))
        scores = self.cosine_similarity(query_embedding, embeddings)
        return sorted(
            ({"text": chunk, "score": float(score)}
             for chunk, score in zip(transcript_chunks, scores)),
            key=lambda item: item["score"], reverse=True
        )

    def extract(self, transcript: str, analysis_result: Dict,
                top_k: Optional[int] = None) -> List[Dict]:
        chunks = self.chunk_transcript(transcript)
        if not chunks:
            return []
        fields = ("core_intent", "customer_problem", "customer_goal",
                  "requested_service", "business_summary", "business_reason")
        parts = [analysis_result[key].strip() for key in fields
                 if isinstance(analysis_result.get(key), str)
                 and analysis_result[key].strip()
                 and analysis_result[key].strip().lower() not in {"none", "unclear"}]
        query = " ".join(parts) or transcript[:500]
        return self.rank_chunks(query, chunks)[:(self.top_k if top_k is None else top_k)]

    def enrich(self, transcript: str, analysis_result: Dict) -> Dict:
        result = analysis_result.copy()
        result["semantic_evidence"] = self.extract(transcript, analysis_result)
        return result

    def __call__(self, transcript: str, analysis_result: Dict) -> Dict:
        return self.enrich(transcript, analysis_result)


def create_evidence_extractor(registry: ModelRegistry,
                              top_k: int = 5) -> EvidenceExtractor:
    return EvidenceExtractor(registry=registry, top_k=top_k)
