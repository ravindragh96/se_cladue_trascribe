"""
==============================================================================
Track A - Service Experts Revenue Opportunity Detection

Model Registry

Responsibilities
----------------
1. Initialize Databricks Foundation Model client
2. Load embedding model
3. Execute Claude Sonnet inference

This module contains NO business logic.

Author
------
Ravindra Gantasala
==============================================================================
"""

import logging
from typing import List, Dict, Optional

from mlflow.deployments import get_deploy_client
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelRegistry:

    def __init__(
        self,
        endpoint_name: str = "databricks-claude-sonnet-5"
    ):

        logger.info("Initializing Model Registry...")

        self.endpoint_name = endpoint_name

        self.client = get_deploy_client("databricks")

        self.embedding_model = None

        logger.info("Model Registry initialized.")

    # ------------------------------------------------------------------
    # Embedding Model
    # ------------------------------------------------------------------

    def load_embedding_model(self):

        if self.embedding_model is None:

            logger.info("Loading embedding model...")

            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            logger.info("Embedding model loaded.")

        return self.embedding_model

    # ------------------------------------------------------------------
    # Claude Prediction
    # ------------------------------------------------------------------

    def predict_llm(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1500,
        endpoint_name: Optional[str] = None
    ):

        endpoint = endpoint_name or self.endpoint_name

        response = self.client.predict(
            endpoint=endpoint,
            inputs={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

        return response