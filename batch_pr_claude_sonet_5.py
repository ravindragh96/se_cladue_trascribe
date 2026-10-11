# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
"""Track A — Service Experts / Claude Sonnet batch classification.

Uses modules.detector.RevenueOpportunityAnalyzer with the client EXACT 14-field
contract. Metadata, processing status, errors, QA warnings, and optional semantic
retrieval evidence are operational fields, never client classification fields.

The notebook intentionally DOES NOT execute a batch when imported. Run
``run_project_batch()`` explicitly after validating the configured Volume paths.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from models.model_registry import ModelRegistry
from modules.cleaner import TranscriptCleaner
from modules.detector import CLIENT_FIELDS, RevenueOpportunityAnalyzer
from modules.llm_client import LLMClient

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = (
    "from_number", "to_number", "call_date_utc", "agent_id",
    "agent_name", "queue_name", "audio_transcript_id",
)

DEFAULT_INPUT_PATH = "/Volumes/dev/se_demo_trans/transcripts/Contact_Center_Transcripts.xlsx"
DEFAULT_OUTPUT_PATH = (
    "/Volumes/dev/se_demo_trans/transcripts/"
    "Contact_Center_Transcripts_results_claude_sonnet.json"
)
DEFAULT_DELTA_TABLE = "dev.se_demo_trans.transcript_analysis_claude_sonnet"


def create_pipeline(
    endpoint_name: str = "databricks-claude-sonnet-5",
    prompt_path: Optional[str] = None,
) -> tuple[TranscriptCleaner, RevenueOpportunityAnalyzer, ModelRegistry]:
    """Construct the existing Databricks registry -> LLM -> detector pipeline."""
    registry = ModelRegistry(endpoint_name=endpoint_name)
    llm = LLMClient(registry)
    cleaner = TranscriptCleaner(lowercase=False, collapse_to_single_line=False)
    detector = RevenueOpportunityAnalyzer(llm, prompt_path=prompt_path)
    return cleaner, detector, registry


def _string_or_none(value: Any) -> Optional[str]:
    """Preserve identifiers as strings; ensure JSON/Delta won't receive NaNs."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def read_transcripts_excel(path: str, spark_session: Any) -> pd.DataFrame:
    """Read an Excel workbook from a Databricks Volume using the Spark FS path."""
    binary_df = spark_session.read.format("binaryFile").load(path)
    records = binary_df.select("content").take(2)
    if len(records) != 1:
        raise ValueError(f"Expected one Excel file at {path}; found {len(records)}")
    df = pd.read_excel(io.BytesIO(records[0]["content"]), engine="openpyxl")
    if "transcription_text" not in df:
        raise ValueError("Input Excel must contain a 'transcription_text' column")
    return df


def process_transcript(
    transcript: Any,
    *,
    cleaner: TranscriptCleaner,
    detector: RevenueOpportunityAnalyzer,
    evidence_extractor: Any = None,
) -> Dict[str, Any]:
    """Run one transcript; never convert an inference failure into 'false'."""
    # Excel nulls and non-text values must not be sent to the LLM as
    # literal strings such as "nan", "<NA>", or "123.0".
    cleaned = cleaner.clean(transcript if isinstance(transcript, str) else "")
    result = detector.analyze_with_status(cleaned)
    # Retrieval evidence has cosine-similarity scores and is NOT field-specific
    # verbatim evidence. Keep it separate and optional (requires embedding model).
    result["semantic_evidence"] = []
    if result["processing_status"] == "success" and evidence_extractor is not None:
        try:
            result["semantic_evidence"] = evidence_extractor.extract(
                cleaned, result["client_result"], top_k=3
            )
        except Exception as exc:
            logger.warning("Auxiliary evidence retrieval failed: %s", type(exc).__name__)
            result["qa_warnings"].append("semantic_evidence_extraction_failed")
    return result


def process_dataframe(
    df: pd.DataFrame,
    *,
    cleaner: TranscriptCleaner,
    detector: RevenueOpportunityAnalyzer,
    evidence_extractor: Any = None,
    batch_size: int = 20,
) -> pd.DataFrame:
    """Return one operational record per input call, including failures."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if "transcription_text" not in df:
        raise ValueError("Input dataframe must contain 'transcription_text'")

    results: List[Dict[str, Any]] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        logger.info("Classifying calls %d–%d of %d", start + 1, start + len(batch), len(df))
        for original_index, row in batch.iterrows():
            call_id = _string_or_none(row.get("call_id")) or f"Unknown_{original_index}"
            source_ids = {name: _string_or_none(row.get(name)) for name in SOURCE_COLUMNS}
            try:
                run = process_transcript(
                    row.get("transcription_text"),
                    cleaner=cleaner,
                    detector=detector,
                    evidence_extractor=evidence_extractor,
                )
                results.append({"call_id": call_id, **source_ids, **run})
            except Exception as exc:
                logger.exception("Unexpected processing error for call %s", call_id)
                results.append({
                    "call_id": call_id,
                    **source_ids,
                    "processing_status": "pipeline_error",
                    "client_result": None,
                    "error": f"Unexpected pipeline error ({type(exc).__name__}).",
                    "qa_warnings": [],
                    "semantic_evidence": [],
                })
    return pd.DataFrame(results)


def write_results_json(results: pd.DataFrame, output_path: str) -> None:
    """Persist nested client JSON and separate operational metadata in a Volume."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(results.to_dict(orient="records"), fp, indent=2, ensure_ascii=False)


def flatten_for_delta(results: pd.DataFrame) -> List[Dict[str, Any]]:
    """Flatten only for analytics columns; evidence stays serialized JSON."""
    output = []
    for row in results.to_dict(orient="records"):
        client = row.get("client_result")
        record = {
            "call_id": _string_or_none(row.get("call_id")),
            **{key: _string_or_none(row.get(key)) for key in SOURCE_COLUMNS},
            "processing_status": row.get("processing_status"),
            "error": row.get("error"),
            "qa_warnings_json": json.dumps(row.get("qa_warnings") or [], ensure_ascii=False),
            "semantic_evidence_json": json.dumps(
                row.get("semantic_evidence") or [], ensure_ascii=False
            ),
        }
        for field in CLIENT_FIELDS:
            if field == "evidence":
                record["evidence_json"] = json.dumps(
                    (client or {}).get("evidence") if client else None,
                    ensure_ascii=False,
                )
            else:
                record[field] = client.get(field) if client is not None else None
        output.append(record)
    return output


def save_delta_table(
    results: pd.DataFrame, spark_session: Any,
    table_name: str = DEFAULT_DELTA_TABLE, mode: str = "overwrite",
) -> Any:
    """Write a stable, typed schema even if all rows fail classification."""
    from pyspark.sql.types import (
        BooleanType, IntegerType, StringType, StructField, StructType,
    )

    string_fields = ["call_id", *SOURCE_COLUMNS, "processing_status", "error"]
    schema_fields = [StructField(key, StringType(), True) for key in string_fields]
    for key in CLIENT_FIELDS:
        if key == "evidence":
            schema_fields.append(StructField("evidence_json", StringType(), True))
        elif key in ("revenue_opportunity", "agent_offered_booking"):
            schema_fields.append(StructField(key, BooleanType(), True))
        elif key == "intent_strength":
            schema_fields.append(StructField(key, IntegerType(), True))
        else:
            schema_fields.append(StructField(key, StringType(), True))
    schema_fields.extend([
        StructField("qa_warnings_json", StringType(), True),
        StructField("semantic_evidence_json", StringType(), True),
    ])
    spark_df = spark_session.createDataFrame(
        flatten_for_delta(results), schema=StructType(schema_fields)
    )
    spark_df.write.mode(mode).saveAsTable(table_name)
    return spark_df


def run_project_batch(
    spark_session: Any,
    *,
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    delta_table: str = DEFAULT_DELTA_TABLE,
    batch_size: int = 20,
    prompt_path: Optional[str] = None,
    with_semantic_evidence: bool = False,
) -> pd.DataFrame:
    """Entry point; optionally enable expensive, auxiliary embedding retrieval."""
    cleaner, detector, registry = create_pipeline(prompt_path=prompt_path)
    extractor = None
    if with_semantic_evidence:
        from modules.evidence import EvidenceExtractor
        extractor = EvidenceExtractor(registry, top_k=3)
    df = read_transcripts_excel(input_path, spark_session)
    results = process_dataframe(
        df, cleaner=cleaner, detector=detector,
        evidence_extractor=extractor, batch_size=batch_size,
    )
    write_results_json(results, output_path)
    save_delta_table(results, spark_session, delta_table)
    return results

# COMMAND ----------
# For Databricks interactive execution, after confirming input/outputs:
# results = run_project_batch(spark)
# display(results)
#
# Or use a shared prompt on a UC Volume:
# results = run_project_batch(spark, prompt_path="/Volumes/.../prompt.txt")
