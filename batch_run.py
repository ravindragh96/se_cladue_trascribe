"""Databricks batch runner; preserves all original detector KPIs.

Set FILE_PATH below (or define file_path in the notebook before executing).
"""
import io
import json
import os
import sys

import pandas as pd
from tqdm.auto import tqdm

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.model_registry import ModelRegistry
from modules.detector import RevenueOpportunityAnalyzer, DEFAULT_RESPONSE
from modules.evidence import EvidenceExtractor
from modules.cleaner import TranscriptCleaner
from modules.llm_client import LLMClient

FILE_PATH = globals().get("file_path", "SE_sample_trans.xlsx")

registry = ModelRegistry()
llm = LLMClient(registry)
detector = RevenueOpportunityAnalyzer(llm)
cleaner = TranscriptCleaner()

# Evidence is supplementary. If embedding model loading fails, keep classifying.
try:
    evidence_extractor = EvidenceExtractor(registry)
    evidence_init_error = None
except Exception as exc:
    evidence_extractor = None
    evidence_init_error = f"{type(exc).__name__}: {exc}"
    print(f"Warning: semantic evidence disabled: {evidence_init_error}")

if os.path.exists(FILE_PATH):
    df = pd.read_excel(FILE_PATH)
else:
    # Databricks binaryFile fallback for paths accessible through Spark.
    binary_df = spark.read.format("binaryFile").load(FILE_PATH)
    content = binary_df.collect()[0]["content"]
    df = pd.read_excel(io.BytesIO(content))

# Normalize column names for matching only; preserve original data.
normalized_names = {str(col).strip().lower(): col for col in df.columns}
transcript_col = next(
    (normalized_names[key] for key in ("transcription_text", "transcript", "transcription", "text")
     if key in normalized_names),
    None,
)
if transcript_col is None:
    transcript_col = next(
        (col for col in df.columns if "transcript" in str(col).lower()), None
    )
if transcript_col is None:
    raise ValueError(f"No transcript column found. Columns: {list(df.columns)}")

df["transcription_text"] = df[transcript_col]
if "audio_id" not in df.columns:
    df["audio_id"] = df["audio_file"] if "audio_file" in df.columns else [
        f"audio_{i}" for i in range(len(df))
    ]
if "call_id" not in df.columns:
    df["call_id"] = df["audio_id"]


def _blank(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def run_batch_analysis(dataframe: pd.DataFrame,
                       output_json_path: str = "output_results.json",
                       output_excel_path: str = "output_results.xlsx",
                       batch_size: int = 10,
                       top_k: int = 4) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    results = []
    print(f"Processing {len(dataframe)} transcripts...")

    for start in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[start:start + batch_size]
        print(f"Batch {start // batch_size + 1}: rows {start}–{start + len(batch) - 1}")
        for offset, (_, row) in enumerate(tqdm(batch.iterrows(), total=len(batch))):
            row_number = start + offset
            audio_id = str(row.get("audio_id", f"audio_{row_number}"))
            call_id = str(row.get("call_id", audio_id))
            raw = row.get("transcription_text")
            base = {
                "row_number": row_number,
                "audio_id": audio_id,
                "call_id": call_id,
                "transcription_text": "" if _blank(raw) else str(raw),
                **{key: None for key in DEFAULT_RESPONSE},
                "semantic_evidence": [],
                "status": None,
                "error_message": None,
                "evidence_error": evidence_init_error,
            }
            if _blank(raw):
                base.update(status="Skipped", error_message="Empty/missing transcript")
                results.append(base)
                continue

            try:
                cleaned = cleaner.clean(str(raw))
                if not isinstance(cleaned, str) or not cleaned.strip():
                    raise ValueError("Transcript empty after cleaning")
                base["transcription_text"] = cleaned
                intent = detector.analyze(cleaned)
                if not isinstance(intent, dict):
                    raise TypeError("Detector did not return a dictionary")
                reason = str(intent.get("business_reason") or "")
                if reason.startswith("PROCESSING_ERROR:"):
                    raise ValueError(reason)
                # Preserve the complete original KPI dictionary.
                for key in DEFAULT_RESPONSE:
                    base[key] = intent.get(key)
                base["status"] = "Success"
                # Evidence failure must not discard successful intent analysis.
                if evidence_extractor is not None:
                    try:
                        base["semantic_evidence"] = evidence_extractor.extract(
                            cleaned, intent, top_k=top_k
                        )
                    except Exception as evidence_exc:
                        base["evidence_error"] = (
                            f"{type(evidence_exc).__name__}: {evidence_exc}"
                        )
            except Exception as exc:
                base["status"] = "Failed"
                base["error_message"] = f"{type(exc).__name__}: {exc}"
                print(f"Call {call_id} failed: {type(exc).__name__}: {exc}")
            results.append(base)

    output = pd.DataFrame(results)
    for path in (output_json_path, output_excel_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump([_json_safe(record) for record in results], handle,
                  indent=2, ensure_ascii=False, default=str)
    excel_output = output.copy()
    for col in ("evidence", "semantic_evidence"):
        excel_output[col] = excel_output[col].apply(
            lambda item: json.dumps(item, ensure_ascii=False, default=str)
        )
    excel_output.to_excel(output_excel_path, index=False)
    print("Batch complete:", output["status"].value_counts(dropna=False).to_dict())
    print("JSON:", output_json_path)
    print("Excel:", output_excel_path)
    return output


results_df = run_batch_analysis(df, batch_size=10, top_k=4)
try:
    display(results_df)
except NameError:
    print(results_df.head())
