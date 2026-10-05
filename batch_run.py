import sys
import os
import json
import pandas as pd

# Set working directory to load local module packages
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 1. Import pipeline components
from models.model_registry import ModelRegistry
from modules.detector import RevenueOpportunityAnalyzer
from modules.evidence import EvidenceExtractor
from modules.cleaner import TranscriptCleaner
from modules.llm_client import LLMClient

# 2. Initialize Models & Pipeline Components
print("Initializing AI models...")
registry = ModelRegistry()
tokenizer, model = registry.load_llm()
embedding_model = registry.load_embedding_model()

llm = LLMClient(tokenizer, model)
detector = RevenueOpportunityAnalyzer(llm)
cleaner = TranscriptCleaner()
# Pass the semantic embedding model to EvidenceExtractor
evidence_extractor = EvidenceExtractor(embedding_model)

# 3. Load Sample File & Align Column Schema
file_path = "SE_sample_trans.xlsx"  # Or your Databricks path if running in cloud

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
else:
    # Databricks Spark fallback
    binary_df = spark.read.format("binaryFile").load(file_path)
    content = binary_df.collect()[0]["content"]
    import io
    df = pd.read_excel(io.BytesIO(content))

# Standardize ID column ('audio_id' -> 'call_id' if needed)
if "call_id" not in df.columns:
    if "audio_id" in df.columns:
        df["call_id"] = df["audio_id"].astype(str)
    elif "audio_file" in df.columns:
        df["call_id"] = df["audio_file"].astype(str)
    else:
        df["call_id"] = [f"sample_{i}" for i in range(len(df))]

# Standardize transcript column
transcript_col = next(
    (col for col in df.columns if any(k in str(col).lower() for k in ["transcript", "text"])),
    None
)
if transcript_col != "transcription_text":
    df["transcription_text"] = df[transcript_col]

print(f"Loaded {len(df)} records. Sample Call ID: {df.iloc[0]['call_id']}")

# 4. Batch Analysis Function
from tqdm import tqdm

def run_batch_analysis(
    dataframe: pd.DataFrame,
    output_json_path: str = "transcripts_results.json",
    batch_size: int = 10,
    top_k: int = 2
):
    results = []
    total = len(dataframe)
    print(f"Processing {total} transcripts...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataframe.iloc[start:end]
        print(f"\nBatch {start // batch_size + 1} ({start} to {end - 1})")

        for index, row in tqdm(batch.iterrows(), total=len(batch)):
            call_id = str(row.get("call_id", f"Unknown_{index}"))
            raw_transcript = row.get("transcription_text", "")

            if not isinstance(raw_transcript, str) or not raw_transcript.strip():
                continue

            try:
                # Step 1: Clean Transcript
                cleaned_text = cleaner.clean(raw_transcript)

                # Step 2: Analyze Business Intent
                intent_results = detector.analyze(cleaned_text)

                # Step 3: Extract Semantic Evidence
                evidence_results = evidence_extractor.extract(
                    cleaned_text,
                    intent_results,
                    top_k=top_k
                )

                results.append({
                    "call_id": call_id,
                    "core_intent": intent_results.get("core_intent", ""),
                    "customer_problem": intent_results.get("customer_problem", ""),
                    "customer_goal": intent_results.get("customer_goal", ""),
                    "requested_service": intent_results.get("requested_service", ""),
                    "trade": intent_results.get("trade", "none"),
                    "service_category": intent_results.get("service_category", "none"),
                    "urgency": intent_results.get("urgency", "none"),
                    "appointment_status": intent_results.get("appointment_status", "Not Scheduled"),
                    "pricing_discussed": intent_results.get("pricing_discussed", False),
                    "revenue_opportunity": intent_results.get("revenue_opportunity", False),
                    "intent_strength": intent_results.get("intent_strength", 1),
                    "confidence": intent_results.get("confidence", "low"),
                    "confidence_score": intent_results.get("confidence_score", 0),
                    "evidence": evidence_results,
                    "business_summary": intent_results.get("business_summary", ""),
                    "business_reason": intent_results.get("business_reason", ""),
                    "status": "Success"
                })

            except Exception as e:
                print(f"Error in Call ID {call_id}: {e}")
                results.append({
                    "call_id": call_id,
                    "status": "Failed",
                    "error_message": str(e)
                })

    # Save output
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults successfully saved to: {output_json_path}")
    return pd.DataFrame(results)

# 5. Execute Pipeline
results_df = run_batch_analysis(df, output_json_path="output_results.json")
results_df.head()
