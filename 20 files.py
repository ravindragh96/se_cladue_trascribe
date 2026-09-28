import os
import requests
import pandas as pd
from azure.identity import DefaultAzureCredential


# ============================================================
# AZURE TRANSCRIPTION ENDPOINT
# ============================================================

AZURE_TRANSCRIBE_ENDPOINT = (
    "https://service-experts-demo-resource.openai.azure.com"
    "/openai/deployments/gpt-4o-transcribe-diarize-2"
    "/audio/transcriptions"
    "?api-version=2025-03-01-preview"
)


# ============================================================
# LOCAL AUDIO FOLDER
# ============================================================

AUDIO_FOLDER = r"C:\Users\YourName\Downloads\audio_sample_20"


audio_files = sorted([
    os.path.join(AUDIO_FOLDER, f)
    for f in os.listdir(AUDIO_FOLDER)
    if f.lower().endswith(".mp3")
])


print("MP3 files found:", len(audio_files))

for f in audio_files:
    print(os.path.basename(f))


AUDIO_FOLDER = r"C:\Users\YourName\Downloads\audio_sample_20"
credential = DefaultAzureCredential()

token = credential.get_token(
    "https://cognitiveservices.azure.com/.default"
).token

headers = {
    "Authorization": f"Bearer {token}"
}

print("Authentication successful")

audio_path = audio_files[0]

audio_file = os.path.basename(audio_path)
audio_id = os.path.splitext(audio_file)[0]

print("Testing file:")
print(audio_file)

with open(audio_path, "rb") as audio:

    response = requests.post(
        AZURE_TRANSCRIBE_ENDPOINT,
        headers=headers,

        files={
            "file": (
                audio_file,
                audio,
                "audio/mpeg"
            )
        },

        data={
            "response_format": "diarized_json",
            "language": "en",
            "chunking_strategy": "auto"
        },

        timeout=1800
    )


print("HTTP Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
else:
    print("Transcription successful")


result = response.json()

transcript_lines = []

for segment in result.get("segments", []):

    speaker = segment.get("speaker", "").strip()
    text = segment.get("text", "").strip()

    if text:
        transcript_lines.append(
            f"{speaker}: {text}"
        )

transcript = "\n".join(transcript_lines)

print(transcript)
test_df = pd.DataFrame([
    {
        "audio_id": audio_id,
        "audio_file": audio_file,
        "transcript": transcript
    }
])

display(test_df)













######################################################################################

import os
import requests
import pandas as pd
from azure.identity import DefaultAzureCredential
AZURE_TRANSCRIBE_ENDPOINT = (
    "https://service-experts-demo-resource.openai.azure.com"
    "/openai/deployments/gpt-4o-transcribe-diarize-2"
    "/audio/transcriptions"
    "?api-version=2025-03-01-preview"
)

TRANSCRIPT_OUTPUT_PATH = "/tmp/service_experts_transcripts"

dbutils.fs.rm(TRANSCRIPT_OUTPUT_PATH, True)
dbutils.fs.mkdirs(TRANSCRIPT_OUTPUT_PATH)
credential = DefaultAzureCredential()

token = credential.get_token(
    "https://cognitiveservices.azure.com/.default"
).token

headers = {
    "Authorization": f"Bearer {token}"
}
results = []

for i, audio_path in enumerate(sample_files, 1):

    audio_file = audio_path.split("/")[-1]
    audio_id = os.path.splitext(audio_file)[0]

    print(f"[{i}/20] Processing: {audio_file}")

    # Copy DBFS/ABFSS file to driver local filesystem
    local_audio_path = f"/tmp/{audio_file}"

    dbutils.fs.cp(
        audio_path,
        f"file:{local_audio_path}"
    )

    try:

        with open(local_audio_path, "rb") as audio:

            response = requests.post(
                AZURE_TRANSCRIBE_ENDPOINT,
                headers=headers,

                files={
                    "file": (
                        audio_file,
                        audio,
                        "audio/mpeg"
                    )
                },

                data={
                    "response_format": "diarized_json",
                    "language": "en",
                    "chunking_strategy": "auto"
                },

                timeout=1800
            )

        if response.status_code != 200:
            print(f"FAILED: {response.status_code}")
            print(response.text)
            continue

        result = response.json()

        # ------------------------------------------------
        # Build one complete transcript for this audio
        # ------------------------------------------------

        transcript_lines = []

        for segment in result.get("segments", []):

            speaker = segment.get("speaker", "").strip()
            text = segment.get("text", "").strip()

            if text:
                transcript_lines.append(
                    f"{speaker}: {text}"
                )

        transcript = "\n".join(transcript_lines)

        results.append({
            "audio_id": audio_id,
            "audio_file": audio_file,
            "transcript": transcript
        })

        print(f"SUCCESS: {audio_file}")

    except Exception as e:

        print(f"ERROR: {audio_file}")
        print(str(e))

transcript_df = pd.DataFrame(
    results,
    columns=[
        "audio_id",
        "audio_file",
        "transcript"
    ]
)

print("Total transcripts:", len(transcript_df))

display(transcript_df)

excel_path = "/tmp/service_experts_20_transcripts.xlsx"

transcript_df.to_excel(
    excel_path,
    index=False
)

print(f"Saved: {excel_path}")
