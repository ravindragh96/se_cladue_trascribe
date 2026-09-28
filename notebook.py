# ============================================================
# STEP 1 — FETCH 20 AUDIO RECORDINGS FROM AZURE BLOB STORAGE
# ============================================================

%pip install azure-storage-blob azure-identity

# ============================================================
# IMPORTS
# ============================================================

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os

# ============================================================
# AZURE BLOB STORAGE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT_NAME = "<your-storage-account-name>"
CONTAINER_NAME = "<your-container-name>"

ACCOUNT_URL = (
    f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
)

# If the recordings are inside a specific folder, put the
# folder path here. Otherwise keep it empty.
BLOB_PREFIX = ""

# ============================================================
# AUTHENTICATION
# ============================================================

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
)

container_client = blob_service_client.get_container_client(
    CONTAINER_NAME
)

print("Connected to Azure Blob Storage")

# ============================================================
# FETCH MP3 RECORDINGS
# ============================================================

audio_files = []

for blob in container_client.list_blobs(
    name_starts_with=BLOB_PREFIX
):
    
    if blob.name.lower().endswith(".mp3"):
        audio_files.append(blob.name)


print("Total MP3 files found:", len(audio_files))

# ============================================================
# TAKE SAMPLE OF 20 AUDIO FILES
# ============================================================

audio_files = sorted(audio_files)

sample_files = audio_files[:20]

print("Selected audio files:", len(sample_files))

for i, file_name in enumerate(sample_files, 1):
    print(f"{i}. {file_name}")



# ============================================================
# DOWNLOAD THE 20 MP3 FILES
# ============================================================

DOWNLOAD_DIR = "audio_sample_20"

os.makedirs(
    DOWNLOAD_DIR,
    exist_ok=True
)

for i, blob_name in enumerate(sample_files, 1):

    blob_client = container_client.get_blob_client(
        blob_name
    )

    file_name = os.path.basename(blob_name)

    local_path = os.path.join(
        DOWNLOAD_DIR,
        file_name
    )

    with open(local_path, "wb") as file:

        download_stream = blob_client.download_blob()

        file.write(
            download_stream.readall()
        )

    print(
        f"[{i}/20] Downloaded: {file_name}"
    )

print("\n20 audio recordings fetched successfully.")



