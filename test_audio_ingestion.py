from modules.audio_ingestion import AzureAudioIngestion


STORAGE_ACCOUNT_URL = (
    "https://<your-storage-account>.blob.core.windows.net"
)

CONTAINER_NAME = "<your-container-name>"


audio_ingestion = AzureAudioIngestion(
    storage_account_url=STORAGE_ACCOUNT_URL,
    container_name=CONTAINER_NAME,
)


audio_records = audio_ingestion.get_sample_audio(
    sample_size=20,
    seed=42,
)


print(
    f"\nSelected {len(audio_records)} audio files\n"
)


for record in audio_records:

    print(
        f"ID: {record['audio_id']}"
    )

    print(
        f"File: {record['audio_file']}"
    )

    print(
        f"Blob: {record['blob_name']}"
    )

    print(
        f"Size: {len(record['audio_bytes']) / (1024 * 1024):.2f} MB"
    )

    print("-" * 80)
