import os
import random

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


class AzureAudioIngestion:

    def __init__(
        self,
        storage_account_url: str,
        container_name: str,
        prefix: str = "",
    ):

        self.container_name = container_name
        self.prefix = prefix

        self.credential = DefaultAzureCredential()

        self.blob_service_client = BlobServiceClient(
            account_url=storage_account_url,
            credential=self.credential,
        )

        self.container_client = (
            self.blob_service_client
            .get_container_client(
                container_name
            )
        )


    def list_mp3_files(self):

        mp3_files = []

        blobs = self.container_client.list_blobs(
            name_starts_with=self.prefix
        )

        for blob in blobs:

            if blob.name.lower().endswith(".mp3"):

                mp3_files.append(
                    blob.name
                )

        return mp3_files


    def sample_files(
        self,
        sample_size: int = 20,
        seed: int = 42,
    ):

        mp3_files = self.list_mp3_files()

        if not mp3_files:

            raise RuntimeError(
                "No MP3 audio files found in Azure Blob Storage."
            )

        random.seed(seed)

        sample_size = min(
            sample_size,
            len(mp3_files)
        )

        return random.sample(
            mp3_files,
            sample_size
        )


    def download_audio(
        self,
        blob_name: str,
    ):

        blob_client = (
            self.container_client
            .get_blob_client(
                blob_name
            )
        )

        audio_bytes = (
            blob_client
            .download_blob()
            .readall()
        )

        return audio_bytes


    def get_sample_audio(
        self,
        sample_size: int = 20,
        seed: int = 42,
    ):

        selected_files = self.sample_files(
            sample_size=sample_size,
            seed=seed,
        )

        records = []

        for blob_name in selected_files:

            audio_bytes = self.download_audio(
                blob_name
            )

            filename = os.path.basename(
                blob_name
            )

            audio_id = os.path.splitext(
                filename
            )[0]

            records.append({

                "audio_id":
                    audio_id,

                "audio_file":
                    filename,

                "blob_name":
                    blob_name,

                "audio_bytes":
                    audio_bytes,

            })

        return records
