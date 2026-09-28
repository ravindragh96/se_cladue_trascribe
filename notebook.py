SOURCE_PATH = "abfss://staging@stsdev.dfs.core.windows.net/contct_center"

# Databricks working location for the 20 selected audio files
LOCAL_AUDIO_PATH = "/tmp/service_experts_audio_20"

display(dbutils.fs.ls(SOURCE_PATH))
audio_files = []

for file in dbutils.fs.ls(SOURCE_PATH):
    if file.path.lower().endswith(".mp3"):
        audio_files.append(file.path)

print("Total MP3 files found:", len(audio_files))
audio_files = sorted(audio_files)

sample_files = audio_files[:20]

print("Selected MP3 files:", len(sample_files))

for i, file_path in enumerate(sample_files, 1):
    print(f"{i}. {file_path}")

dbutils.fs.rm(LOCAL_AUDIO_PATH, True)
dbutils.fs.mkdirs(LOCAL_AUDIO_PATH)

print("Created:", LOCAL_AUDIO_PATH)

for i, source_file in enumerate(sample_files, 1):

    file_name = source_file.split("/")[-1]
    destination_file = f"{LOCAL_AUDIO_PATH}/{file_name}"

    dbutils.fs.cp(source_file, destination_file)

    print(f"[{i}/20] Copied: {file_name}")

downloaded_files = [
    file.path
    for file in dbutils.fs.ls(LOCAL_AUDIO_PATH)
    if file.path.lower().endswith(".mp3")
]

print("Total audio files copied:", len(downloaded_files))

for i, file_path in enumerate(sorted(downloaded_files), 1):
    print(f"{i}. {file_path}")

audio_df = spark.createDataFrame(
    [(i + 1, path) for i, path in enumerate(sorted(downloaded_files))],
    ["file_id", "audio_path"]
)

display(audio_df)

count = audio_df.count()

if count == 20:
    print("SUCCESS: 20 MP3 files are ready for processing.")
else:
    print(f"WARNING: Only {count} MP3 files are available.")
