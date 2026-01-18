from huggingface_hub import snapshot_download

repo_id = "BAAI/bge-base-en-v1.5"
local_dir = "bge-base-en-v1.5"  # 下载到当前目录下这个文件夹

print(f"Downloading {repo_id} to {local_dir} ...")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Done. Model saved to:", local_dir)
