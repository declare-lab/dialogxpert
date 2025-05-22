from huggingface_hub import snapshot_download

# Define the repo ID and your desired download location
repo_id = "Qwen/Qwen1.5-1.8B-Chat"
local_dir = "Qwen1.5-1.8B"

# Download all files in the repo to the local directory
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)