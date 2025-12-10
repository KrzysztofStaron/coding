# Upload model to Hugging Face Hub
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "0.5b-coding"
MODEL_PATH = "./qwen2.5-0.5b-coding-h200-final"

if not HF_TOKEN:
    print("Error: HF_TOKEN not found in environment variables.")
    print("Please set HF_TOKEN in your .env file or environment.")
    exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model directory not found at {MODEL_PATH}")
    print("Please ensure the model has been trained and saved first.")
    exit(1)

print(f"\nUploading model from {MODEL_PATH} to Hugging Face Hub: {HF_REPO_ID}\n")
api = HfApi(token=HF_TOKEN)

# Ensure the repo_id is in the form "username/repo_name" (user namespace required!)
if "/" not in HF_REPO_ID:
    username = api.whoami()["name"]
    full_repo_id = f"{username}/{HF_REPO_ID}"
else:
    full_repo_id = HF_REPO_ID

# Create repository if it doesn't exist
try:
    api.create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True)
    print(f"Repository '{full_repo_id}' is ready for upload.\n")
except Exception as e:
    print(f"Note: Repository creation check: {e}")

try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=full_repo_id,
        repo_type="model"
    )
    print("Upload complete!\n")
except Exception as e:
    print(f"Error during upload: {e}")
    print(
        "\nIf this is a private repo or you're using an org, make sure the token is correct and repo exists.\n"
        "Check 'repo_id' format – it must include your username or org, e.g. 'your-username/0.5b-coding'."
    )

