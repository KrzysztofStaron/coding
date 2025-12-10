# Upload model to Hugging Face Hub
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

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

print(f"Uploading model from {MODEL_PATH} to Hugging Face Hub: {HF_REPO_ID}")
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_REPO_ID,
    repo_type="model"
)
print("Upload complete!")

