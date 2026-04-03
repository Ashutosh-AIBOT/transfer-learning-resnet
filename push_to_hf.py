import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
REPO_ID = "Ashutosh1975/transfer-learning-resnet" 
PROJECT_FOLDER = "/home/ashutosh/Desktop/NO-AI-USE/Deep-learning/Transfer Learning & Fine-tuning"

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

print(f"🚀 Starting professional Docker-based deployment of {PROJECT_FOLDER} to {REPO_ID}...")

# 1. Create space if not exists
try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", private=False)
    print(f"✅ Created/Verified Space: {REPO_ID}")
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Space {REPO_ID} already exists. Proceeding with update.")
    else:
        print(f"❌ Error creating repo: {e}")

# 2. Upload the folder contents
api.upload_folder(
    folder_path=PROJECT_FOLDER,
    repo_id=REPO_ID,
    repo_type="space",
    allow_patterns=["*.py", "*.txt", "*.md", "Dockerfile", "models/*", "data/artifacts/*", "charts/*", "data/processed/*.json"],
    ignore_patterns=["notebooks/*", "__pycache__/*", "*.pdf", ".git/*", ".gitattributes", "push_to_hf.py", "data/raw/*", "project_problem.md"],
)

print("-" * 30)
print(f"✅ SUCCESS! Your Dockerized app is live at: https://huggingface.co/spaces/{REPO_ID}")
print("Monitor build logs at the URL above. It will take ~5 mins to build the container.")
print("-" * 30)
