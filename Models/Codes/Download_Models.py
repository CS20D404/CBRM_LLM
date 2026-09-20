"""
source myenv/bin/activate
pip install -U huggingface_hub pyyaml
python3 Models/Codes/Download_Models.py > Models/Codes/Download_Models.txt
"""

import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
 
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "../Models"
models = yaml.safe_load(open(SCRIPT_DIR / "models.yaml"))
 
for name, info in models.items():
    print(f"\n=== {info['repo_id']} -> {OUTPUT_DIR / name} ===")
    snapshot_download(repo_id=info["repo_id"], local_dir=OUTPUT_DIR / name)