"""
source myenv/bin/activate
pip install -U huggingface_hub pyyaml
python3 Models/Codes/Model_RAM_Usage.py > Models/Codes/Model_RAM_Usage.txt
"""

import yaml
from pathlib import Path
 
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "../Models"
models = yaml.safe_load(open(SCRIPT_DIR / "models.yaml"))
 
def folder_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)
 
print(f"{'Model':<28} {'Size (GB)':>10}")
print("-" * 40)
 
total_gb = 0.0
for name in models:
    path = MODELS_DIR / name
    if not path.exists():
        print(f"{name:<28} {'missing':>10}")
        continue
    size_gb = folder_size_gb(path)
    total_gb += size_gb
    print(f"{name:<28} {size_gb:>10.2f}")
 
print("-" * 40)
print(f"{'Total':<28} {total_gb:>10.2f}")