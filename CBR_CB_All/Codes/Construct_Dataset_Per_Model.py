"""
source myenv/bin/activate
pip install tqdm psutil
python3 CBR_CB_All/Codes/Construct_Dataset_Per_Model.py > CBR_CB_All/Codes/Construct_Dataset_Per_Model.txt
"""

import json
import time
import resource
from pathlib import Path

from tqdm import tqdm

# ==================================================

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "../../Zero_Shot_Reasoning/Outputs"
OUTPUT_DIR = SCRIPT_DIR / "../Inputs"

THINKING_TAGS = ["false", "na"]                            # thinking-off predictions file is
                                                            # tagged "false" or "na" depending
                                                            # on Models.yaml
INPUT_FILENAMES = [f"predictions_thinking_{tag}.json" for tag in THINKING_TAGS]

FILTER_MODE = "all"                                       # config 1: keep every instance
OUTPUT_FILENAME = "corpus_all.json"

# ==================================================

def filter_records(records, mode):
    if mode == "all":
        return records
    if mode == "failure":
        return [r for r in records if r["evaluation_correct"] is False]
    if mode == "success":
        return [r for r in records if r["evaluation_correct"] is True]
    raise ValueError(f"Unknown FILTER_MODE: {mode}")


def find_input_file(model_dir):
    for filename in INPUT_FILENAMES:
        candidate = model_dir / filename
        if candidate.exists():
            return candidate
    return None


def discover_pairs():
    pairs = []
    for dataset_dir in sorted(d for d in INPUT_DIR.iterdir() if d.is_dir()):
        for model_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            if find_input_file(model_dir) is not None:
                pairs.append((dataset_dir.name, model_dir.name))
    return pairs


def process_pair(dataset_name, model_name):
    input_path = find_input_file(INPUT_DIR / dataset_name / model_name)
    out_dir = OUTPUT_DIR / dataset_name / model_name
    out_path = out_dir / OUTPUT_FILENAME

    if out_path.exists():
        return None  # resumable: already built for this (dataset, model)

    records = json.loads(input_path.read_text())

    t0 = time.time()
    filtered = filter_records(records, FILTER_MODE)
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))

    total = len(records)
    saved = len(filtered)
    proportion = saved / total if total else 0.0
    per_instance = elapsed / total if total else 0.0
    peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes

    return {
        "dataset": dataset_name,
        "model": model_name,
        "mode": FILTER_MODE,
        "saved": saved,
        "total": total,
        "proportion": round(proportion, 4),
        "ram_gb": round(peak_ram_gb, 2),
        "total_time_sec": round(elapsed, 6),
        "per_instance_time_sec": round(per_instance, 8),
    }


if __name__ == "__main__":
    pairs = discover_pairs()
    pbar = tqdm(pairs, desc=f"Building corpora [{FILTER_MODE}]")
    for dataset_name, model_name in pbar:
        pbar.set_postfix(dataset=dataset_name, model=model_name)
        stats = process_pair(dataset_name, model_name)
        if stats is None:
            continue
        print(f"[{stats['model']} | {stats['dataset']}] saved={stats['saved']}/{stats['total']} "
              f"({stats['proportion']:.2%}) | RAM={stats['ram_gb']:.2f} GB "
              f"| total_time={stats['total_time_sec']:.4f}s "
              f"| per_instance={stats['per_instance_time_sec']:.6f}s", flush=True)

    print("Done.")