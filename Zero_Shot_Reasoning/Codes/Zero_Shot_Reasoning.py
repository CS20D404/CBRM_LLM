"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil
python -u Zero_Shot_Reasoning/Codes/Zero_Shot_Reasoning.py > Zero_Shot_Reasoning/Codes/Zero_Shot_Reasoning.txt
"""

import json
import re
import time
import resource
from pathlib import Path

import yaml
from tqdm import tqdm



# ==================================================

SEED = 42
TOP_N = 5                           # -1 => run on all instances, else first N per dataset
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
PROMPT_FILE = SCRIPT_DIR / "Zero_Shot_Prompt.txt"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
OUTPUT_DIR = SCRIPT_DIR / "../Outputs"
SAMPLE_QUERIES_FILE = SCRIPT_DIR / "../Outputs/Sample_LLM_Queries.json"
EVAL_RESULTS_FILE = SCRIPT_DIR / "../Outputs/Zero_Shot_Evaluation_Results.jsonl"

THINKING_ORDER = [False, True]      # thinking=false for all models first, then thinking=true for all models
MAX_TOKENS_THINKING = 1024
MAX_TOKENS_ANSWER = 20

# ==================================================


def format_prompt(template, ex):
    options = "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
    return template.format(problem_statement=ex["problem_statement"], options=options)


def extract_letter(text):
    # Ordered from most- to least-specific. Within each pattern, the LAST match
    # wins, since thinking output may mention other letters before the final answer.
    patterns = [
        r"Option:\s*\(?([A-E])\)?",   # expected format: "Option: B"
        r"Answer:\s*\(?([A-E])\)?",   # common alternate phrasing
        r"\(([A-E])\)",               # "(B)"
        r"\b([A-E])[\).:]",           # "B)" / "B." / "B:"
        r"\b([A-E])\b",               # bare standalone letter, last resort
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def save_sample_query(key, prompt, output):
    samples = {}
    if SAMPLE_QUERIES_FILE.exists():
        samples = json.loads(SAMPLE_QUERIES_FILE.read_text())
    samples[key] = {"prompt": prompt, "output": output}
    SAMPLE_QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_QUERIES_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False))


def log_eval_result(record):
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def generate_text(loader, model, tok_or_proc, config, prompt, temperature, thinking, max_tokens):
    kwargs = {} if thinking is None else {"enable_thinking": thinking}
    if loader == "mlx-vlm":
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(tok_or_proc, config, prompt, num_images=0, **kwargs)
        out = vlm_generate(model, tok_or_proc, formatted, max_tokens=max_tokens, temperature=temperature, verbose=False)
        return out if isinstance(out, str) else out.text
    else:
        from mlx_lm import generate as lm_generate
        from mlx_lm.sample_utils import make_sampler
        formatted = tok_or_proc.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, **kwargs
        )
        sampler = make_sampler(temp=temperature)
        return lm_generate(model, tok_or_proc, prompt=formatted, sampler=sampler, max_tokens=max_tokens, verbose=False)


def run_single_model(model_name, cfg, prompt_template, dataset_dirs, thinking):
    """Runs one model at ONE fixed thinking value, across all datasets.
    Timings/accuracy are computed and logged PER DATASET, not mushed together."""
    path = str(MODELS_DIR / model_name)

    if cfg["loader"] == "mlx-vlm":
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, tok_or_proc = load(path)
        config = load_config(path)
    else:
        from mlx_lm import load
        model, tok_or_proc = load(path)
        config = None

    tag = "na" if thinking is None else str(thinking).lower()
    sample_captured = False

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name.replace("_CLEANED", "")
        samples = json.loads((dataset_dir / "sampled.json").read_text())
        if TOP_N != -1:
            samples = samples[:TOP_N]

        out_dir = OUTPUT_DIR / dataset_name / model_name
        final_path = out_dir / f"predictions_thinking_{tag}.json"
        ckpt_path = out_dir / f"predictions_thinking_{tag}.checkpoint.jsonl"

        if final_path.exists():
            continue  # resumable: whole combo already done (already logged in a prior run)

        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        if ckpt_path.exists():
            results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
        start_idx = len(results)

        max_tokens = MAX_TOKENS_THINKING if thinking else MAX_TOKENS_ANSWER
        desc = f"{model_name} | {dataset_name} | thinking={tag}"
        failures, dataset_time = 0, 0.0

        with open(ckpt_path, "a") as ckpt_f:
            pbar = tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples), desc=desc)
            for idx in pbar:
                ex = samples[idx]
                prompt = format_prompt(prompt_template, ex)

                t0 = time.time()
                try:
                    text = generate_text(cfg["loader"], model, tok_or_proc, config, prompt,
                                          cfg["temperature"], thinking, max_tokens)
                except Exception as e:
                    text = f"[GENERATION ERROR] {e}"
                dataset_time += time.time() - t0

                if not sample_captured:
                    save_sample_query(f"{model_name}__thinking_{tag}", prompt, text)
                    sample_captured = True

                pred_letter = extract_letter(text)
                if pred_letter is None:
                    failures += 1
                pbar.set_postfix(failures=failures)

                record = {k: v for k, v in ex.items() if k != "metadata"}
                record["llm_output"] = text
                record["prediction"] = pred_letter
                record["evaluation_correct"] = pred_letter == ex["ground_truth"]
                results.append(record)
                ckpt_f.write(json.dumps(record) + "\n")
                ckpt_f.flush()

        final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        ckpt_path.unlink()

        # -------- per (model, dataset, thinking) summary: printed + logged immediately --------
        n_new = len(samples) - start_idx
        n_total = len(results)
        n_correct = sum(1 for r in results if r["evaluation_correct"])
        accuracy = n_correct / n_total if n_total else 0.0
        per_query = dataset_time / n_new if n_new else 0.0
        peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes

        print(f"[{model_name} | {dataset_name} | thinking={tag}] RAM={peak_ram_gb:.2f} GB "
              f"| total_time={dataset_time:.1f}s | queries={n_total} | per_query={per_query:.2f}s "
              f"| accuracy={accuracy:.2%} | failures={failures}", flush=True)

        log_eval_result({
            "model": model_name,
            "dataset": dataset_name,
            "thinking": tag,
            "temperature": cfg["temperature"],
            "ram_gb": round(peak_ram_gb, 2),
            "total_time_sec": round(dataset_time, 2),
            "per_query_time_sec": round(per_query, 4),
            "queries": n_total,
            "failures": failures,
            "accuracy": round(accuracy, 4),
        })


if __name__ == "__main__":
    models_cfg = yaml.safe_load(open(MODELS_YAML))
    prompt_template = PROMPT_FILE.read_text()
    dataset_dirs = sorted(d for d in SAMPLED_DIR.iterdir() if d.is_dir())

    for thinking in THINKING_ORDER:                  # false for ALL models first, then true for ALL models
        for model_name, cfg in models_cfg.items():
            if cfg["thinking"] is None:
                if thinking is True:
                    continue                          # unsupported models only run once, during the False pass
                effective_thinking = None
            else:
                effective_thinking = thinking
            run_single_model(model_name, cfg, prompt_template, dataset_dirs, effective_thinking)

    print("Done.")