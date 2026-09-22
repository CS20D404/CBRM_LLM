"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil sentence-transformers numpy scipy
python3 CBR_CB_Success/Codes/RAG.py > CBR_CB_Success/Codes/RAG.txt
"""

import os

# Force fully offline behaviour for anything that touches the HF hub (sentence-transformers).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import math
import re
import time
import resource
import warnings
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer

# ==================================================

SEED = 42
TOP_N = -1                          # -1 => run on all instances, else first N per dataset
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
ZERO_SHOT_OUTPUT_DIR = SCRIPT_DIR / "../../Zero_Shot_Reasoning/Outputs"   # baseline used to partition + paired t-test

RAG_PROMPT_FILE = SCRIPT_DIR / "RAG_Prompt.txt"
RAG_CASE_TEMPLATE_FILE = SCRIPT_DIR / "RAG_Case_Template.txt"

CONFIG_NAME = "success"                                    # config 3: retrieval corpus = success instances only
CORPUS_DIR = SCRIPT_DIR / "../Inputs"
CORPUS_FILENAME = f"corpus_{CONFIG_NAME}.json"

OUTPUT_DIR = SCRIPT_DIR / "../Outputs" / CONFIG_NAME
SAMPLE_QUERIES_FILE = OUTPUT_DIR / "Sample_LLM_Queries.json"
EVAL_RESULTS_FILE = OUTPUT_DIR / "RAG_Evaluation_Results.jsonl"

THINKING_ORDER = [False]            # RAG is only ever run with thinking=false
MAX_TOKENS_ANSWER = 20

RAG_K = 3                                                  # number of retrieved neighbors per query
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_LOCAL_PATH = MODELS_DIR / "all-mpnet-base-v2"   # preferred local snapshot, if present
EMBEDDING_BATCH_SIZE = 64

# ==================================================

np.random.seed(SEED)


def load_embedder():
    """Loads the embedding model strictly offline. Requires either a local snapshot at
    EMBEDDING_MODEL_LOCAL_PATH, or an already-populated local HF cache for EMBEDDING_MODEL_NAME."""
    if EMBEDDING_MODEL_LOCAL_PATH.exists():
        return SentenceTransformer(str(EMBEDDING_MODEL_LOCAL_PATH))
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def embedding_text(ex):
    return ex["problem_statement"]


def encode(embedder, texts):
    if not texts:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    return embedder.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)


def load_corpus(dataset_name, model_name):
    path = CORPUS_DIR / dataset_name / model_name / CORPUS_FILENAME
    return json.loads(path.read_text())


def retrieve_neighbors(query_embs, query_ids, corpus_embs, corpus_ids, k):
    """Top-k corpus neighbors per query by cosine similarity (embeddings are pre-normalized,
    so dot product == cosine similarity). Self is excluded by matching the instance "id" field,
    not by embedding similarity or text equality (two distinct instances could otherwise share
    identical problem_statement text)."""
    all_neighbors = []
    for q_emb, q_id in tqdm(list(zip(query_embs, query_ids)), desc="Retrieving neighbors", leave=False):
        if corpus_embs.shape[0] == 0:
            all_neighbors.append([])
            continue
        sims = corpus_embs @ q_emb
        order = np.argsort(-sims)
        neighbors = []
        for idx in order:
            if corpus_ids[idx] == q_id:
                continue  # never retrieve self
            neighbors.append((corpus_ids[idx], float(sims[idx])))
            if len(neighbors) == k:
                break
        all_neighbors.append(neighbors)
    return all_neighbors


def build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_by_id):
    options = "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
    cases = []
    for i, (neighbor_id, _sim) in enumerate(neighbors, start=1):
        neighbor = corpus_by_id[neighbor_id]
        ground_truth_text = neighbor["options"][neighbor["ground_truth"]]
        cases.append(case_template.format(i=i, problem_statement=neighbor["problem_statement"],
                                           ground_truth_text=ground_truth_text))
    similar_cases = "\n".join(cases)
    return prompt_template.format(similar_cases=similar_cases,
                                   problem_statement=ex["problem_statement"], options=options)


def safe_round(x, ndigits):
    """Rounds, but passes through None/NaN as None (so they serialize as JSON null instead of NaN)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, ndigits)


def paired_ttest(zero_shot_correct, rag_correct):
    """Paired t-test between per-instance zero-shot and RAG correctness (0/1 lists), paired by
    instance. Returns (t_stat, p_value); both NaN when there are fewer than 2 paired instances
    or when the per-instance differences have zero variance (e.g. RAG agrees with zero-shot on
    every instance in the group)."""
    if len(zero_shot_correct) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = ttest_rel(np.asarray(rag_correct, dtype=float), np.asarray(zero_shot_correct, dtype=float))
    return float(t_stat), float(p_value)


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


def run_single_model(model_name, cfg, prompt_template, case_template, embedder, dataset_dirs, thinking):
    """Runs one model at ONE fixed thinking value, across all datasets, with RAG retrieval
    from that model's own CONFIG_NAME corpus. Timings/accuracy are computed and logged
    PER DATASET, not mushed together."""
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

        corpus = load_corpus(dataset_name, model_name)
        corpus_ids = [r["id"] for r in corpus]
        corpus_by_id = {r["id"]: r for r in corpus}
        corpus_texts = [embedding_text(r) for r in corpus]
        corpus_embs = encode(embedder, corpus_texts)

        query_ids = [ex["id"] for ex in samples]
        query_texts = [embedding_text(ex) for ex in samples]
        query_embs = encode(embedder, query_texts)

        neighbors_per_query = retrieve_neighbors(query_embs, query_ids, corpus_embs, corpus_ids, RAG_K)

        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        if ckpt_path.exists():
            results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
        start_idx = len(results)

        max_tokens = MAX_TOKENS_ANSWER  # thinking is always False/None for RAG, never True
        desc = f"{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}"
        failures, dataset_time = 0, 0.0

        with open(ckpt_path, "a") as ckpt_f:
            pbar = tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples), desc=desc)
            for idx in pbar:
                ex = samples[idx]
                neighbors = neighbors_per_query[idx]
                prompt = build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_by_id)

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
                record["retrieval_metadata"] = neighbors  # [(neighbor_instance_id, similarity), ...] over the CONFIG_NAME corpus
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

        top1_sims = [r["retrieval_metadata"][0][1] for r in results if r["retrieval_metadata"]]
        avg_top1_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0.0

        print(f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}] RAM={peak_ram_gb:.2f} GB "
              f"| total_time={dataset_time:.1f}s | queries={n_total} | per_query={per_query:.2f}s "
              f"| accuracy={accuracy:.2%} | failures={failures} | corpus_size={len(corpus)} "
              f"| k={RAG_K} | avg_top1_sim={avg_top1_sim:.4f}", flush=True)

        # -------- partition by ZERO-SHOT outcome + paired t-test (RAG vs zero-shot), per instance --------
        baseline_path = ZERO_SHOT_OUTPUT_DIR / dataset_name / model_name / f"predictions_thinking_{tag}.json"
        zs_stats = {}
        if baseline_path.exists():
            baseline_lookup = {r["problem_statement"]: r["evaluation_correct"]
                                for r in json.loads(baseline_path.read_text())}

            zs_all, rag_all = [], []
            zs_success, rag_success = [], []
            zs_failure, rag_failure = [], []
            for ex, record in zip(samples, results):
                baseline_correct = baseline_lookup.get(ex["problem_statement"])
                if baseline_correct is None:
                    continue  # no matching zero-shot baseline instance; excluded from this analysis
                record["zero_shot_baseline_correct"] = baseline_correct
                zs_bit, rag_bit = int(baseline_correct), int(record["evaluation_correct"])
                zs_all.append(zs_bit)
                rag_all.append(rag_bit)
                (zs_success if baseline_correct else zs_failure).append(zs_bit)
                (rag_success if baseline_correct else rag_failure).append(rag_bit)

            zs_success_acc = sum(zs_success) / len(zs_success) if zs_success else float("nan")
            rag_success_acc = sum(rag_success) / len(rag_success) if rag_success else float("nan")
            zs_failure_acc = sum(zs_failure) / len(zs_failure) if zs_failure else float("nan")
            rag_failure_acc = sum(rag_failure) / len(rag_failure) if rag_failure else float("nan")
            zs_overall_acc = sum(zs_all) / len(zs_all) if zs_all else float("nan")
            rag_overall_acc = sum(rag_all) / len(rag_all) if rag_all else float("nan")

            t_success, p_success = paired_ttest(zs_success, rag_success)
            t_failure, p_failure = paired_ttest(zs_failure, rag_failure)
            t_overall, p_overall = paired_ttest(zs_all, rag_all)

            print(f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}] "
                  f"Zero-shot SUCCESS group (n={len(zs_success)}): ZS_acc={zs_success_acc:.2%} "
                  f"RAG_acc={rag_success_acc:.2%} | paired t-test t={t_success:.4f} p={p_success:.4g}",
                  flush=True)
            print(f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}] "
                  f"Zero-shot FAILURE group (n={len(zs_failure)}): ZS_acc={zs_failure_acc:.2%} "
                  f"RAG_acc={rag_failure_acc:.2%} | paired t-test t={t_failure:.4f} p={p_failure:.4g}",
                  flush=True)
            print(f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}] "
                  f"OVERALL (n={len(zs_all)}): ZS_acc={zs_overall_acc:.2%} RAG_acc={rag_overall_acc:.2%} "
                  f"| paired t-test t={t_overall:.4f} p={p_overall:.4g}", flush=True)

            zs_stats = {
                "n_baseline_matched": len(zs_all),
                "zero_shot_accuracy_overall": safe_round(zs_overall_acc, 4),
                "rag_accuracy_overall": safe_round(rag_overall_acc, 4),
                "paired_ttest_t_overall": safe_round(t_overall, 4),
                "paired_ttest_p_overall": safe_round(p_overall, 6),
                "n_zero_shot_success": len(zs_success),
                "zero_shot_accuracy_success_group": safe_round(zs_success_acc, 4),
                "rag_accuracy_success_group": safe_round(rag_success_acc, 4),
                "paired_ttest_t_success_group": safe_round(t_success, 4),
                "paired_ttest_p_success_group": safe_round(p_success, 6),
                "n_zero_shot_failure": len(zs_failure),
                "zero_shot_accuracy_failure_group": safe_round(zs_failure_acc, 4),
                "rag_accuracy_failure_group": safe_round(rag_failure_acc, 4),
                "paired_ttest_t_failure_group": safe_round(t_failure, 4),
                "paired_ttest_p_failure_group": safe_round(p_failure, 6),
            }
        else:
            print(f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}] "
                  f"WARNING: no zero-shot baseline found at {baseline_path}; skipping partitioned analysis.",
                  flush=True)

        log_eval_result({
            "model": model_name,
            "dataset": dataset_name,
            "thinking": tag,
            "config": CONFIG_NAME,
            "temperature": cfg["temperature"],
            "ram_gb": round(peak_ram_gb, 2),
            "total_time_sec": round(dataset_time, 2),
            "per_query_time_sec": round(per_query, 4),
            "queries": n_total,
            "failures": failures,
            "accuracy": round(accuracy, 4),
            "corpus_size": len(corpus),
            "k": RAG_K,
            "avg_top1_similarity": round(avg_top1_sim, 4),
            **zs_stats,
        })


if __name__ == "__main__":
    models_cfg = yaml.safe_load(open(MODELS_YAML))
    prompt_template = RAG_PROMPT_FILE.read_text()
    case_template = RAG_CASE_TEMPLATE_FILE.read_text()
    dataset_dirs = sorted(d for d in SAMPLED_DIR.iterdir() if d.is_dir())
    embedder = load_embedder()

    for thinking in THINKING_ORDER:                  # RAG only ever runs with thinking=false
        for model_name, cfg in models_cfg.items():
            effective_thinking = None if cfg["thinking"] is None else thinking
            run_single_model(model_name, cfg, prompt_template, case_template, embedder, dataset_dirs, effective_thinking)

    print("Done.")