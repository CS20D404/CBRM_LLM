"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil sentence-transformers numpy scipy
python3 CBR_CB_All_Cross_Val/Codes/RAG_CrossVal.py > CBR_CB_All_Cross_Val/Codes/RAG_CrossVal.txt
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
TOP_N = -1                          # -1 => run on all instances, else first N per dataset (applied BEFORE splitting)
N_FOLDS = 5                         # 5 folds => every fold is an 80% train / 20% test split
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
ZERO_SHOT_OUTPUT_DIR = SCRIPT_DIR / "../../Zero_Shot_Reasoning/Outputs"   # baseline used to partition + paired t-test

RAG_PROMPT_FILE = SCRIPT_DIR / "RAG_Prompt.txt"
RAG_CASE_TEMPLATE_FILE = SCRIPT_DIR / "RAG_Case_Template.txt"

CONFIG_NAME = "cross_val"                                  # retrieval corpus = TRAIN fold only
OUTPUT_DIR = SCRIPT_DIR / "../Outputs" / CONFIG_NAME
SPLITS_DIR = OUTPUT_DIR / "Splits"                         # model-independent splits, shared by every model
SAMPLE_QUERIES_FILE = OUTPUT_DIR / "Sample_LLM_Queries_cross_val.json"
EVAL_RESULTS_FILE = OUTPUT_DIR / "RAG_Evaluation_Results_cross_val.jsonl"

THINKING_ORDER = [False]            # RAG is only ever run with thinking=false
MAX_TOKENS_ANSWER = 20

RAG_K = 3                                                  # number of retrieved neighbors per query
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_LOCAL_PATH = MODELS_DIR / "all-mpnet-base-v2"   # preferred local snapshot, if present
EMBEDDING_BATCH_SIZE = 64

# ==================================================

np.random.seed(SEED)


def load_embedder():
    """Loads the embedding model strictly offline."""
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


# ---------------------------------------------------------------- splits ----

def get_splits(dataset_name, samples):
    """Returns a list of N_FOLDS dicts {"fold": f, "train_ids": [...], "test_ids": [...]}.
    The split depends ONLY on the dataset's instance ids, SEED, N_FOLDS and TOP_N -- never on the model --
    and is persisted to disk so every model (and every re-run) uses exactly the same folds."""
    ids = [ex["id"] for ex in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset_name}: instance ids are not unique; cannot build a reliable split.")

    path = SPLITS_DIR / dataset_name / f"splits_cross_val_k{N_FOLDS}_seed{SEED}_top{TOP_N}.json"
    if path.exists():
        splits = json.loads(path.read_text())
        covered = sorted(i for s in splits for i in s["test_ids"])
        if covered != sorted(ids):
            raise ValueError(f"{path} does not match the current dataset ids; delete it to regenerate.")
        return splits

    rng = np.random.RandomState(SEED)
    sorted_ids = sorted(ids, key=str)                     # order-independent of file ordering
    perm = rng.permutation(len(sorted_ids))
    fold_indices = np.array_split(perm, N_FOLDS)
    splits = []
    for f, test_idx in enumerate(fold_indices):
        test_set = {sorted_ids[i] for i in test_idx}
        splits.append({
            "fold": f,
            "train_ids": [i for i in sorted_ids if i not in test_set],
            "test_ids": [i for i in sorted_ids if i in test_set],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    return splits


# ------------------------------------------------------------- retrieval ----

def retrieve_neighbors(query_embs, train_embs, train_ids, k):
    """Top-k TRAIN neighbors per query by cosine similarity (embeddings are pre-normalized, so
    dot product == cosine). Train and test are disjoint by construction, so no self-exclusion is needed."""
    if train_embs.shape[0] == 0:
        return [[] for _ in range(query_embs.shape[0])]
    sims = query_embs @ train_embs.T                       # (n_test, n_train)
    kk = min(k, train_embs.shape[0])
    all_neighbors = []
    for row in sims:
        top = np.argpartition(-row, kk - 1)[:kk]
        top = top[np.argsort(-row[top])]
        all_neighbors.append([(train_ids[i], float(row[i])) for i in top])
    return all_neighbors


def build_rag_prompt(prompt_template, case_template, ex, neighbors, train_by_id):
    options = "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
    cases = []
    for i, (neighbor_id, _sim) in enumerate(neighbors, start=1):
        neighbor = train_by_id[neighbor_id]
        ground_truth_text = neighbor["options"][neighbor["ground_truth"]]
        cases.append(case_template.format(i=i, problem_statement=neighbor["problem_statement"],
                                          ground_truth_text=ground_truth_text))
    similar_cases = "\n".join(cases)
    return prompt_template.format(similar_cases=similar_cases,
                                  problem_statement=ex["problem_statement"], options=options)


# ----------------------------------------------------------------- utils ----

def safe_round(x, ndigits):
    """Rounds, but passes through None/NaN as None (so they serialize as JSON null instead of NaN)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, ndigits)


def paired_ttest(zero_shot_correct, rag_correct):
    """Paired t-test between per-instance zero-shot and RAG correctness (0/1 lists).
    Returns (t_stat, p_value); both NaN when <2 pairs or when differences have zero variance."""
    if len(zero_shot_correct) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = ttest_rel(np.asarray(rag_correct, dtype=float), np.asarray(zero_shot_correct, dtype=float))
    return float(t_stat), float(p_value)


def extract_letter(text):
    # Ordered from most- to least-specific. Within each pattern, the LAST match wins.
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


def acc(bits):
    return sum(bits) / len(bits) if bits else float("nan")


# ------------------------------------------------------- dataset prepping ----

def prepare_datasets(dataset_dirs, embedder):
    """Loads samples, builds the (model-independent) folds and embeds every instance ONCE per dataset.
    Embeddings are reused across all models and folds."""
    prepared = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name.replace("_CLEANED", "")
        samples = json.loads((dataset_dir / "sampled.json").read_text())
        if TOP_N != -1:
            samples = samples[:TOP_N]
        splits = get_splits(dataset_name, samples)
        embs = encode(embedder, [embedding_text(ex) for ex in samples])
        by_id = {ex["id"]: ex for ex in samples}
        pos_by_id = {ex["id"]: i for i, ex in enumerate(samples)}
        prepared.append({"name": dataset_name, "samples": samples, "splits": splits,
                         "embs": embs, "by_id": by_id, "pos_by_id": pos_by_id})
    return prepared


# ------------------------------------------------------------- one fold ----

def run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template, case_template,
             ds, split, thinking, tag, out_dir, sample_state):
    fold = split["fold"]
    fold_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.json"
    ckpt_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.checkpoint.jsonl"

    if fold_path.exists():
        return json.loads(fold_path.read_text())        # resumable: fold already done

    train_ids, test_ids = split["train_ids"], split["test_ids"]
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
    train_by_id = {i: ds["by_id"][i] for i in train_ids}
    train_embs = ds["embs"][[ds["pos_by_id"][i] for i in train_ids]]
    test_samples = [ds["by_id"][i] for i in test_ids]
    test_embs = ds["embs"][[ds["pos_by_id"][i] for i in test_ids]]

    neighbors_per_query = retrieve_neighbors(test_embs, train_embs, train_ids, RAG_K)

    results, prior_time, prior_failures = [], 0.0, 0
    if ckpt_path.exists():
        results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
    start_idx = len(results)
    failures = sum(1 for r in results if r["prediction"] is None)
    fold_time = 0.0

    desc = f"{model_name} | {ds['name']} | thinking={tag} | fold {fold + 1}/{N_FOLDS}"
    with open(ckpt_path, "a") as ckpt_f:
        pbar = tqdm(range(start_idx, len(test_samples)), initial=start_idx, total=len(test_samples), desc=desc)
        for idx in pbar:
            ex = test_samples[idx]
            neighbors = neighbors_per_query[idx]
            prompt = build_rag_prompt(prompt_template, case_template, ex, neighbors, train_by_id)

            t0 = time.time()
            try:
                text = generate_text(cfg["loader"], model, tok_or_proc, config, prompt,
                                     cfg["temperature"], thinking, MAX_TOKENS_ANSWER)
            except Exception as e:
                text = f"[GENERATION ERROR] {e}"
            fold_time += time.time() - t0

            if not sample_state["captured"]:
                save_sample_query(f"{model_name}__thinking_{tag}__cross_val", prompt, text)
                sample_state["captured"] = True

            pred_letter = extract_letter(text)
            if pred_letter is None:
                failures += 1
            pbar.set_postfix(failures=failures)

            record = {k: v for k, v in ex.items() if k != "metadata"}
            record["llm_output"] = text
            record["prediction"] = pred_letter
            record["evaluation_correct"] = pred_letter == ex["ground_truth"]
            record["fold_cross_val"] = fold
            record["retrieval_metadata_cross_val"] = neighbors   # [(train_instance_id, similarity), ...]
            results.append(record)
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()

    fold_payload = {
        "fold": fold,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "time_sec": round(fold_time, 2),
        "failures": failures,
        "results": results,
    }
    fold_path.write_text(json.dumps(fold_payload, indent=2, ensure_ascii=False))
    ckpt_path.unlink()
    return fold_payload


# ------------------------------------------------------------ one model ----

def run_single_model(model_name, cfg, prompt_template, case_template, prepared, thinking):
    """Runs one model at ONE fixed thinking value across all datasets and all folds.
    Queries come from the TEST fold; retrieval is only over the TRAIN fold."""
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
    sample_state = {"captured": False}

    for ds in prepared:
        dataset_name = ds["name"]
        out_dir = OUTPUT_DIR / dataset_name / model_name
        final_path = out_dir / f"predictions_thinking_{tag}_cross_val.json"
        if final_path.exists():
            continue  # resumable: whole (model, dataset) already done and logged
        out_dir.mkdir(parents=True, exist_ok=True)

        fold_payloads = [run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template,
                                  case_template, ds, split, thinking, tag, out_dir, sample_state)
                         for split in ds["splits"]]

        # Out-of-fold predictions: every instance is tested exactly once across the folds.
        results = [r for fp in fold_payloads for r in fp["results"]]
        final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # -------- aggregate summary --------
        n_total = len(results)
        n_correct = sum(1 for r in results if r["evaluation_correct"])
        accuracy = n_correct / n_total if n_total else 0.0
        dataset_time = sum(fp["time_sec"] for fp in fold_payloads)
        failures = sum(fp["failures"] for fp in fold_payloads)
        per_query = dataset_time / n_total if n_total else 0.0
        peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes

        fold_accs = [acc([int(r["evaluation_correct"]) for r in fp["results"]]) for fp in fold_payloads]
        fold_acc_mean = float(np.mean(fold_accs))
        fold_acc_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else float("nan")

        top1_sims = [r["retrieval_metadata_cross_val"][0][1] for r in results if r["retrieval_metadata_cross_val"]]
        avg_top1_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0.0

        tagline = f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CONFIG_NAME}]"
        print(f"{tagline} RAM={peak_ram_gb:.2f} GB | total_time={dataset_time:.1f}s | queries={n_total} "
              f"| per_query={per_query:.2f}s | accuracy={accuracy:.2%} | failures={failures} "
              f"| folds={len(fold_payloads)} | k={RAG_K} | avg_top1_sim={avg_top1_sim:.4f}", flush=True)
        print(f"{tagline} per-fold accuracy: "
              + ", ".join(f"{a:.2%}" for a in fold_accs)
              + f" | mean={fold_acc_mean:.2%} std={fold_acc_std:.4f}", flush=True)

        # -------- partition by ZERO-SHOT outcome + paired t-test (RAG vs zero-shot), per instance --------
        baseline_path = ZERO_SHOT_OUTPUT_DIR / dataset_name / model_name / f"predictions_thinking_{tag}.json"
        zs_stats = {}
        if baseline_path.exists():
            baseline_lookup = {r["problem_statement"]: r["evaluation_correct"]
                               for r in json.loads(baseline_path.read_text())}

            zs_all, rag_all = [], []
            zs_success, rag_success = [], []
            zs_failure, rag_failure = [], []
            for record in results:
                baseline_correct = baseline_lookup.get(record["problem_statement"])
                if baseline_correct is None:
                    continue  # no matching zero-shot baseline instance; excluded from this analysis
                record["zero_shot_baseline_correct"] = baseline_correct
                zs_bit, rag_bit = int(baseline_correct), int(record["evaluation_correct"])
                zs_all.append(zs_bit)
                rag_all.append(rag_bit)
                (zs_success if baseline_correct else zs_failure).append(zs_bit)
                (rag_success if baseline_correct else rag_failure).append(rag_bit)

            zs_success_acc, rag_success_acc = acc(zs_success), acc(rag_success)
            zs_failure_acc, rag_failure_acc = acc(zs_failure), acc(rag_failure)
            zs_overall_acc, rag_overall_acc = acc(zs_all), acc(rag_all)

            t_success, p_success = paired_ttest(zs_success, rag_success)
            t_failure, p_failure = paired_ttest(zs_failure, rag_failure)
            t_overall, p_overall = paired_ttest(zs_all, rag_all)

            print(f"{tagline} Zero-shot SUCCESS group (n={len(zs_success)}): ZS_acc={zs_success_acc:.2%} "
                  f"RAG_acc={rag_success_acc:.2%} | paired t-test t={t_success:.4f} p={p_success:.4g}", flush=True)
            print(f"{tagline} Zero-shot FAILURE group (n={len(zs_failure)}): ZS_acc={zs_failure_acc:.2%} "
                  f"RAG_acc={rag_failure_acc:.2%} | paired t-test t={t_failure:.4f} p={p_failure:.4g}", flush=True)
            print(f"{tagline} OVERALL (n={len(zs_all)}): ZS_acc={zs_overall_acc:.2%} RAG_acc={rag_overall_acc:.2%} "
                  f"| paired t-test t={t_overall:.4f} p={p_overall:.4g}", flush=True)

            # re-write the combined file so it includes zero_shot_baseline_correct
            final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

            zs_stats = {
                "n_baseline_matched_cross_val": len(zs_all),
                "zero_shot_accuracy_overall_cross_val": safe_round(zs_overall_acc, 4),
                "rag_accuracy_overall_cross_val": safe_round(rag_overall_acc, 4),
                "paired_ttest_t_overall_cross_val": safe_round(t_overall, 4),
                "paired_ttest_p_overall_cross_val": safe_round(p_overall, 6),
                "n_zero_shot_success_cross_val": len(zs_success),
                "zero_shot_accuracy_success_group_cross_val": safe_round(zs_success_acc, 4),
                "rag_accuracy_success_group_cross_val": safe_round(rag_success_acc, 4),
                "paired_ttest_t_success_group_cross_val": safe_round(t_success, 4),
                "paired_ttest_p_success_group_cross_val": safe_round(p_success, 6),
                "n_zero_shot_failure_cross_val": len(zs_failure),
                "zero_shot_accuracy_failure_group_cross_val": safe_round(zs_failure_acc, 4),
                "rag_accuracy_failure_group_cross_val": safe_round(rag_failure_acc, 4),
                "paired_ttest_t_failure_group_cross_val": safe_round(t_failure, 4),
                "paired_ttest_p_failure_group_cross_val": safe_round(p_failure, 6),
            }
        else:
            print(f"{tagline} WARNING: no zero-shot baseline found at {baseline_path}; "
                  f"skipping partitioned analysis.", flush=True)

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
            "accuracy_cross_val": round(accuracy, 4),
            "n_folds_cross_val": len(fold_payloads),
            "fold_accuracies_cross_val": [safe_round(a, 4) for a in fold_accs],
            "fold_accuracy_mean_cross_val": safe_round(fold_acc_mean, 4),
            "fold_accuracy_std_cross_val": safe_round(fold_acc_std, 4),
            "train_sizes_cross_val": [fp["n_train"] for fp in fold_payloads],
            "test_sizes_cross_val": [fp["n_test"] for fp in fold_payloads],
            "k": RAG_K,
            "avg_top1_similarity_cross_val": round(avg_top1_sim, 4),
            **zs_stats,
        })


if __name__ == "__main__":
    models_cfg = yaml.safe_load(open(MODELS_YAML))
    prompt_template = RAG_PROMPT_FILE.read_text()
    case_template = RAG_CASE_TEMPLATE_FILE.read_text()
    dataset_dirs = sorted(d for d in SAMPLED_DIR.iterdir() if d.is_dir())
    embedder = load_embedder()

    # Folds + embeddings are built once, up front, and shared by every model.
    prepared = prepare_datasets(dataset_dirs, embedder)

    for thinking in THINKING_ORDER:                  # RAG only ever runs with thinking=false
        for model_name, cfg in models_cfg.items():
            effective_thinking = None if cfg["thinking"] is None else thinking
            run_single_model(model_name, cfg, prompt_template, case_template, prepared, effective_thinking)

    print("Done.")