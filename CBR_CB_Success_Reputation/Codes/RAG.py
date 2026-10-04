"""
source myenv/bin/activate
pip install mlx-lm mlx-vlm pyyaml tqdm psutil sentence-transformers numpy scipy
python3 CBR_CB_Success_Reputation/Codes/RAG.py success > CBR_CB_Success_Reputation/Codes/RAG.txt

Forgetting via consolidation
----------------------------
For every (model, dataset, fold):
  1. Build the retrieval pool from the TRAIN fold ("all" / "success" / "failure").
  2. Leave-one-out (LOO) pass over the pool: each pool case is a query, retrieved neighbours come from the
     pool minus the query itself. If the RAG answer is correct, every retrieved neighbour gets +1; otherwise -1.
  3. Consolidate: keep a pool case iff (it was NOT solved as a LOO query) OR (accumulated contribution > 0).
  4. Test queries (ALL test instances) are answered using ONLY the consolidated pool.
  5. Per fold we store: pool size, retained count, retained case ids (own file), train error (pre-consolidation
     LOO error) and test error.
"""

import os

# Force fully offline behaviour for anything that touches the HF hub (sentence-transformers).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import math
import re
import sys
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

# Retrieval-corpus configuration (applied to the TRAIN fold only; test queries are always ALL test instances):
#   "all"     -> every train instance is retrievable
#   "success" -> only train instances the model got RIGHT in zero-shot (evaluation_correct == True)
#   "failure" -> only train instances the model got WRONG in zero-shot (evaluation_correct == False)
# The selected pool is then consolidated (leave-one-out reward/penalty) before being used for the test fold.
VALID_CONFIGS = ("all", "success", "failure")
CORPUS_CONFIG = sys.argv[1] if len(sys.argv) > 1 else "all"
if CORPUS_CONFIG not in VALID_CONFIGS:
    raise SystemExit(f"CORPUS_CONFIG must be one of {VALID_CONFIGS}, got {CORPUS_CONFIG!r}")

SEED = 42                           # same seed => identical folds for every model AND every corpus config
TOP_N = -1                          # -1 => run on all instances, else first N per dataset (applied BEFORE splitting)
N_FOLDS = 5                         # 5 folds => every fold is an 80% train / 20% test split
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_YAML = SCRIPT_DIR / "../../Models/Codes/Models.yaml"
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
ZERO_SHOT_OUTPUT_DIR = SCRIPT_DIR / "../../Zero_Shot_Reasoning/Outputs"   # source of success/failure labels + paired t-test

RAG_PROMPT_FILE = SCRIPT_DIR / "RAG_Prompt.txt"
RAG_CASE_TEMPLATE_FILE = SCRIPT_DIR / "RAG_Case_Template.txt"

CV_ROOT = SCRIPT_DIR / "../Outputs" / "cross_val"
SPLITS_DIR = CV_ROOT / "Splits"                            # shared by every model and every corpus config
# Separate output dir so results never collide with the non-consolidated runs (which would look "already done").
OUTPUT_DIR = CV_ROOT / f"{CORPUS_CONFIG}_consolidated"
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
    The split depends ONLY on the dataset's instance ids, SEED, N_FOLDS and TOP_N -- never on the model
    or the corpus config -- and is persisted to disk so every model / config / re-run uses the same folds."""
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


# ------------------------------------------------- zero-shot baseline / pool ----

def load_baseline(dataset_name, model_name, tag):
    """Returns ({instance_id: evaluation_correct (bool)}, path). The dict is None if the file is missing."""
    path = ZERO_SHOT_OUTPUT_DIR / dataset_name / model_name / f"predictions_thinking_{tag}.json"
    if not path.exists():
        return None, path
    return {r["id"]: bool(r["evaluation_correct"]) for r in json.loads(path.read_text())}, path


def filter_train_ids(train_ids, baseline):
    """Restricts the retrieval corpus (train fold) according to CORPUS_CONFIG. Train instances with no
    zero-shot baseline entry are dropped from the 'success' / 'failure' pools."""
    if CORPUS_CONFIG == "all":
        return list(train_ids)
    want = (CORPUS_CONFIG == "success")
    return [i for i in train_ids if baseline.get(i) is want]


# ------------------------------------------------------------- retrieval ----

def _topk_rows(sims, corpus_ids, kk):
    out = []
    for row in sims:
        top = np.argpartition(-row, kk - 1)[:kk]
        top = top[np.argsort(-row[top])]
        out.append([(corpus_ids[i], float(row[i])) for i in top])
    return out


def retrieve_neighbors(query_embs, corpus_embs, corpus_ids, k):
    """Top-k corpus neighbors per query by cosine similarity (embeddings are pre-normalized, so
    dot product == cosine). Used for TEST queries: the corpus is a subset of the train fold, which is
    disjoint from the test fold, so no self-exclusion is needed. If the corpus has fewer than k items,
    all of them are returned."""
    if corpus_embs.shape[0] == 0:
        return [[] for _ in range(query_embs.shape[0])]
    sims = query_embs @ corpus_embs.T                      # (n_test, n_corpus)
    kk = min(k, corpus_embs.shape[0])
    return _topk_rows(sims, corpus_ids, kk)


def retrieve_neighbors_loo(corpus_embs, corpus_ids, k):
    """Leave-one-out retrieval WITHIN the corpus: for every corpus case, its top-k neighbours among all
    OTHER corpus cases (the case itself is explicitly excluded). Returns one list per corpus case, in the
    order of corpus_ids. Lists are empty when the corpus has fewer than 2 cases."""
    n = corpus_embs.shape[0]
    kk = min(k, n - 1)
    if kk <= 0:
        return [[] for _ in range(n)]
    sims = corpus_embs @ corpus_embs.T                     # (n, n)
    np.fill_diagonal(sims, -np.inf)                        # exclude self
    return _topk_rows(sims, corpus_ids, kk)


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


# ----------------------------------------------------------------- utils ----

def safe_round(x, ndigits):
    """Rounds, but passes through None/NaN as None (so they serialize as JSON null instead of NaN)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, ndigits)


def none_to_nan(x):
    return float("nan") if x is None else float(x)


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
    """Loads samples, builds the (model- and config-independent) folds and embeds every instance ONCE per
    dataset. Embeddings are reused across all models and folds."""
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


# ------------------------------------------------- consolidation (LOO) ----

def ids_file_path(out_dir, tag, fold):
    return out_dir / f"consolidated_ids_thinking_{tag}_cross_val_fold{fold}.json"


def write_ids_file(out_dir, tag, fold, loo_payload, test_error=None, n_test=None):
    """Persists the retained case ids (+ counts and errors) for one fold. Written right after consolidation
    (test_error=None) and rewritten once the test fold has been evaluated."""
    payload = {
        "fold": fold,
        "corpus_config": CORPUS_CONFIG,
        "n_pool_before_consolidation": loo_payload["n_pool"],
        "n_retained": loo_payload["n_retained"],
        "train_error": loo_payload["train_error"],      # pre-consolidation LOO error over the pool
        "test_error": test_error,                       # error on the test fold using the consolidated pool
        "n_test": n_test,
        "retained_ids": loo_payload["retained_ids"],
    }
    p = ids_file_path(out_dir, tag, fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_loo_consolidation(model_name, cfg, model, tok_or_proc, config, prompt_template, case_template,
                          ds, fold, pool_ids, thinking, tag, out_dir, sample_state):
    """Leave-one-out RAG over the retrieval pool, then forgetting.
      - every pool case is a query; neighbours come from the pool minus the query
      - query solved  -> +1 to each retrieved neighbour; query unsolved -> -1 to each retrieved neighbour
      - retained = cases that were NOT solved as a query OR have accumulated contribution > 0
    Resumable (checkpoint per fold). Returns a dict with retained_ids, train_error, counts, etc."""
    loo_path = out_dir / f"loo_thinking_{tag}_cross_val_fold{fold}.json"
    ckpt_path = out_dir / f"loo_thinking_{tag}_cross_val_fold{fold}.checkpoint.jsonl"

    if loo_path.exists():
        return json.loads(loo_path.read_text())          # resumable: consolidation already done

    n_pool = len(pool_ids)

    # Degenerate pool: LOO impossible (no neighbours). Keep everything, no train error.
    if n_pool < 2:
        payload = {"fold": fold, "n_pool": n_pool, "n_retained": n_pool, "loo_accuracy": None,
                   "train_error": None, "loo_failures": 0, "time_sec": 0.0,
                   "retained_ids": list(pool_ids), "contributions": {str(i): 0 for i in pool_ids},
                   "loo_results": []}
        loo_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    corpus_by_id = {i: ds["by_id"][i] for i in pool_ids}
    corpus_embs = ds["embs"][[ds["pos_by_id"][i] for i in pool_ids]]
    neighbors_per_query = retrieve_neighbors_loo(corpus_embs, pool_ids, RAG_K)

    results = []
    if ckpt_path.exists():
        results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
    start_idx = len(results)
    failures = sum(1 for r in results if r["prediction"] is None)
    loo_time = 0.0

    desc = f"{model_name} | {ds['name']} | LOO consolidation | RAG={CORPUS_CONFIG} | fold {fold + 1}/{N_FOLDS}"
    with open(ckpt_path, "a") as ckpt_f:
        pbar = tqdm(range(start_idx, n_pool), initial=start_idx, total=n_pool, desc=desc)
        for idx in pbar:
            qid = pool_ids[idx]
            ex = corpus_by_id[qid]
            neighbors = neighbors_per_query[idx]
            prompt = build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_by_id)

            t0 = time.time()
            try:
                text = generate_text(cfg["loader"], model, tok_or_proc, config, prompt,
                                     cfg["temperature"], thinking, MAX_TOKENS_ANSWER)
            except Exception as e:
                text = f"[GENERATION ERROR] {e}"
            loo_time += time.time() - t0

            if not sample_state["captured_loo"]:
                save_sample_query(f"{model_name}__thinking_{tag}__cross_val__loo", prompt, text)
                sample_state["captured_loo"] = True

            pred_letter = extract_letter(text)
            if pred_letter is None:
                failures += 1
            pbar.set_postfix(failures=failures)

            record = {"id": qid, "llm_output": text, "prediction": pred_letter,
                      "correct": pred_letter == ex["ground_truth"],
                      "neighbors": neighbors}               # [(pool_instance_id, similarity), ...]
            results.append(record)
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()

    # ---- contributions: +1 / -1 to every retrieved neighbour depending on whether the query was solved ----
    contrib = {i: 0 for i in pool_ids}
    solved = {}
    for r in results:
        solved[r["id"]] = bool(r["correct"])
        delta = 1 if r["correct"] else -1
        for neighbor_id, _sim in r["neighbors"]:
            contrib[neighbor_id] += delta

    # ---- forgetting: keep unsolved-as-query cases OR cases with positive accumulated contribution ----
    retained_ids = [i for i in pool_ids if (not solved[i]) or contrib[i] > 0]

    loo_acc = acc([int(r["correct"]) for r in results])
    payload = {
        "fold": fold,
        "n_pool": n_pool,
        "n_retained": len(retained_ids),
        "loo_accuracy": loo_acc,
        "train_error": 1.0 - loo_acc,                        # pre-consolidation LOO error
        "loo_failures": failures,
        "time_sec": round(loo_time, 2),
        "retained_ids": retained_ids,
        "contributions": {str(i): c for i, c in contrib.items()},
        "loo_results": results,
    }
    loo_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    ckpt_path.unlink()
    return payload


# ------------------------------------------------------------- one fold ----

def run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template, case_template,
             ds, split, baseline, thinking, tag, out_dir, sample_state):
    fold = split["fold"]
    fold_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.json"
    ckpt_path = out_dir / f"predictions_thinking_{tag}_cross_val_fold{fold}.checkpoint.jsonl"

    if fold_path.exists():
        return json.loads(fold_path.read_text())        # resumable: fold already done

    train_ids, test_ids = split["train_ids"], split["test_ids"]
    assert not (set(train_ids) & set(test_ids)), "train/test overlap!"

    # 1) retrieval pool (subset of TRAIN only), before forgetting
    pool_ids = filter_train_ids(train_ids, baseline)

    # 2) leave-one-out consolidation within the pool -> retained corpus
    loo = run_loo_consolidation(model_name, cfg, model, tok_or_proc, config, prompt_template, case_template,
                                ds, fold, pool_ids, thinking, tag, out_dir, sample_state)
    corpus_ids = loo["retained_ids"]
    write_ids_file(out_dir, tag, fold, loo)
    print(f"[{model_name} | {ds['name']} | fold {fold}] pool={loo['n_pool']} retained={loo['n_retained']} "
          f"train_error(LOO)={none_to_nan(loo['train_error']):.4f}", flush=True)

    if not corpus_ids:
        raise RuntimeError(f"Consolidated corpus is empty for {model_name}/{ds['name']}/fold {fold}.")

    # 3) test queries against the consolidated corpus ONLY
    corpus_by_id = {i: ds["by_id"][i] for i in corpus_ids}
    corpus_embs = ds["embs"][[ds["pos_by_id"][i] for i in corpus_ids]]
    test_samples = [ds["by_id"][i] for i in test_ids]   # queries: ALL test instances, regardless of config
    test_embs = ds["embs"][[ds["pos_by_id"][i] for i in test_ids]]

    neighbors_per_query = retrieve_neighbors(test_embs, corpus_embs, corpus_ids, RAG_K)

    results = []
    if ckpt_path.exists():
        results = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l]
    start_idx = len(results)
    failures = sum(1 for r in results if r["prediction"] is None)
    fold_time = 0.0

    desc = f"{model_name} | {ds['name']} | thinking={tag} | RAG={CORPUS_CONFIG} | fold {fold + 1}/{N_FOLDS}"
    with open(ckpt_path, "a") as ckpt_f:
        pbar = tqdm(range(start_idx, len(test_samples)), initial=start_idx, total=len(test_samples), desc=desc)
        for idx in pbar:
            ex = test_samples[idx]
            neighbors = neighbors_per_query[idx]
            prompt = build_rag_prompt(prompt_template, case_template, ex, neighbors, corpus_by_id)

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
            record["retrieval_metadata_cross_val"] = neighbors   # [(corpus_instance_id, similarity), ...]
            results.append(record)
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()

    test_acc = acc([int(r["evaluation_correct"]) for r in results])
    test_error = 1.0 - test_acc if results else float("nan")

    fold_payload = {
        "fold": fold,
        "corpus_config": CORPUS_CONFIG,
        "n_train": len(train_ids),
        "n_pool_before_consolidation": loo["n_pool"],
        "n_corpus": len(corpus_ids),                     # retained cases actually used for test retrieval
        "n_retained": loo["n_retained"],
        "n_test": len(test_ids),
        "train_error": loo["train_error"],               # pre-consolidation LOO error
        "test_error": safe_round(test_error, 6),
        "loo_time_sec": loo["time_sec"],
        "loo_failures": loo["loo_failures"],
        "time_sec": round(fold_time, 2),
        "failures": failures,
        "results": results,
    }
    fold_path.write_text(json.dumps(fold_payload, indent=2, ensure_ascii=False))
    write_ids_file(out_dir, tag, fold, loo, test_error=safe_round(test_error, 6), n_test=len(test_ids))
    ckpt_path.unlink()
    return fold_payload


# ------------------------------------------------------------ one model ----

def run_single_model(model_name, cfg, prompt_template, case_template, prepared, thinking):
    """Runs one model at ONE fixed thinking value across all datasets and all folds.
    Queries come from the TEST fold; retrieval is only over the consolidated (config-filtered) TRAIN fold."""
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
    sample_state = {"captured": False, "captured_loo": False}

    for ds in prepared:
        dataset_name = ds["name"]
        tagline = f"[{model_name} | {dataset_name} | thinking={tag} | RAG={CORPUS_CONFIG}+consolidation]"
        out_dir = OUTPUT_DIR / dataset_name / model_name
        final_path = out_dir / f"predictions_thinking_{tag}_cross_val.json"
        if final_path.exists():
            continue  # resumable: whole (model, dataset) already done and logged

        baseline, baseline_path = load_baseline(dataset_name, model_name, tag)
        if baseline is None:
            print(f"{tagline} WARNING: zero-shot baseline not found at {baseline_path}; skipping.", flush=True)
            continue

        pool_sizes = [len(filter_train_ids(s["train_ids"], baseline)) for s in ds["splits"]]
        if min(pool_sizes) == 0:
            print(f"{tagline} WARNING: empty '{CORPUS_CONFIG}' retrieval pool in at least one fold "
                  f"(pool sizes per fold: {pool_sizes}); skipping.", flush=True)
            continue
        if min(pool_sizes) < RAG_K:
            print(f"{tagline} WARNING: retrieval pool smaller than k={RAG_K} in some folds "
                  f"(pool sizes per fold: {pool_sizes}); fewer cases will be shown there.", flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        fold_payloads = [run_fold(model_name, cfg, model, tok_or_proc, config, prompt_template,
                                  case_template, ds, split, baseline, thinking, tag, out_dir, sample_state)
                         for split in ds["splits"]]

        # Out-of-fold predictions: every instance is tested exactly once across the folds.
        results = [r for fp in fold_payloads for r in fp["results"]]

        # -------- aggregate summary --------
        n_total = len(results)
        n_correct = sum(1 for r in results if r["evaluation_correct"])
        accuracy = n_correct / n_total if n_total else 0.0
        dataset_time = sum(fp["time_sec"] for fp in fold_payloads)               # test-time only
        loo_time = sum(fp["loo_time_sec"] for fp in fold_payloads)               # consolidation time
        failures = sum(fp["failures"] for fp in fold_payloads)
        loo_failures = sum(fp["loo_failures"] for fp in fold_payloads)
        per_query = dataset_time / n_total if n_total else 0.0
        peak_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes

        fold_accs = [acc([int(r["evaluation_correct"]) for r in fp["results"]]) for fp in fold_payloads]
        fold_acc_mean = float(np.mean(fold_accs))
        fold_acc_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else float("nan")

        # train / test error per fold (train = pre-consolidation LOO error over the pool)
        train_errors = [none_to_nan(fp["train_error"]) for fp in fold_payloads]
        test_errors = [none_to_nan(fp["test_error"]) for fp in fold_payloads]
        train_err_mean = float(np.nanmean(train_errors)) if not all(math.isnan(x) for x in train_errors) else float("nan")
        test_err_mean = float(np.nanmean(test_errors)) if not all(math.isnan(x) for x in test_errors) else float("nan")

        pool_before = [fp["n_pool_before_consolidation"] for fp in fold_payloads]
        retained = [fp["n_retained"] for fp in fold_payloads]

        top1_sims = [r["retrieval_metadata_cross_val"][0][1] for r in results if r["retrieval_metadata_cross_val"]]
        avg_top1_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0.0

        print(f"{tagline} RAM={peak_ram_gb:.2f} GB | test_time={dataset_time:.1f}s | loo_time={loo_time:.1f}s "
              f"| queries={n_total} | per_query={per_query:.2f}s | accuracy={accuracy:.2%} "
              f"| failures={failures} | loo_failures={loo_failures} | folds={len(fold_payloads)} "
              f"| k={RAG_K} | avg_top1_sim={avg_top1_sim:.4f}", flush=True)
        print(f"{tagline} pool_before={pool_before} | retained={retained}", flush=True)
        print(f"{tagline} train_error(LOO, pre-consolidation) per fold: "
              + ", ".join(f"{e:.2%}" for e in train_errors) + f" | mean={train_err_mean:.2%}", flush=True)
        print(f"{tagline} test_error per fold: "
              + ", ".join(f"{e:.2%}" for e in test_errors) + f" | mean={test_err_mean:.2%}", flush=True)
        print(f"{tagline} per-fold accuracy: "
              + ", ".join(f"{a:.2%}" for a in fold_accs)
              + f" | mean={fold_acc_mean:.2%} std={fold_acc_std:.4f}", flush=True)

        # -------- overall paired t-test (RAG vs zero-shot), paired by instance id --------
        zs_all, rag_all = [], []
        for record in results:
            baseline_correct = baseline.get(record["id"])
            if baseline_correct is None:
                continue  # no matching zero-shot baseline instance; excluded from this analysis
            record["zero_shot_baseline_correct"] = baseline_correct
            zs_all.append(int(baseline_correct))
            rag_all.append(int(record["evaluation_correct"]))

        zs_overall_acc, rag_overall_acc = acc(zs_all), acc(rag_all)
        t_overall, p_overall = paired_ttest(zs_all, rag_all)

        print(f"{tagline} OVERALL (n={len(zs_all)}): ZS_acc={zs_overall_acc:.2%} RAG_acc={rag_overall_acc:.2%} "
              f"| paired t-test t={t_overall:.4f} p={p_overall:.4g}", flush=True)

        zs_stats = {
            "n_baseline_matched_cross_val": len(zs_all),
            "zero_shot_accuracy_overall_cross_val": safe_round(zs_overall_acc, 4),
            "rag_accuracy_overall_cross_val": safe_round(rag_overall_acc, 4),
            "paired_ttest_t_overall_cross_val": safe_round(t_overall, 4),
            "paired_ttest_p_overall_cross_val": safe_round(p_overall, 6),
        }

        final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # -------- aggregate file with the retained case ids of every fold --------
        aggregate_ids_path = out_dir / f"consolidated_ids_thinking_{tag}_cross_val.json"
        aggregate_ids_path.write_text(json.dumps([
            {"fold": fp["fold"],
             "n_pool_before_consolidation": fp["n_pool_before_consolidation"],
             "n_retained": fp["n_retained"],
             "train_error": safe_round(none_to_nan(fp["train_error"]), 6),
             "test_error": safe_round(none_to_nan(fp["test_error"]), 6),
             "retained_ids": json.loads(ids_file_path(out_dir, tag, fp["fold"]).read_text())["retained_ids"]}
            for fp in fold_payloads
        ], indent=2, ensure_ascii=False))

        log_eval_result({
            "model": model_name,
            "dataset": dataset_name,
            "thinking": tag,
            "config": CORPUS_CONFIG,
            "consolidation": "loo_contribution",
            "temperature": cfg["temperature"],
            "ram_gb": round(peak_ram_gb, 2),
            "total_time_sec": round(dataset_time, 2),
            "loo_time_sec": round(loo_time, 2),
            "per_query_time_sec": round(per_query, 4),
            "queries": n_total,
            "failures": failures,
            "loo_failures": loo_failures,
            "accuracy_cross_val": round(accuracy, 4),
            "n_folds_cross_val": len(fold_payloads),
            "fold_accuracies_cross_val": [safe_round(a, 4) for a in fold_accs],
            "fold_accuracy_mean_cross_val": safe_round(fold_acc_mean, 4),
            "fold_accuracy_std_cross_val": safe_round(fold_acc_std, 4),
            "train_sizes_cross_val": [fp["n_train"] for fp in fold_payloads],
            "pool_sizes_before_consolidation_cross_val": pool_before,
            "retained_sizes_cross_val": retained,
            "corpus_sizes_cross_val": [fp["n_corpus"] for fp in fold_payloads],
            "test_sizes_cross_val": [fp["n_test"] for fp in fold_payloads],
            "train_errors_loo_pre_consolidation_cross_val": [safe_round(e, 4) for e in train_errors],
            "train_error_mean_cross_val": safe_round(train_err_mean, 4),
            "test_errors_cross_val": [safe_round(e, 4) for e in test_errors],
            "test_error_mean_cross_val": safe_round(test_err_mean, 4),
            "k": RAG_K,
            "avg_top1_similarity_cross_val": round(avg_top1_sim, 4),
            **zs_stats,
        })


if __name__ == "__main__":
    print(f"CORPUS_CONFIG={CORPUS_CONFIG} (+LOO consolidation) | N_FOLDS={N_FOLDS} | SEED={SEED} | k={RAG_K}", flush=True)
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