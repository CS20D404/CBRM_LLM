import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "../../Models/Models"
SAMPLED_DIR = SCRIPT_DIR / "../../Datasets/Outputs/Datasets_SAMPLED"
CORPUS_DIR = SCRIPT_DIR / "../Inputs"
CORPUS_FILENAME = "corpus_all.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_LOCAL_PATH = MODELS_DIR / "all-mpnet-base-v2"
K = 3


def load_embedder():
    if EMBEDDING_MODEL_LOCAL_PATH.exists():
        return SentenceTransformer(str(EMBEDDING_MODEL_LOCAL_PATH))
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def embedding_text(case):
    return case["problem_statement"]


embedder = load_embedder()

for dataset_dir in sorted(p for p in SAMPLED_DIR.iterdir() if p.is_dir()):
    locale = dataset_dir.name.replace("_CLEANED", "")
    queries = json.loads((dataset_dir / "sampled.json").read_text())

    # The reference stores each corpus under locale/model-name/.
    # This uses the first model folder found for the locale.
    locale_corpus_dir = CORPUS_DIR / locale
    model_dirs = sorted(p for p in locale_corpus_dir.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"{locale}: no corpus model directory found")
        continue

    corpus_path = model_dirs[0] / CORPUS_FILENAME
    corpus = json.loads(corpus_path.read_text())

    corpus_vectors = embedder.encode(
        [embedding_text(case) for case in corpus],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    corpus_ids = [case["id"] for case in corpus]

    query_vectors = embedder.encode(
        [embedding_text(case) for case in queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    best = None
    for query, query_vector in zip(queries, query_vectors):
        similarities = corpus_vectors @ query_vector
        ranked = np.argsort(-similarities)

        top_3 = []
        for index in ranked:
            if corpus_ids[index] == query["id"]:
                continue
            top_3.append({
                "case": corpus[index],
                "similarity": float(similarities[index]),
            })
            if len(top_3) == K:
                break

        if len(top_3) == K:
            average_similarity = sum(x["similarity"] for x in top_3) / K
            if best is None or average_similarity > best["average_similarity"]:
                best = {
                    "locale": locale,
                    "average_similarity": average_similarity,
                    "query": query,
                    "top_3_retrieved_cases": top_3,
                }

    print(json.dumps(best, indent=2, ensure_ascii=False) if best else f"{locale}: no query had 3 retrieved cases")