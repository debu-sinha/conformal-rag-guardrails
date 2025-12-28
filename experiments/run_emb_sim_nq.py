#!/usr/bin/env python3
"""
Run Embedding Similarity baseline on Natural Questions for ConformalDrift paper.
This generates the Emb-Sim NQ (in-dist) results for Table 2.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denominator
    adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return (max(0, centre - adjustment), min(1, centre + adjustment))


def load_natural_questions(n_samples: int = 400, seed: int = 42):
    """Load Natural Questions with synthetic hallucinations."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required")

    print(f"Loading Natural Questions (n={n_samples})...")
    np.random.seed(seed)

    dataset = load_dataset("nq_open", split="validation")

    faithful = []
    hallucinated = []

    for idx, item in enumerate(tqdm(dataset, total=min(n_samples, len(dataset)))):
        if len(faithful) >= n_samples // 2:
            break

        question = item['question']
        answers = item['answer']

        if not answers:
            continue

        gold_answer = answers[0]
        doc_text = f"The answer to '{question}' is {gold_answer}."

        # Faithful example
        faithful.append({
            'query': question,
            'document': doc_text,
            'response': gold_answer
        })

        # Synthetic hallucination
        wrong_answers = [
            "The information is not available.",
            "This cannot be determined.",
            "No answer found.",
            "Unknown",
            "N/A"
        ]
        hallucinated.append({
            'query': question,
            'document': doc_text,
            'response': np.random.choice(wrong_answers)
        })

    return faithful, hallucinated


def compute_embedding_similarity(model, doc, response):
    """Compute cosine similarity between document and response embeddings."""
    doc_emb = model.encode(doc, convert_to_numpy=True)
    resp_emb = model.encode(response, convert_to_numpy=True)

    sim = np.dot(doc_emb, resp_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(resp_emb))
    return float(sim)


def main():
    global SentenceTransformer
    if not ST_AVAILABLE:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers --quiet")
    from sentence_transformers import SentenceTransformer

    # Load model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load data
    faithful, hallucinated = load_natural_questions(n_samples=400)
    print(f"Loaded {len(faithful)} faithful, {len(hallucinated)} hallucinated samples")

    # Compute scores
    print("\nComputing embedding similarities...")
    scores_faithful = []
    scores_hallucinated = []

    print("Scoring faithful samples...")
    for s in tqdm(faithful):
        sim = compute_embedding_similarity(model, s['document'], s['response'])
        scores_faithful.append(1 - sim)  # Non-conformity: 1 - similarity

    print("Scoring hallucinated samples...")
    for s in tqdm(hallucinated):
        sim = compute_embedding_similarity(model, s['document'], s['response'])
        scores_hallucinated.append(1 - sim)

    scores_f = np.array(scores_faithful)
    scores_h = np.array(scores_hallucinated)

    print(f"\nFaithful scores: mean={scores_f.mean():.3f}, std={scores_f.std():.3f}")
    print(f"Hallucinated scores: mean={scores_h.mean():.3f}, std={scores_h.std():.3f}")

    # Conformal calibration for 95% coverage
    n = min(len(scores_f), len(scores_h))
    cal_size = n // 2

    # Use first half for calibration
    cal_h = scores_h[:cal_size]
    test_f = scores_f[cal_size:]
    test_h = scores_h[cal_size:]

    # Find threshold for 95% coverage on hallucinated
    # For non-conformity scores, we flag if score >= threshold
    alpha = 0.05
    target_coverage = 0.95

    # Threshold is the (1-alpha) quantile of calibration hallucinated scores
    threshold = np.percentile(cal_h, 100 * alpha)  # 5th percentile

    # Actually for non-conformity, we want scores >= threshold to be flagged
    # So threshold should be the alpha quantile (to catch 1-alpha of hallucinations)
    threshold = np.percentile(cal_h, 100 * (1 - target_coverage))

    # Test
    coverage = np.mean(test_h >= threshold)  # Hallucinations flagged
    fpr = np.mean(test_f >= threshold)  # Faithful incorrectly flagged

    fpr_ci = wilson_ci(fpr, len(test_f))
    cov_ci = wilson_ci(coverage, len(test_h))

    results = {
        "dataset": "nq",
        "method": "Emb-Sim (all-MiniLM-L6-v2)",
        "n_samples": n * 2,
        "n_calibration": cal_size,
        "n_test": n - cal_size,
        "target_coverage": target_coverage,
        "threshold": float(threshold),
        "coverage": float(coverage),
        "coverage_ci": [float(cov_ci[0]), float(cov_ci[1])],
        "fpr": float(fpr),
        "fpr_ci": [float(fpr_ci[0]), float(fpr_ci[1])],
        "score_stats": {
            "faithful_mean": float(scores_f.mean()),
            "faithful_std": float(scores_f.std()),
            "hallucinated_mean": float(scores_h.mean()),
            "hallucinated_std": float(scores_h.std())
        }
    }

    print(f"\n{'='*50}")
    print(f"RESULTS: Emb-Sim on Natural Questions")
    print(f"{'='*50}")
    print(f"  Target coverage: {target_coverage*100:.0f}%")
    print(f"  Effective coverage: {coverage*100:.1f}% [{cov_ci[0]*100:.1f}%, {cov_ci[1]*100:.1f}%]")
    print(f"  FPR: {fpr*100:.1f}% [{fpr_ci[0]*100:.1f}%, {fpr_ci[1]*100:.1f}%]")
    print(f"{'='*50}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "emb_sim_nq_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
