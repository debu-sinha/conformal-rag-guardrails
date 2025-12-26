#!/usr/bin/env python3
"""
Run SOTA baseline experiments for Conformal RAG Guardrails.
Tests: OpenAI text-embedding-3-large, BART-large-MNLI
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Check for dependencies
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denominator
    adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return (max(0, centre - adjustment), min(1, centre + adjustment))


def load_halueval_samples(n_samples=200):
    """Load HaluEval samples."""
    if not HF_AVAILABLE:
        raise ImportError("datasets required")

    print("Loading HaluEval dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    faithful = [s for s in dataset if s['hallucination'] == 'no']
    hallucinated = [s for s in dataset if s['hallucination'] == 'yes']

    np.random.seed(42)
    n = min(n_samples, len(faithful), len(hallucinated))
    f_idx = np.random.choice(len(faithful), n, replace=False)
    h_idx = np.random.choice(len(hallucinated), n, replace=False)

    return (
        [faithful[i] for i in f_idx],
        [hallucinated[i] for i in h_idx]
    )


def run_openai_embeddings(faithful, hallucinated):
    """Test OpenAI text-embedding-3-large."""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available, skipping")
        return None

    client = openai.OpenAI()

    def get_embedding(text):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nRunning OpenAI text-embedding-3-large baseline...")
    scores_faithful = []
    scores_hallucinated = []

    # Score faithful samples
    print("Scoring faithful samples...")
    for s in tqdm(faithful):
        try:
            ctx_emb = get_embedding(s['knowledge'])
            ans_emb = get_embedding(s['answer'])
            sim = cosine_sim(ctx_emb, ans_emb)
            scores_faithful.append(1 - sim)  # Non-conformity: 1 - similarity
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Score hallucinated samples
    print("Scoring hallucinated samples...")
    for s in tqdm(hallucinated):
        try:
            ctx_emb = get_embedding(s['knowledge'])
            ans_emb = get_embedding(s['answer'])
            sim = cosine_sim(ctx_emb, ans_emb)
            scores_hallucinated.append(1 - sim)
        except Exception as e:
            print(f"Error: {e}")
            continue

    scores_f = np.array(scores_faithful)
    scores_h = np.array(scores_hallucinated)

    # Conformal calibration for 95% recall
    n = min(len(scores_f), len(scores_h))
    cal_size = n // 2

    cal_h = scores_h[:cal_size]
    test_f = scores_f[cal_size:]
    test_h = scores_h[cal_size:]

    # Find threshold for 95% coverage on hallucinated
    alpha = 0.05
    threshold = np.percentile(cal_h, 5)  # 5th percentile to catch 95%

    # Test
    coverage = np.mean(test_h >= threshold)
    fpr = np.mean(test_f >= threshold)

    fpr_ci = wilson_ci(fpr, len(test_f))

    results = {
        "method": "OpenAI text-embedding-3-large",
        "n_samples": n * 2,
        "faithful_mean": float(np.mean(scores_f)),
        "hallucinated_mean": float(np.mean(scores_h)),
        "coverage": float(coverage),
        "fpr": float(fpr),
        "fpr_ci": [float(fpr_ci[0]), float(fpr_ci[1])],
        "threshold": float(threshold)
    }

    print(f"\nOpenAI Embeddings Results:")
    print(f"  Coverage: {coverage*100:.1f}%")
    print(f"  FPR: {fpr*100:.1f}% [{fpr_ci[0]*100:.1f}%, {fpr_ci[1]*100:.1f}%]")

    return results


def run_bart_mnli(faithful, hallucinated):
    """Test BART-large-MNLI."""
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, skipping")
        return None

    device = 0 if torch.cuda.is_available() else -1
    print(f"\nRunning BART-large-MNLI baseline (device: {'cuda' if device == 0 else 'cpu'})...")

    nli = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

    scores_faithful = []
    scores_hallucinated = []

    print("Scoring faithful samples...")
    for s in tqdm(faithful):
        try:
            premise = f"Context: {s['knowledge']}"
            result = nli(premise, [s['answer']], hypothesis_template="{}")
            score = result['scores'][0]
            scores_faithful.append(1 - score)  # Non-conformity
        except Exception as e:
            print(f"Error: {e}")
            continue

    print("Scoring hallucinated samples...")
    for s in tqdm(hallucinated):
        try:
            premise = f"Context: {s['knowledge']}"
            result = nli(premise, [s['answer']], hypothesis_template="{}")
            score = result['scores'][0]
            scores_hallucinated.append(1 - score)
        except Exception as e:
            print(f"Error: {e}")
            continue

    scores_f = np.array(scores_faithful)
    scores_h = np.array(scores_hallucinated)

    # Conformal calibration
    n = min(len(scores_f), len(scores_h))
    cal_size = n // 2

    cal_h = scores_h[:cal_size]
    test_f = scores_f[cal_size:]
    test_h = scores_h[cal_size:]

    threshold = np.percentile(cal_h, 5)

    coverage = np.mean(test_h >= threshold)
    fpr = np.mean(test_f >= threshold)
    fpr_ci = wilson_ci(fpr, len(test_f))

    results = {
        "method": "BART-large-MNLI",
        "n_samples": n * 2,
        "faithful_mean": float(np.mean(scores_f)),
        "hallucinated_mean": float(np.mean(scores_h)),
        "coverage": float(coverage),
        "fpr": float(fpr),
        "fpr_ci": [float(fpr_ci[0]), float(fpr_ci[1])],
        "threshold": float(threshold)
    }

    print(f"\nBART-MNLI Results:")
    print(f"  Coverage: {coverage*100:.1f}%")
    print(f"  FPR: {fpr*100:.1f}% [{fpr_ci[0]*100:.1f}%, {fpr_ci[1]*100:.1f}%]")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai", action="store_true", help="Run OpenAI embeddings")
    parser.add_argument("--bart", action="store_true", help="Run BART-MNLI")
    parser.add_argument("--n_samples", type=int, default=200, help="Samples per class")
    args = parser.parse_args()

    if not args.openai and not args.bart:
        args.openai = True
        args.bart = True

    faithful, hallucinated = load_halueval_samples(args.n_samples)
    print(f"Loaded {len(faithful)} faithful, {len(hallucinated)} hallucinated samples")

    results = {}

    if args.openai:
        openai_results = run_openai_embeddings(faithful, hallucinated)
        if openai_results:
            results["openai_embeddings"] = openai_results

    if args.bart:
        bart_results = run_bart_mnli(faithful, hallucinated)
        if bart_results:
            results["bart_mnli"] = bart_results

    # Save results
    output_path = Path(__file__).parent.parent / "outputs" / "sota_baselines_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
