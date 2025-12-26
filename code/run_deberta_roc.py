#!/usr/bin/env python3
"""
DeBERTa ROC analysis for proper FPR at 95% recall.
"""

import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denominator
    adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return (max(0, centre - adjustment), min(1, centre + adjustment))


def run_roc_analysis():
    print("=" * 60)
    print("DeBERTa-v3-large ROC Analysis")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else -1
    print(f"\nUsing device: {'CUDA' if device == 0 else 'CPU'}")

    print("\nLoading DeBERTa-v3-large-MNLI model...")
    nli_model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device=device
    )
    print("Model loaded!")

    print("\nLoading HaluEval QA dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    faithful = [s for s in dataset if s['hallucination'] == 'no']
    hallucinated = [s for s in dataset if s['hallucination'] == 'yes']

    # Use more samples for better statistics
    n_samples = 200
    np.random.seed(42)

    f_idx = np.random.choice(len(faithful), min(n_samples, len(faithful)), replace=False)
    h_idx = np.random.choice(len(hallucinated), min(n_samples, len(hallucinated)), replace=False)

    print(f"Processing {len(f_idx)} faithful and {len(h_idx)} hallucinated samples...")

    scores_f = []
    scores_h = []

    print("\nScoring faithful samples...")
    for idx in tqdm(f_idx, desc="Faithful"):
        s = faithful[int(idx)]
        premise = f"Context: {s['knowledge']}\nQuestion: {s['question']}"
        try:
            r = nli_model(premise, [s['answer']], hypothesis_template="{}")
            scores_f.append(r['scores'][0])
        except:
            continue

    print("\nScoring hallucinated samples...")
    for idx in tqdm(h_idx, desc="Hallucinated"):
        s = hallucinated[int(idx)]
        premise = f"Context: {s['knowledge']}\nQuestion: {s['question']}"
        try:
            r = nli_model(premise, [s['answer']], hypothesis_template="{}")
            scores_h.append(r['scores'][0])
        except:
            continue

    scores_f = np.array(scores_f)
    scores_h = np.array(scores_h)

    print(f"\n{'='*60}")
    print("SCORE STATISTICS")
    print(f"{'='*60}")
    print(f"Faithful:     mean={np.mean(scores_f):.3f}, std={np.std(scores_f):.3f}")
    print(f"Hallucinated: mean={np.mean(scores_h):.3f}, std={np.std(scores_h):.3f}")
    print(f"Separation:   {np.mean(scores_f) - np.mean(scores_h):.3f}")

    # ROC Analysis: Find threshold for 95% recall (TPR)
    # Hallucinations have LOWER scores, so flag if score < threshold
    # TPR = P(score < threshold | hallucinated)
    # FPR = P(score < threshold | faithful)

    print(f"\n{'='*60}")
    print("ROC ANALYSIS: Finding FPR at 95% Recall")
    print(f"{'='*60}")

    # Sort hallucinated scores to find 95th percentile
    # We want threshold such that 95% of hallucinated have score < threshold
    threshold_95 = np.percentile(scores_h, 95)

    tpr = np.mean(scores_h <= threshold_95)
    fpr = np.mean(scores_f <= threshold_95)

    tpr_ci = wilson_ci(tpr, len(scores_h))
    fpr_ci = wilson_ci(fpr, len(scores_f))

    print(f"\nAt threshold = {threshold_95:.3f} (95th percentile of hallucinated):")
    print(f"  TPR (Recall): {tpr*100:.1f}% [{tpr_ci[0]*100:.1f}%, {tpr_ci[1]*100:.1f}%]")
    print(f"  FPR:          {fpr*100:.1f}% [{fpr_ci[0]*100:.1f}%, {fpr_ci[1]*100:.1f}%]")

    # Also compute AUC manually
    thresholds = np.linspace(0, 1, 100)
    tprs = [np.mean(scores_h <= t) for t in thresholds]
    fprs = [np.mean(scores_f <= t) for t in thresholds]

    # AUC via trapezoidal rule
    auc = np.trapz(tprs, fprs)

    print(f"\n  AUC: {auc:.3f}")

    # Save results
    results = {
        "model": "DeBERTa-v3-large-MNLI",
        "n_faithful": len(scores_f),
        "n_hallucinated": len(scores_h),
        "faithful_mean": float(np.mean(scores_f)),
        "hallucinated_mean": float(np.mean(scores_h)),
        "separation": float(np.mean(scores_f) - np.mean(scores_h)),
        "threshold_95_recall": float(threshold_95),
        "tpr_at_95": float(tpr),
        "fpr_at_95_recall": float(fpr),
        "fpr_ci": [float(fpr_ci[0]), float(fpr_ci[1])],
        "auc": float(auc)
    }

    output_path = Path("outputs") / "deberta_roc_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")

    if fpr < 0.30:
        print(f"\nDeBERTa-v3-large DOES distinguish hallucinations!")
        print(f"FPR of {fpr*100:.0f}% at 95% recall is significantly better than embeddings.")
        print(f"This supports the hypothesis that NLI-based reasoning")
        print(f"can detect hallucinations that fool embedding similarity.")
    else:
        print(f"\nDeBERTa-v3-large partially affected by Semantic Illusion.")
        print(f"FPR of {fpr*100:.0f}% at 95% recall shows embedding-level methods")
        print(f"are fundamentally limited even with SOTA NLI.")

    return results


if __name__ == "__main__":
    run_roc_analysis()
