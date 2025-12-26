#!/usr/bin/env python3
"""
DeBERTa-v3-large-MNLI baseline experiment for Conformal RAG Guardrails.
Tests the "Semantic Illusion" hypothesis with state-of-the-art NLI model.
"""

import json
import numpy as np
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for proportions."""
    if n == 0:
        return (0.0, 1.0)
    denominator = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denominator
    adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return (max(0, centre - adjustment), min(1, centre + adjustment))


def run_deberta_experiment():
    """Run DeBERTa-v3-large-MNLI baseline on HaluEval."""

    print("=" * 60)
    print("DeBERTa-v3-large-MNLI Baseline Experiment")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else -1
    print(f"\nUsing device: {'CUDA' if device == 0 else 'CPU'}")

    # Load DeBERTa-v3-large-MNLI (using publicly available variant)
    print("\nLoading DeBERTa-v3-large-MNLI model...")
    nli_model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device=device
    )
    print("Model loaded successfully!")

    # Load HaluEval QA subset
    print("\nLoading HaluEval QA dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # HaluEval structure:
    # - knowledge: context
    # - question: the question
    # - answer: the response (may be hallucinated or not)
    # - hallucination: "yes" or "no"

    # Separate faithful and hallucinated samples
    faithful_samples = []
    hallucinated_samples = []

    for sample in dataset:
        if sample['hallucination'] == 'no':
            faithful_samples.append(sample)
        else:
            hallucinated_samples.append(sample)

    print(f"Total samples: {len(dataset)}")
    print(f"Faithful: {len(faithful_samples)}, Hallucinated: {len(hallucinated_samples)}")

    # Sample for efficiency
    n_samples = 100  # 100 from each class
    np.random.seed(42)

    faithful_indices = np.random.choice(len(faithful_samples), min(n_samples, len(faithful_samples)), replace=False)
    hallucinated_indices = np.random.choice(len(hallucinated_samples), min(n_samples, len(hallucinated_samples)), replace=False)

    print(f"\nSampling {len(faithful_indices)} faithful and {len(hallucinated_indices)} hallucinated")

    # Collect NLI scores
    scores_faithful = []
    scores_hallucinated = []

    print("\nProcessing faithful samples...")
    for idx in tqdm(faithful_indices, desc="Faithful"):
        sample = faithful_samples[int(idx)]
        premise = f"Context: {sample['knowledge']}\nQuestion: {sample['question']}"
        answer = sample['answer']

        try:
            result = nli_model(
                premise,
                [answer],
                hypothesis_template="{}"
            )
            score = result['scores'][0]
            scores_faithful.append(score)
        except Exception as e:
            print(f"Error: {e}")
            continue

    print("\nProcessing hallucinated samples...")
    for idx in tqdm(hallucinated_indices, desc="Hallucinated"):
        sample = hallucinated_samples[int(idx)]
        premise = f"Context: {sample['knowledge']}\nQuestion: {sample['question']}"
        answer = sample['answer']

        try:
            result = nli_model(
                premise,
                [answer],
                hypothesis_template="{}"
            )
            score = result['scores'][0]
            scores_hallucinated.append(score)
        except Exception as e:
            print(f"Error: {e}")
            continue

    scores_faithful = np.array(scores_faithful)
    scores_hallucinated = np.array(scores_hallucinated)

    print(f"\n{'='*60}")
    print("SCORE DISTRIBUTION")
    print(f"{'='*60}")

    print(f"\nFaithful answer NLI scores:")
    print(f"  Mean: {np.mean(scores_faithful):.3f}")
    print(f"  Std:  {np.std(scores_faithful):.3f}")
    print(f"  Min:  {np.min(scores_faithful):.3f}")
    print(f"  Max:  {np.max(scores_faithful):.3f}")

    print(f"\nHallucinated answer NLI scores:")
    print(f"  Mean: {np.mean(scores_hallucinated):.3f}")
    print(f"  Std:  {np.std(scores_hallucinated):.3f}")
    print(f"  Min:  {np.min(scores_hallucinated):.3f}")
    print(f"  Max:  {np.max(scores_hallucinated):.3f}")

    # Score overlap analysis
    overlap_threshold = min(np.max(scores_faithful), np.max(scores_hallucinated))
    faithful_in_overlap = np.mean(scores_faithful <= overlap_threshold)
    hallucinated_in_overlap = np.mean(scores_hallucinated <= overlap_threshold)

    print(f"\nScore Overlap Analysis:")
    print(f"  Score range overlap exists: {np.min(scores_hallucinated) < np.max(scores_faithful)}")

    # Compute split conformal prediction threshold for 95% coverage
    n = min(len(scores_faithful), len(scores_hallucinated))
    cal_size = n // 2

    cal_hallucinated = scores_hallucinated[:cal_size]
    test_faithful = scores_faithful[cal_size:]
    test_hallucinated = scores_hallucinated[cal_size:]

    # Non-conformity score: 1 - NLI_score (higher = less conforming/more suspicious)
    # Calibrate on hallucinated samples to ensure 95% recall
    alpha = 0.05
    cal_scores = 1 - cal_hallucinated  # Non-conformity for hallucinated

    # Quantile for 95% coverage
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    threshold = np.quantile(cal_scores, min(q_level, 1.0))

    print(f"\n{'='*60}")
    print("CONFORMAL PREDICTION CALIBRATION")
    print(f"{'='*60}")
    print(f"  Alpha: {alpha}")
    print(f"  Calibration size: {cal_size}")
    print(f"  Non-conformity threshold: {threshold:.4f}")
    print(f"  Equivalent NLI threshold: {1 - threshold:.4f}")
    print(f"  (Flag as hallucination if NLI score < {1-threshold:.4f})")

    # Test predictions
    # Predict hallucination if non-conformity > threshold (i.e., NLI score < 1-threshold)
    test_nonconf_faithful = 1 - test_faithful
    test_nonconf_hallucinated = 1 - test_hallucinated

    pred_faithful = test_nonconf_faithful > threshold  # FP: faithful flagged as hallucination
    pred_hallucinated = test_nonconf_hallucinated > threshold  # TP: hallucination correctly flagged

    # Coverage: P(flagged | hallucinated) = Recall/Sensitivity
    coverage = np.mean(pred_hallucinated)
    coverage_ci = wilson_ci(coverage, len(pred_hallucinated))

    # FPR: P(flagged | faithful)
    fpr = np.mean(pred_faithful)
    fpr_ci = wilson_ci(fpr, len(pred_faithful))

    print(f"\n{'='*60}")
    print("CONFORMAL PREDICTION RESULTS @ 95% TARGET COVERAGE")
    print(f"{'='*60}")
    print(f"  Test set size: {len(test_faithful)} faithful, {len(test_hallucinated)} hallucinated")
    print(f"  Coverage (Recall): {coverage*100:.1f}% [{coverage_ci[0]*100:.1f}%, {coverage_ci[1]*100:.1f}%]")
    print(f"  FPR:               {fpr*100:.1f}% [{fpr_ci[0]*100:.1f}%, {fpr_ci[1]*100:.1f}%]")

    # Save results
    results = {
        "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "dataset": "HaluEval-QA",
        "n_faithful": len(scores_faithful),
        "n_hallucinated": len(scores_hallucinated),
        "alpha": alpha,
        "threshold": float(threshold),
        "nli_threshold": float(1 - threshold),
        "coverage": float(coverage),
        "coverage_ci": [float(coverage_ci[0]), float(coverage_ci[1])],
        "fpr": float(fpr),
        "fpr_ci": [float(fpr_ci[0]), float(fpr_ci[1])],
        "faithful_scores": {
            "mean": float(np.mean(scores_faithful)),
            "std": float(np.std(scores_faithful)),
            "min": float(np.min(scores_faithful)),
            "max": float(np.max(scores_faithful))
        },
        "hallucinated_scores": {
            "mean": float(np.mean(scores_hallucinated)),
            "std": float(np.std(scores_hallucinated)),
            "min": float(np.min(scores_hallucinated)),
            "max": float(np.max(scores_hallucinated))
        }
    }

    with open("deberta_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to deberta_results.json")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")

    score_diff = np.mean(scores_faithful) - np.mean(scores_hallucinated)
    print(f"Mean score difference (faithful - hallucinated): {score_diff:.3f}")

    if fpr > 0.90:
        print(f"\nDeBERTa-v3-large CONFIRMS the 'Semantic Illusion' hypothesis!")
        print(f"Even SOTA NLI achieves {fpr*100:.0f}% FPR at {coverage*100:.0f}% coverage.")
        print(f"Hallucinations are semantically indistinguishable from faithful responses.")
    elif fpr > 0.50:
        print(f"\nDeBERTa-v3-large shows PARTIAL vulnerability to Semantic Illusion.")
        print(f"FPR of {fpr*100:.0f}% is better than simpler models but still problematic.")
    else:
        print(f"\nDeBERTa-v3-large achieves acceptable FPR: {fpr*100:.0f}%")
        print(f"This would challenge the Semantic Illusion hypothesis.")

    return results


if __name__ == "__main__":
    run_deberta_experiment()
