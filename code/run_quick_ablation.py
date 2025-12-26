#!/usr/bin/env python3
"""Quick calibration ablation for local testing."""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

import torch
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import ConformalRAGGuardrails, CRGConfig, RAGExample

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_halueval_qa(n_samples: int = 800, seed: int = 42) -> list:
    """Load HaluEval QA subset."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required")

    print(f"Loading HaluEval QA (n={n_samples})...")
    np.random.seed(seed)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data", trust_remote_code=True)

    examples = []
    for item in tqdm(dataset, total=min(n_samples, len(dataset))):
        if len(examples) >= n_samples:
            break

        query = item.get('question', '')
        knowledge = item.get('knowledge', '')
        answer = item.get('answer', '')
        hallucination = item.get('hallucination', 'no')

        if not query or not knowledge or not answer:
            continue

        # Label: 1 if hallucinated, 0 if faithful
        label = 1 if hallucination.lower() == 'yes' else 0

        examples.append(RAGExample(
            query=str(query),
            documents=[str(knowledge)],
            response=str(answer),
            label=label
        ))

    np.random.shuffle(examples)
    return examples[:n_samples]


def compute_clopper_pearson_ci(successes: int, trials: int, alpha: float = 0.05):
    """Compute Clopper-Pearson exact confidence interval."""
    if trials == 0:
        return (0.0, 1.0)

    lower = stats.beta.ppf(alpha/2, successes, trials - successes + 1) if successes > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha/2, successes + 1, trials - successes) if successes < trials else 1.0

    return (lower, upper)


def run_calibration_ablation(model, examples, calibration_sizes, alpha=0.05, n_trials=3):
    """Quick calibration ablation study."""
    print("\n" + "="*60)
    print("CALIBRATION SIZE ABLATION (Quick)")
    print("="*60)

    results = []

    # Split: 30% test, rest for calibration pool
    n_test = int(len(examples) * 0.3)
    test_data = examples[-n_test:]
    cal_pool = [ex for ex in examples[:-n_test] if ex.label == 1]  # Only hallucinated for calibration

    print(f"  Test set: {len(test_data)} samples")
    print(f"  Calibration pool: {len(cal_pool)} hallucinated samples")

    # Precompute test scores once
    print("  Computing test scores...")
    test_labels = torch.tensor([ex.label for ex in test_data])
    test_scores = model.compute_individual_scores(test_data)
    test_ensemble = model.compute_ensemble_score(test_scores)

    for n_cal in calibration_sizes:
        if n_cal > len(cal_pool):
            print(f"  Skipping n={n_cal}: not enough calibration samples ({len(cal_pool)} available)")
            continue

        trial_coverages = []
        trial_fprs = []

        for trial in range(n_trials):
            np.random.seed(42 + trial)
            cal_indices = np.random.choice(len(cal_pool), n_cal, replace=False)
            cal_subset = [cal_pool[i] for i in cal_indices]

            # Calibrate
            model.calibrate(cal_subset, alpha=alpha)
            threshold = model._threshold

            # Evaluate
            predictions = (test_ensemble >= threshold).float().cpu()
            test_labels_cpu = test_labels.cpu()

            hall_mask = test_labels_cpu == 1
            faith_mask = test_labels_cpu == 0

            coverage = (predictions[hall_mask] == 1).float().mean().item() if hall_mask.sum() > 0 else 0
            fpr = (predictions[faith_mask] == 1).float().mean().item() if faith_mask.sum() > 0 else 0

            trial_coverages.append(coverage)
            trial_fprs.append(fpr)

        result = {
            "n_calibration": n_cal,
            "target_coverage": 1 - alpha,
            "mean_coverage": float(np.mean(trial_coverages)),
            "std_coverage": float(np.std(trial_coverages)),
            "mean_fpr": float(np.mean(trial_fprs)),
            "std_fpr": float(np.std(trial_fprs)),
            "n_trials": n_trials
        }
        results.append(result)

        print(f"  n={n_cal:4d}: Coverage={result['mean_coverage']:.1%}±{result['std_coverage']:.1%}, "
              f"FPR={result['mean_fpr']:.1%}±{result['std_fpr']:.1%}")

    return results


def _parse_sizes(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    sizes = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"Invalid calibration size: {part}")
        sizes.append(int(part))
    if not sizes:
        raise ValueError("No calibration sizes provided")
    return sizes


def main():
    print("="*70)
    print("QUICK CALIBRATION ABLATION")
    print("="*70)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=800)
    parser.add_argument("--calibration_sizes", type=str, default="200,400,600")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")
    print(f"Calibration sizes: {args.calibration_sizes}")
    print(f"Alpha: {args.alpha}")
    print(f"Trials: {args.n_trials}")

    # Initialize model
    config = CRGConfig(device=device, use_fp16=True)
    model = ConformalRAGGuardrails(config)

    # Load data
    examples = load_halueval_qa(n_samples=args.n_samples, seed=42)
    print(f"Loaded {len(examples)} examples")

    n_hall = sum(1 for ex in examples if ex.label == 1)
    n_faith = sum(1 for ex in examples if ex.label == 0)
    print(f"  Hallucinated: {n_hall}, Faithful: {n_faith}")

    # Run ablation
    calibration_sizes = _parse_sizes(args.calibration_sizes)
    results = run_calibration_ablation(
        model,
        examples,
        calibration_sizes=calibration_sizes,
        alpha=args.alpha,
        n_trials=args.n_trials
    )

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "calibration_ablation_quick.json"
    with open(results_file, "w") as f:
        json.dump({
            "calibration_ablation": results,
            "device": device,
            "n_samples": len(examples),
            "alpha": args.alpha,
            "calibration_sizes": calibration_sizes,
            "n_trials": args.n_trials
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE (for paper):")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("$|\\mathcal{D}_{cal}|$ & Coverage & FPR \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['n_calibration']} & {r['mean_coverage']:.1%}$\\pm${r['std_coverage']:.1%} & {r['mean_fpr']:.1%}$\\pm${r['std_fpr']:.1%} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Effect of calibration set size on coverage and FPR.}")
    print("\\label{tab:calibration_ablation}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
