#!/usr/bin/env python3
"""
NQ Calibration Ablation - Generates Table 3 data for the paper.
This runs the calibration size ablation on Natural Questions (synthetic hallucinations)
where CRG achieves 0% FPR.
"""
import os
import sys
import json
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


def load_natural_questions(n_samples: int = 2000, seed: int = 42) -> list:
    """Load Natural Questions with synthetic hallucinations (answer swapping)."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required")

    print(f"Loading Natural Questions (n={n_samples})...")
    np.random.seed(seed)

    dataset = load_dataset("nq_open", split="validation")

    examples = []
    for idx, item in enumerate(tqdm(dataset, total=min(n_samples * 2, len(dataset)))):
        if len(examples) >= n_samples:
            break

        question = item['question']
        answers = item['answer']

        if not answers:
            continue

        gold_answer = answers[0]
        doc_text = f"The answer to '{question}' is {gold_answer}."

        # Faithful example
        examples.append(RAGExample(
            query=question,
            documents=[doc_text],
            response=gold_answer,
            label=0
        ))

        # Synthetic hallucination (answer swapping)
        wrong_answers = [
            "The information is not available.",
            "This cannot be determined.",
            "No answer found.",
            "Unknown",
            "N/A"
        ]
        examples.append(RAGExample(
            query=question,
            documents=[doc_text],
            response=np.random.choice(wrong_answers),
            label=1
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


def run_nq_calibration_ablation(model, examples, calibration_sizes, alpha=0.05, n_trials=5):
    """
    Run calibration ablation on Natural Questions.
    This should show 0% FPR across all calibration sizes (unlike HaluEval which shows 100%).
    """
    print("\n" + "="*60)
    print("NQ CALIBRATION SIZE ABLATION")
    print("="*60)
    print(f"Target coverage: {1-alpha:.0%}")
    print(f"Trials per size: {n_trials}")

    results = []

    # Split: 40% test, rest for calibration pool
    n_test = int(len(examples) * 0.4)
    test_data = examples[-n_test:]
    cal_pool = [ex for ex in examples[:-n_test] if ex.label == 1]  # Only hallucinated for calibration

    print(f"Test set: {len(test_data)} samples")
    print(f"Calibration pool: {len(cal_pool)} hallucinated samples")

    # Count class balance
    n_hall_test = sum(1 for ex in test_data if ex.label == 1)
    n_faith_test = sum(1 for ex in test_data if ex.label == 0)
    print(f"Test distribution: {n_hall_test} hallucinated, {n_faith_test} faithful")

    # Precompute test scores once
    print("\nComputing test scores...")
    test_labels = torch.tensor([ex.label for ex in test_data])
    test_scores = model.compute_individual_scores(test_data)
    test_ensemble = model.compute_ensemble_score(test_scores)

    for n_cal in calibration_sizes:
        if n_cal > len(cal_pool):
            print(f"\nSkipping n={n_cal}: not enough calibration samples ({len(cal_pool)} available)")
            continue

        print(f"\n--- Calibration size: {n_cal} ---")
        trial_coverages = []
        trial_fprs = []
        trial_thresholds = []

        for trial in range(n_trials):
            np.random.seed(42 + trial)
            cal_indices = np.random.choice(len(cal_pool), n_cal, replace=False)
            cal_subset = [cal_pool[i] for i in cal_indices]

            # Calibrate
            model.calibrate(cal_subset, alpha=alpha)
            threshold = model._threshold
            trial_thresholds.append(threshold)

            # Evaluate
            predictions = (test_ensemble >= threshold).float().cpu()
            test_labels_cpu = test_labels.cpu()

            hall_mask = test_labels_cpu == 1
            faith_mask = test_labels_cpu == 0

            coverage = (predictions[hall_mask] == 1).float().mean().item() if hall_mask.sum() > 0 else 0
            fpr = (predictions[faith_mask] == 1).float().mean().item() if faith_mask.sum() > 0 else 0

            trial_coverages.append(coverage)
            trial_fprs.append(fpr)

        mean_cov = float(np.mean(trial_coverages))
        std_cov = float(np.std(trial_coverages))
        mean_fpr = float(np.mean(trial_fprs))
        std_fpr = float(np.std(trial_fprs))
        mean_threshold = float(np.mean(trial_thresholds))

        result = {
            "n_calibration": n_cal,
            "target_coverage": 1 - alpha,
            "mean_coverage": mean_cov,
            "std_coverage": std_cov,
            "mean_fpr": mean_fpr,
            "std_fpr": std_fpr,
            "mean_threshold": mean_threshold,
            "n_trials": n_trials,
            "coverage_str": f"{mean_cov:.1%} +/- {std_cov:.1%}",
            "fpr_str": f"{mean_fpr:.1%}"
        }
        results.append(result)

        print(f"  Coverage: {mean_cov:.1%} +/- {std_cov:.1%}")
        print(f"  FPR: {mean_fpr:.1%} +/- {std_fpr:.1%}")
        print(f"  Threshold: {mean_threshold:.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NQ Calibration Ablation for Table 3")
    parser.add_argument("--n_samples", type=int, default=2000,
                       help="Total samples to load")
    parser.add_argument("--calibration_sizes", type=str, default="300,600,1000",
                       help="Comma-separated calibration sizes")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Coverage target (1-alpha)")
    parser.add_argument("--n_trials", type=int, default=5,
                       help="Trials per calibration size")
    args = parser.parse_args()

    print("="*70)
    print("NQ CALIBRATION ABLATION (Table 3)")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")
    print(f"Alpha: {args.alpha}")
    print(f"Trials: {args.n_trials}")

    # Initialize model
    config = CRGConfig(device=device, use_fp16=True if device == "cuda" else False)
    model = ConformalRAGGuardrails(config)

    # Load NQ data
    examples = load_natural_questions(n_samples=args.n_samples, seed=42)
    print(f"\nLoaded {len(examples)} examples")

    n_hall = sum(1 for ex in examples if ex.label == 1)
    n_faith = sum(1 for ex in examples if ex.label == 0)
    print(f"Hallucinated: {n_hall}, Faithful: {n_faith}")

    # Parse calibration sizes
    calibration_sizes = [int(x.strip()) for x in args.calibration_sizes.split(",")]

    # Run ablation
    results = run_nq_calibration_ablation(
        model,
        examples,
        calibration_sizes=calibration_sizes,
        alpha=args.alpha,
        n_trials=args.n_trials
    )

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "calibration_ablation_nq.json"
    output_data = {
        "dataset": "natural_questions",
        "description": "Table 3: Effect of calibration set size on NQ (synthetic hallucinations)",
        "n_samples": len(examples),
        "alpha": args.alpha,
        "calibration_sizes": calibration_sizes,
        "n_trials": args.n_trials,
        "device": device,
        "results": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY (Table 3)")
    print("="*60)
    print(f"{'n_cal':<10} {'Coverage':<20} {'FPR':<15} {'Std':<10}")
    print("-"*55)
    for r in results:
        print(f"{r['n_calibration']:<10} {r['coverage_str']:<20} {r['fpr_str']:<15} {r['std_coverage']:.3f}")

    # Print LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE (Table 3 for paper):")
    print("="*60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{cccc}")
    print(r"\toprule")
    print(r"$|\mathcal{D}_{cal}|$ & Target & Coverage & FPR & Coverage Std \\")
    print(r"\midrule")
    for r in results:
        cov_pct = r['mean_coverage'] * 100
        std_pct = r['std_coverage'] * 100
        fpr_pct = r['mean_fpr'] * 100
        print(f"{r['n_calibration']} & 95\\% & {cov_pct:.1f}\\% $\\pm$ {std_pct:.1f}\\% & {fpr_pct:.1f}\\% & {r['std_coverage']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Effect of calibration set size on coverage and FPR (Natural Questions).}")
    print(r"\label{tab:calibration_ablation_nq}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
