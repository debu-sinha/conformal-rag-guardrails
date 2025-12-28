#!/usr/bin/env python3
"""Comprehensive experiments: WikiBio, RAGTruth, calibration ablation, cost analysis."""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import ConformalRAGGuardrails, CRGConfig, RAGExample

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def load_wikibio(n_samples: int = 1000, seed: int = 42) -> list:
    """Load WikiBio GPT-3 hallucination dataset."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required")

    print("  Loading WikiBio (GPT-3 generations)...")
    np.random.seed(seed)

    try:
        dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
    except Exception as e:
        print(f"  WikiBio load failed: {e}")
        return []

    examples = []
    for item in tqdm(dataset, total=min(n_samples, len(dataset))):
        if len(examples) >= n_samples:
            break

        wiki_bio = item.get('wiki_bio_text', '')
        gpt3_text = item.get('gpt3_text', '')
        annotations = item.get('annotation', [])

        if not wiki_bio or not gpt3_text:
            continue

        # Check if any sentence is hallucinated
        # Annotations are strings directly: 'accurate', 'minor_inaccurate', 'major_inaccurate'
        is_hallucinated = any(
            ann in ['minor_inaccurate', 'major_inaccurate']
            for ann in annotations if isinstance(ann, str)
        ) if annotations else False

        examples.append(RAGExample(
            query="Generate a biography based on the following information.",
            documents=[wiki_bio],
            response=gpt3_text,
            label=1 if is_hallucinated else 0
        ))

    np.random.shuffle(examples)
    return examples[:n_samples]


def load_ragtruth_full(n_samples: int = 2000, seed: int = 42) -> list:
    """Load RAGTruth dataset with multiple LLM sources."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required")

    print("  Loading RAGTruth (multi-LLM)...")
    np.random.seed(seed)

    try:
        dataset = load_dataset("lytang/RAGTruth", split="test")
    except Exception:
        try:
            dataset = load_dataset("nvidia/RAGTruth-corpus", split="test")
        except Exception as e:
            print(f"  RAGTruth not available: {e}")
            return []

    examples = []
    for item in tqdm(dataset, total=min(n_samples * 2, len(dataset))):
        if len(examples) >= n_samples:
            break

        query = item.get('question', item.get('query', item.get('input', '')))
        context = item.get('context', item.get('passage', item.get('source', '')))
        response = item.get('response', item.get('answer', item.get('output', '')))
        label = item.get('label', item.get('is_hallucination', 0))

        if isinstance(label, str):
            label = 1 if label.lower() in ['yes', 'true', '1', 'hallucinated'] else 0

        if not query or not context or not response:
            continue

        examples.append(RAGExample(
            query=str(query),
            documents=[str(context)] if isinstance(context, str) else [str(c) for c in context],
            response=str(response),
            label=int(label)
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


def run_calibration_ablation(model, examples, calibration_sizes=[300, 600, 1000], alpha=0.05, n_trials=5):
    """Ablation study on calibration set size."""
    print("\n" + "="*60)
    print("CALIBRATION SIZE ABLATION")
    print("="*60)

    results = []

    # Split into calibration pool and test
    n_test = int(len(examples) * 0.3)
    test_data = examples[-n_test:]
    cal_pool = [ex for ex in examples[:-n_test] if ex.label == 1]

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

            model.calibrate(cal_subset, alpha=alpha)
            threshold = model._threshold

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

        print(f"  n={n_cal:4d}: Coverage={result['mean_coverage']:.2%}±{result['std_coverage']:.2%}, "
              f"FPR={result['mean_fpr']:.2%}±{result['std_fpr']:.2%}")

    return results


def run_cost_analysis(model, examples, n_samples=100):
    """Measure compute cost and latency for different methods."""
    print("\n" + "="*60)
    print("COMPUTE COST ANALYSIS")
    print("="*60)

    test_subset = examples[:n_samples]
    results = {}

    # 1. Embedding-based (CRG)
    print("  Measuring CRG (embedding) latency...")
    start = time.time()
    for _ in range(3):  # 3 trials
        _ = model.compute_individual_scores(test_subset)
    crg_time = (time.time() - start) / 3

    results["crg_embedding"] = {
        "method": "CRG (BGE + BART-MNLI)",
        "latency_per_sample_ms": (crg_time / n_samples) * 1000,
        "throughput_samples_per_sec": n_samples / crg_time,
        "cost_per_1k_samples_usd": 0.0001 * 1000,  # ~$0.0001 per sample (compute only)
        "requires_api": False
    }
    print(f"    Latency: {results['crg_embedding']['latency_per_sample_ms']:.1f}ms/sample")

    # 2. GPT-4o-mini (if available)
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        print("  Measuring GPT-4o-mini latency...")
        client = openai.OpenAI()

        latencies = []
        for ex in test_subset[:10]:  # Only 10 samples to save cost
            start = time.time()
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Is this response factually supported? Context: {ex.documents[0][:500]} Response: {ex.response[:200]} Answer 0-1:"
                    }],
                    max_tokens=5,
                    temperature=0
                )
                latencies.append(time.time() - start)
            except Exception as e:
                print(f"    GPT-4 call failed: {e}")
                break

        if latencies:
            avg_latency = np.mean(latencies)
            results["gpt4o_mini"] = {
                "method": "GPT-4o-mini Judge",
                "latency_per_sample_ms": avg_latency * 1000,
                "throughput_samples_per_sec": 1 / avg_latency,
                "cost_per_1k_samples_usd": 0.015 * 1000,  # ~$0.015 per sample
                "requires_api": True
            }
            print(f"    Latency: {results['gpt4o_mini']['latency_per_sample_ms']:.1f}ms/sample")

    # 3. Summary comparison
    print("\n  COST COMPARISON:")
    print("  " + "-"*50)
    for name, data in results.items():
        speedup = results["gpt4o_mini"]["latency_per_sample_ms"] / data["latency_per_sample_ms"] if "gpt4o_mini" in results and name != "gpt4o_mini" else 1
        cost_ratio = data["cost_per_1k_samples_usd"] / results["gpt4o_mini"]["cost_per_1k_samples_usd"] if "gpt4o_mini" in results else 1
        print(f"    {data['method']:25s}: {data['latency_per_sample_ms']:6.1f}ms, ${data['cost_per_1k_samples_usd']:.2f}/1k")
        if name != "gpt4o_mini" and "gpt4o_mini" in results:
            print(f"      -> {speedup:.0f}x faster, {1/cost_ratio:.0f}x cheaper than GPT-4")

    return results


def run_dataset_experiment(model, examples, dataset_name, alpha=0.05):
    """Run full experiment on a dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    if not examples:
        print("  No examples loaded, skipping...")
        return None

    # Split
    n_cal = int(len(examples) * 0.6)
    cal_data = examples[:n_cal]
    test_data = examples[n_cal:]

    cal_hall = [ex for ex in cal_data if ex.label == 1]
    cal_faith = [ex for ex in cal_data if ex.label == 0]

    print(f"  Samples: {len(examples)} total")
    print(f"  Calibration: {len(cal_hall)} hallucinated, {len(cal_faith)} faithful")
    print(f"  Test: {len(test_data)}")

    if len(cal_hall) < 50:
        print("  WARNING: Not enough hallucinated samples for calibration")
        return None

    # Compute scores
    print("  Computing scores...")
    test_scores = model.compute_individual_scores(test_data)
    test_ensemble = model.compute_ensemble_score(test_scores)
    test_labels = torch.tensor([ex.label for ex in test_data])

    # Calibrate
    model.calibrate(cal_hall, alpha=alpha)
    threshold = model._threshold

    # Evaluate
    predictions = (test_ensemble >= threshold).float().cpu()
    test_labels_cpu = test_labels.cpu()

    hall_mask = test_labels_cpu == 1
    faith_mask = test_labels_cpu == 0

    n_hall = int(hall_mask.sum().item())
    n_faith = int(faith_mask.sum().item())

    coverage = (predictions[hall_mask] == 1).float().mean().item() if n_hall > 0 else 0
    fpr = (predictions[faith_mask] == 1).float().mean().item() if n_faith > 0 else 0
    accuracy = (predictions == test_labels_cpu.float()).float().mean().item()

    # Confidence intervals
    n_detected = int((predictions[hall_mask] == 1).sum().item())
    n_fp = int((predictions[faith_mask] == 1).sum().item())

    coverage_ci = compute_clopper_pearson_ci(n_detected, n_hall)
    fpr_ci = compute_clopper_pearson_ci(n_fp, n_faith)

    result = {
        "dataset": dataset_name,
        "n_samples": len(examples),
        "n_calibration": len(cal_hall),
        "n_test": len(test_data),
        "n_test_hall": n_hall,
        "n_test_faith": n_faith,
        "target_coverage": 1 - alpha,
        "coverage": coverage,
        "coverage_ci": list(coverage_ci),
        "coverage_ci_str": f"[{coverage_ci[0]:.1%}, {coverage_ci[1]:.1%}]",
        "fpr": fpr,
        "fpr_ci": list(fpr_ci),
        "fpr_ci_str": f"[{fpr_ci[0]:.1%}, {fpr_ci[1]:.1%}]",
        "accuracy": accuracy,
        "threshold": float(threshold)
    }

    print(f"\n  RESULTS:")
    print(f"    Coverage: {coverage:.1%} {result['coverage_ci_str']}")
    print(f"    FPR:      {fpr:.1%} {result['fpr_ci_str']}")
    print(f"    Accuracy: {accuracy:.1%}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CRG experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wikibio", action="store_true", help="Run WikiBio experiment")
    parser.add_argument("--ragtruth", action="store_true", help="Run RAGTruth experiment")
    parser.add_argument("--calibration_ablation", action="store_true", help="Run calibration size ablation")
    parser.add_argument("--cost_analysis", action="store_true", help="Run cost/latency analysis")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if args.all:
        args.wikibio = True
        args.ragtruth = True
        args.calibration_ablation = True
        args.cost_analysis = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*70)
    print("COMPREHENSIVE CRG EXPERIMENTS")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Samples per dataset: {args.n_samples}")

    # Initialize model
    config = CRGConfig(device=args.device, use_fp16=True)
    model = ConformalRAGGuardrails(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    all_results = {"timestamp": datetime.now().isoformat()}

    # WikiBio experiment
    if args.wikibio:
        wikibio_examples = load_wikibio(args.n_samples, args.seed)
        result = run_dataset_experiment(model, wikibio_examples, "wikibio")
        if result:
            all_results["wikibio"] = result

    # RAGTruth experiment
    if args.ragtruth:
        ragtruth_examples = load_ragtruth_full(args.n_samples, args.seed)
        result = run_dataset_experiment(model, ragtruth_examples, "ragtruth")
        if result:
            all_results["ragtruth"] = result

    # Calibration ablation (using HaluEval as base)
    if args.calibration_ablation:
        print("\nLoading HaluEval for calibration ablation...")
        from run_experiment import load_halueval
        halueval_examples = load_halueval(2000, args.seed)
        ablation_results = run_calibration_ablation(
            model, halueval_examples,
            calibration_sizes=[300, 600, 1000, 1500],
            n_trials=5
        )
        all_results["calibration_ablation"] = ablation_results

    # Cost analysis
    if args.cost_analysis:
        print("\nLoading samples for cost analysis...")
        from run_experiment import load_halueval
        cost_examples = load_halueval(200, args.seed)
        cost_results = run_cost_analysis(model, cost_examples, n_samples=100)
        all_results["cost_analysis"] = cost_results

    # Save results
    results_file = output_dir / "comprehensive_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
