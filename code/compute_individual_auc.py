#!/usr/bin/env python3
"""
Compute individual AUC values for RAD, SEC, TFG scores.
Validates paper Table 4 claims: RAD AUC=0.72, SEC AUC=0.81, TFG AUC=0.65
"""
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import ConformalRAGGuardrails, CRGConfig, RAGExample

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_halueval(n_samples: int = 2000, seed: int = 42) -> list:
    """Load HaluEval dataset."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required.")

    print("Loading HaluEval from HuggingFace...")
    np.random.seed(seed)
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    examples = []
    for idx, item in enumerate(tqdm(dataset, total=min(n_samples, len(dataset)))):
        if len(examples) >= n_samples:
            break
        question = item.get('question', '')
        knowledge = item.get('knowledge', '')
        answer = item.get('answer', '')
        is_hallucination = item.get('hallucination', 'no').lower() == 'yes'

        if not question or not knowledge or not answer:
            continue

        examples.append(RAGExample(
            query=question,
            documents=[knowledge],
            response=answer,
            label=1 if is_hallucination else 0
        ))

    np.random.shuffle(examples)
    return examples[:n_samples]


def compute_individual_aucs(examples: list, device: str = "cuda") -> dict:
    """Compute individual AUC for RAD, SEC, TFG scores."""

    print(f"\nComputing individual scores for {len(examples)} examples...")
    print(f"Device: {device}")

    config = CRGConfig(device=device)
    model = ConformalRAGGuardrails(config)

    # Compute individual scores
    individual_scores = model.compute_individual_scores(examples)

    # Get labels
    labels = np.array([ex.label for ex in examples])

    # Compute AUC for each score type
    results = {}

    for score_name, scores in individual_scores.items():
        scores_np = scores.cpu().numpy()

        # For hallucination detection, higher score = more likely hallucinated
        # So we want AUC where hallucinated (label=1) has higher scores
        try:
            auc = roc_auc_score(labels, scores_np)
            # If AUC < 0.5, the score is inverted (lower = hallucinated)
            # We report the corrected AUC
            if auc < 0.5:
                auc_corrected = 1 - auc
                inverted = True
            else:
                auc_corrected = auc
                inverted = False

            results[score_name] = {
                'auc_raw': float(auc),
                'auc_corrected': float(auc_corrected),
                'inverted': inverted,
                'mean_hallucinated': float(scores_np[labels == 1].mean()),
                'mean_faithful': float(scores_np[labels == 0].mean()),
                'std_hallucinated': float(scores_np[labels == 1].std()),
                'std_faithful': float(scores_np[labels == 0].std()),
            }

            print(f"\n{score_name.upper()}:")
            print(f"  AUC (raw): {auc:.4f}")
            print(f"  AUC (corrected): {auc_corrected:.4f}")
            print(f"  Inverted: {inverted}")
            print(f"  Mean (hallucinated): {scores_np[labels == 1].mean():.4f}")
            print(f"  Mean (faithful): {scores_np[labels == 0].mean():.4f}")

        except Exception as e:
            print(f"  Error computing AUC for {score_name}: {e}")
            results[score_name] = {'error': str(e)}

    # Also compute ensemble AUC
    ensemble_scores = model.compute_ensemble_score(individual_scores)
    ensemble_np = ensemble_scores.cpu().numpy()

    try:
        ensemble_auc = roc_auc_score(labels, ensemble_np)
        if ensemble_auc < 0.5:
            ensemble_auc_corrected = 1 - ensemble_auc
            inverted = True
        else:
            ensemble_auc_corrected = ensemble_auc
            inverted = False

        results['ensemble'] = {
            'auc_raw': float(ensemble_auc),
            'auc_corrected': float(ensemble_auc_corrected),
            'inverted': inverted,
        }

        print(f"\nENSEMBLE:")
        print(f"  AUC (corrected): {ensemble_auc_corrected:.4f}")

    except Exception as e:
        print(f"  Error computing ensemble AUC: {e}")
        results['ensemble'] = {'error': str(e)}

    return results


def main():
    print("=" * 60)
    print("INDIVIDUAL AUC COMPUTATION FOR PAPER VALIDATION")
    print("=" * 60)

    # Load HaluEval
    examples = load_halueval(n_samples=2000, seed=42)

    # Count labels
    n_hallucinated = sum(1 for ex in examples if ex.label == 1)
    n_faithful = len(examples) - n_hallucinated
    print(f"\nDataset: {len(examples)} examples")
    print(f"  Hallucinated: {n_hallucinated}")
    print(f"  Faithful: {n_faithful}")

    # Compute AUCs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = compute_individual_aucs(examples, device=device)

    # Add metadata
    results['metadata'] = {
        'dataset': 'HaluEval',
        'n_samples': len(examples),
        'n_hallucinated': n_hallucinated,
        'n_faithful': n_faithful,
        'seed': 42,
    }

    # Paper claims
    paper_claims = {
        'rad': 0.72,
        'sec': 0.81,
        'tfg': 0.65,
    }

    # Validation
    print("\n" + "=" * 60)
    print("PAPER VALIDATION (Table 4)")
    print("=" * 60)

    validation = {}
    for score_name, claimed_auc in paper_claims.items():
        if score_name in results and 'auc_corrected' in results[score_name]:
            actual_auc = results[score_name]['auc_corrected']
            diff = abs(actual_auc - claimed_auc)
            match = diff < 0.05  # 5% tolerance

            status = "MATCH" if match else "MISMATCH"
            print(f"{score_name.upper()}: Paper={claimed_auc:.2f}, Actual={actual_auc:.2f} [{status}]")

            validation[score_name] = {
                'paper_claim': claimed_auc,
                'actual': actual_auc,
                'difference': diff,
                'match': match,
            }
        else:
            print(f"{score_name.upper()}: ERROR - could not compute")
            validation[score_name] = {'error': 'computation failed'}

    results['validation'] = validation

    # Save results
    output_path = Path(__file__).parent.parent / "outputs" / "individual_auc_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
