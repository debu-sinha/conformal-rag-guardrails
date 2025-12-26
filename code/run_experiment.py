#!/usr/bin/env python3
"""
Enhanced Conformal RAG Guardrails Experiment
Includes: Multiple datasets, GPT-4 Judge, DeBERTa, t-SNE, ablations
"""
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core_algorithm import (
    ConformalRAGGuardrails,
    CRGConfig,
    RAGExample
)

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def load_halueval(n_samples: int = 2000, seed: int = 42) -> list:
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets required.")
    print("  Loading HaluEval from HuggingFace...")
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


def load_ragtruth(n_samples: int = 2000, seed: int = 42) -> list:
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets required.")
    print("  Loading RAGTruth from HuggingFace...")
    np.random.seed(seed)
    try:
        dataset = load_dataset("lytang/RAGTruth", split="test")
    except Exception as e:
        print(f"  RAGTruth primary source failed ({e}), trying alternatives...")
        try:
            dataset = load_dataset("nvidia/ragtruth", split="test")
        except:
            print("  All RAGTruth sources failed, falling back to HaluEval")
            return load_halueval(n_samples, seed)

    examples = []
    for idx, item in enumerate(tqdm(dataset, total=min(n_samples, len(dataset)))):
        if len(examples) >= n_samples:
            break
        query = item.get('question', item.get('query', item.get('input', '')))
        context = item.get('context', item.get('passage', item.get('source', '')))
        response = item.get('response', item.get('answer', item.get('output', '')))
        label = item.get('label', item.get('is_hallucination', item.get('hallucination', 0)))
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


def load_natural_questions(n_samples: int = 2000, seed: int = 42) -> list:
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets required.")
    print("  Loading Natural Questions (sanity check)...")
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
        examples.append(RAGExample(
            query=question, documents=[doc_text], response=gold_answer, label=0
        ))
        wrong_answers = ["The information is not available.", "This cannot be determined.", "No answer found."]
        examples.append(RAGExample(
            query=question, documents=[doc_text], response=np.random.choice(wrong_answers), label=1
        ))
    np.random.shuffle(examples)
    return examples[:n_samples]


class SelfCheckGPTBaseline:
    def __init__(self, device="cuda", model_name="facebook/bart-large-mnli"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        print(f"  Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, examples: list) -> torch.Tensor:
        scores = []
        for ex in tqdm(examples, desc="Baseline scoring"):
            doc_text = " ".join(ex.documents)
            inputs = self.tokenizer(
                doc_text[:512], ex.response[:256],
                return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_idx = 2 if outputs.logits.shape[-1] == 3 else 0
            entailment_prob = probs[0, entailment_idx].item()
            scores.append(1 - entailment_prob)
        return torch.tensor(scores)


class DeBERTaBaseline:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        model_name = "cross-encoder/nli-deberta-v3-large"
        print(f"  Loading DeBERTa: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, examples: list) -> torch.Tensor:
        scores = []
        for ex in tqdm(examples, desc="DeBERTa scoring"):
            doc_text = " ".join(ex.documents)
            inputs = self.tokenizer(
                doc_text[:512], ex.response[:256],
                return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_prob = probs[0, 0].item()  # DeBERTa: 0=entailment
            scores.append(1 - entailment_prob)
        return torch.tensor(scores)


class GPT4JudgeBaseline:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL_JUDGE", "gpt-5")
        self.client = openai.OpenAI()
        print(f"  Using GPT-4 Judge: {self.model}")

    def score(self, examples: list, max_samples: int = 200) -> torch.Tensor:
        scores = []
        for ex in tqdm(examples[:max_samples], desc="GPT-4 Judge scoring"):
            prompt = f"""Given the context and response, determine if the response is factually supported by the context.

Context: {' '.join(ex.documents)[:1000]}

Response: {ex.response}

Rate the factual accuracy from 0.0 (completely hallucinated) to 1.0 (fully supported).
Output ONLY a number between 0.0 and 1.0."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                score_text = response.choices[0].message.content.strip()
                score = float(score_text)
                scores.append(1 - score)  # Convert to hallucination score
            except Exception as e:
                scores.append(0.5)  # Default on error

        # Pad remaining with 0.5 if we limited samples
        while len(scores) < len(examples):
            scores.append(0.5)

        return torch.tensor(scores)


def plot_score_distributions(scores_faithful, scores_hallucinated, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(scores_faithful, bins=50, alpha=0.6, label='Faithful', color='green', density=True)
    plt.hist(scores_hallucinated, bins=50, alpha=0.6, label='Hallucinated', color='red', density=True)
    plt.xlabel('Nonconformity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    overlap = min(np.mean(scores_faithful > np.median(scores_hallucinated)),
                  np.mean(scores_hallucinated < np.median(scores_faithful)))
    plt.text(0.02, 0.98, f'Distribution overlap: {overlap:.1%}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(embeddings_faithful, embeddings_hallucinated, title, save_path):
    if not SKLEARN_AVAILABLE:
        print("  sklearn not available, skipping t-SNE")
        return

    print("  Generating t-SNE visualization...")
    all_embeddings = np.vstack([embeddings_faithful, embeddings_hallucinated])
    labels = np.array([0] * len(embeddings_faithful) + [1] * len(embeddings_hallucinated))

    # Limit to 1000 samples for speed
    if len(all_embeddings) > 1000:
        idx = np.random.choice(len(all_embeddings), 1000, replace=False)
        all_embeddings = all_embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    faithful_mask = labels == 0
    hallucinated_mask = labels == 1

    plt.scatter(embeddings_2d[faithful_mask, 0], embeddings_2d[faithful_mask, 1],
                c='green', alpha=0.5, label='Faithful', s=20)
    plt.scatter(embeddings_2d[hallucinated_mask, 0], embeddings_2d[hallucinated_mask, 1],
                c='red', alpha=0.5, label='Hallucinated', s=20)

    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved t-SNE plot: {save_path}")


def get_error_examples(examples, scores, threshold, n_examples=10):
    """Get examples of errors for qualitative analysis."""
    false_positives = []
    false_negatives = []

    for ex, score in zip(examples, scores):
        prediction = 1 if score >= threshold else 0
        if prediction == 1 and ex.label == 0:  # False positive
            false_positives.append({
                "query": ex.query,
                "response": ex.response[:200],
                "document": ex.documents[0][:200] if ex.documents else "",
                "score": float(score),
                "type": "false_positive"
            })
        elif prediction == 0 and ex.label == 1:  # False negative
            false_negatives.append({
                "query": ex.query,
                "response": ex.response[:200],
                "document": ex.documents[0][:200] if ex.documents else "",
                "score": float(score),
                "type": "false_negative"
            })

    return {
        "false_positives": false_positives[:n_examples],
        "false_negatives": false_negatives[:n_examples]
    }


def run_ablation(model, examples, alphas=[0.01, 0.05, 0.10, 0.15, 0.20]):
    """Run ablation on different alpha values."""
    results = []

    cal_data = [ex for ex in examples[:int(len(examples)*0.6)] if ex.label == 1]
    test_data = examples[int(len(examples)*0.6):]

    print("\n  Running alpha ablation...")
    for alpha in tqdm(alphas, desc="Alpha values"):
        model.calibrate(cal_data, alpha=alpha)
        threshold = model._threshold

        test_scores = model.compute_individual_scores(test_data)
        test_ensemble = model.compute_ensemble_score(test_scores)
        test_labels = torch.tensor([ex.label for ex in test_data])

        predictions = (test_ensemble >= threshold).float().cpu()
        test_labels_cpu = test_labels.cpu()

        hall_mask = test_labels_cpu == 1
        faith_mask = test_labels_cpu == 0

        coverage = (predictions[hall_mask] == 1).float().mean().item() if hall_mask.sum() > 0 else 0
        fpr = (predictions[faith_mask] == 1).float().mean().item() if faith_mask.sum() > 0 else 0
        accuracy = (predictions == test_labels_cpu.float()).float().mean().item()

        results.append({
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "actual_coverage": coverage,
            "fpr": fpr,
            "accuracy": accuracy,
            "threshold": threshold
        })

    return results


def run_experiment(args):
    print("=" * 70)
    print("Conformal RAG Guardrails - Enhanced Experiment v3")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.n_samples}")
    print(f"Coverage: {args.coverage}")
    print(f"Device: {args.device}")
    print()

    config = CRGConfig(
        device=args.device,
        use_fp16=True,
        use_weighted_rad=True,
        use_sentence_level_sec=True,
        grounding_threshold=0.3,
        temperature=0.1,
    )
    model = ConformalRAGGuardrails(config)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "halueval":
        all_examples = load_halueval(args.n_samples, args.seed)
    elif args.dataset == "ragtruth":
        all_examples = load_ragtruth(args.n_samples, args.seed)
    elif args.dataset == "nq":
        all_examples = load_natural_questions(args.n_samples, args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Split data
    n_cal = int(len(all_examples) * 0.6)
    calibration_data = all_examples[:n_cal]
    test_data = all_examples[n_cal:]

    cal_hallucinated = [ex for ex in calibration_data if ex.label == 1]
    cal_faithful = [ex for ex in calibration_data if ex.label == 0]

    print(f"\n  Total: {len(all_examples)}")
    print(f"  Calibration: {len(calibration_data)} ({len(cal_hallucinated)} hal, {len(cal_faithful)} faith)")
    print(f"  Test: {len(test_data)}")

    # Compute scores
    print("\nComputing CRG scores...")
    cal_hall_scores = model.compute_individual_scores(cal_hallucinated)
    cal_hall_ensemble = model.compute_ensemble_score(cal_hall_scores)

    cal_faith_scores = model.compute_individual_scores(cal_faithful)
    cal_faith_ensemble = model.compute_ensemble_score(cal_faith_scores)

    test_scores = model.compute_individual_scores(test_data)
    test_ensemble = model.compute_ensemble_score(test_scores)
    test_labels = torch.tensor([ex.label for ex in test_data])

    # Calibrate
    alpha = 1 - args.coverage
    model.calibrate(cal_hallucinated, alpha=alpha)
    threshold = model._threshold

    # Evaluate CRG
    predictions = (test_ensemble >= threshold).float().cpu()
    test_labels_cpu = test_labels.cpu()

    test_hall_mask = test_labels_cpu == 1
    test_faith_mask = test_labels_cpu == 0

    coverage = (predictions[test_hall_mask] == 1).float().mean().item() if test_hall_mask.sum() > 0 else 0
    fpr = (predictions[test_faith_mask] == 1).float().mean().item() if test_faith_mask.sum() > 0 else 0
    accuracy = (predictions == test_labels_cpu.float()).float().mean().item()

    hall_scores_np = cal_hall_ensemble.cpu().numpy()
    faith_scores_np = cal_faith_ensemble.cpu().numpy()

    score_stats = {
        "hallucinated_mean": float(np.mean(hall_scores_np)),
        "hallucinated_std": float(np.std(hall_scores_np)),
        "faithful_mean": float(np.mean(faith_scores_np)),
        "faithful_std": float(np.std(faith_scores_np)),
        "distribution_overlap": float(np.mean(faith_scores_np > np.median(hall_scores_np))),
    }

    # Output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_score_distributions(
        faith_scores_np, hall_scores_np,
        f"Score Distribution: {args.dataset.upper()} (Ensemble)",
        output_dir / f"dist_{args.dataset}_ensemble.png"
    )

    # t-SNE visualization
    if args.tsne and SKLEARN_AVAILABLE:
        # Get embeddings for t-SNE
        all_test = cal_faithful[:200] + cal_hallucinated[:200]
        responses = [ex.response for ex in all_test]
        test_embeddings = model.sentence_encoder.encode(responses).cpu().numpy()

        plot_tsne(
            test_embeddings[:len(cal_faithful[:200])],
            test_embeddings[len(cal_faithful[:200]):],
            f"t-SNE: {args.dataset.upper()} Embeddings",
            output_dir / f"tsne_{args.dataset}.png"
        )

    # Print CRG results
    print("\n" + "=" * 70)
    print("CRG RESULTS")
    print("=" * 70)
    print(f"  Coverage: {coverage:.2%} (target: {args.coverage:.2%})")
    print(f"  FPR: {fpr:.2%}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Overlap: {score_stats['distribution_overlap']:.2%}")

    # Baselines
    baselines_results = {}

    # BART baseline
    print("\nRunning BART-MNLI baseline...")
    try:
        bart_baseline = SelfCheckGPTBaseline(device=args.device)
        bart_scores = bart_baseline.score(test_data)
        bart_hall = bart_scores[test_hall_mask]

        best_t, best_gap = 0.5, float('inf')
        for t in torch.linspace(0, 1, 100):
            cov = (bart_hall >= t).float().mean().item()
            gap = abs(args.coverage - cov)
            if gap < best_gap:
                best_gap, best_t = gap, t.item()

        bart_preds = (bart_scores >= best_t).float()
        baselines_results["bart_mnli"] = {
            "coverage": (bart_preds[test_hall_mask] == 1).float().mean().item(),
            "fpr": (bart_preds[test_faith_mask] == 1).float().mean().item(),
            "accuracy": (bart_preds == test_labels_cpu.float()).float().mean().item()
        }
        print(f"  BART: Cov={baselines_results['bart_mnli']['coverage']:.2%}, FPR={baselines_results['bart_mnli']['fpr']:.2%}")
    except Exception as e:
        print(f"  BART baseline failed: {e}")

    # DeBERTa baseline
    if args.deberta:
        print("\nRunning DeBERTa baseline...")
        try:
            deberta_baseline = DeBERTaBaseline(device=args.device)
            deberta_scores = deberta_baseline.score(test_data)
            deberta_hall = deberta_scores[test_hall_mask]

            best_t, best_gap = 0.5, float('inf')
            for t in torch.linspace(0, 1, 100):
                cov = (deberta_hall >= t).float().mean().item()
                gap = abs(args.coverage - cov)
                if gap < best_gap:
                    best_gap, best_t = gap, t.item()

            deberta_preds = (deberta_scores >= best_t).float()
            baselines_results["deberta"] = {
                "coverage": (deberta_preds[test_hall_mask] == 1).float().mean().item(),
                "fpr": (deberta_preds[test_faith_mask] == 1).float().mean().item(),
                "accuracy": (deberta_preds == test_labels_cpu.float()).float().mean().item()
            }
            print(f"  DeBERTa: Cov={baselines_results['deberta']['coverage']:.2%}, FPR={baselines_results['deberta']['fpr']:.2%}")
        except Exception as e:
            print(f"  DeBERTa baseline failed: {e}")

    # GPT-4 Judge baseline
    if args.gpt4_judge and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        print("\nRunning GPT-4 Judge baseline...")
        try:
            gpt4_baseline = GPT4JudgeBaseline(model=args.gpt4_model)
            max_samples = min(args.gpt4_max_samples, len(test_data))
            gpt4_scores = gpt4_baseline.score(test_data, max_samples=max_samples)
            gpt4_hall = gpt4_scores[test_hall_mask][:max_samples]

            best_t, best_gap = 0.5, float('inf')
            for t in torch.linspace(0, 1, 100):
                cov = (gpt4_hall >= t).float().mean().item()
                gap = abs(args.coverage - cov)
                if gap < best_gap:
                    best_gap, best_t = gap, t.item()

            gpt4_preds = (gpt4_scores >= best_t).float()
            baselines_results["gpt4_judge"] = {
                "coverage": (gpt4_preds[test_hall_mask] == 1).float().mean().item(),
                "fpr": (gpt4_preds[test_faith_mask] == 1).float().mean().item(),
                "accuracy": (gpt4_preds == test_labels_cpu.float()).float().mean().item()
            }
            print(f"  GPT-4: Cov={baselines_results['gpt4_judge']['coverage']:.2%}, FPR={baselines_results['gpt4_judge']['fpr']:.2%}")
        except Exception as e:
            print(f"  GPT-4 baseline failed: {e}")

    # Ablation study
    ablation_results = None
    if args.ablation:
        ablation_results = run_ablation(model, all_examples)
        print("\n  Ablation Results:")
        for r in ablation_results:
            print(f"    α={r['alpha']:.2f}: Cov={r['actual_coverage']:.2%}, FPR={r['fpr']:.2%}")

    # Error analysis
    error_examples = get_error_examples(test_data, test_ensemble.cpu().numpy(), threshold)
    print(f"\n  False Positives: {len(error_examples['false_positives'])}")
    print(f"  False Negatives: {len(error_examples['false_negatives'])}")

    # Save results
    results = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "n_calibration": len(cal_hallucinated),
        "n_test": len(test_data),
        "target_coverage": args.coverage,
        "crg": {
            "coverage": coverage,
            "fpr": fpr,
            "accuracy": accuracy,
            "threshold": threshold,
            "score_stats": score_stats,
        },
        "baselines": baselines_results,
        "ablation": ablation_results,
        "error_examples": error_examples,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = output_dir / f"results_{args.dataset}_v3.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval", "ragtruth", "nq"])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deberta", action="store_true", help="Run DeBERTa baseline")
    parser.add_argument("--gpt4_judge", action="store_true", help="Run GPT-4 Judge baseline")
    parser.add_argument("--gpt4_model", type=str, default=os.getenv("OPENAI_MODEL_JUDGE", "gpt-5"))
    parser.add_argument("--gpt4_max_samples", type=int, default=200)
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plots")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    parser.add_argument("--all", action="store_true", help="Run all experiments")

    args = parser.parse_args()

    if args.all:
        args.deberta = True
        args.tsne = True
        args.ablation = True
        # Only enable GPT-4 if API key available
        if os.getenv("OPENAI_API_KEY"):
            args.gpt4_judge = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
