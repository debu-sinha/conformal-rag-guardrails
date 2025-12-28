# Conformal RAG Guardrails

Code and experiments for conformal prediction-based hallucination detection in RAG systems.

**Author**: Debu Sinha

---

## Papers

### The Semantic Illusion (arXiv:2512.15068)

**Full Title**: The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems

We demonstrate a fundamental dichotomy: on synthetic hallucinations, embedding methods achieve 95% coverage with 0% FPR; on real RLHF-model hallucinations (HaluEval), the same methods yield 100% FPR.

| Dataset | Method | Coverage | FPR |
|---------|--------|----------|-----|
| Natural Questions | CRG Ensemble | 95.8% | 0% |
| HaluEval | CRG Ensemble | 94.5% | 100% |
| HaluEval | GPT-4o-mini Judge | 67% | 7% |

### ConformalDrift

**Full Title**: ConformalDrift: An Audit Protocol for Testing Conformal Guardrails Under Distribution Shift

Audit protocol for stress-testing conformal guardrails under temporal and cross-dataset shift.

| Shift Type | Coverage | FPR | Finding |
|------------|----------|-----|---------|
| Temporal (FastAPI v1->v2) | 95.2% | 76% | FPR catastrophe |
| Cross-dataset (NQ->RAGTruth) | 11% | - | Coverage collapse |

---

## Repository Structure

```
conformal-rag-guardrails/
├── src/models/
│   ├── core_algorithm.py          # CRG conformal prediction
│   └── hybrid_detector.py         # Hybrid detection pipeline
├── experiments/
│   ├── run_experiment.py          # Main experiment runner
│   ├── run_temporal_shift.py      # ConformalDrift temporal shift
│   ├── run_sota_baselines.py      # SOTA baseline comparisons
│   └── run_deberta_baseline.py    # DeBERTa NLI baseline
├── results/                       # Experiment results (JSON)
├── figures/                       # Score distributions & t-SNE plots
├── tests/                         # Unit tests
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Run main experiment

```bash
python experiments/run_experiment.py --dataset halueval --coverage 0.95
```

### Run temporal shift experiment (ConformalDrift)

```bash
python experiments/run_temporal_shift.py --device cpu
```

---

## Key Results

### Semantic Illusion Finding

Embedding-based hallucination detectors suffer from a fundamental limitation:
- **Synthetic hallucinations**: 95% coverage, 0% FPR
- **RLHF hallucinations**: 95% coverage, 100% FPR

The "Semantic Illusion" - RLHF-trained models produce hallucinations that are semantically indistinguishable from faithful responses in embedding space.

### ConformalDrift Finding

Conformal guarantees fail differently under different shifts:
- **Temporal shift**: Coverage maintained (95.2%) but FPR catastrophic (76%)
- **Cross-dataset shift**: Coverage collapse (95% -> 11%)

---

## Citation

```bibtex
@article{sinha2025semantic,
  title={The Semantic Illusion: Certified Limits of Embedding-Based
         Hallucination Detection in RAG Systems},
  author={Sinha, Debu},
  journal={arXiv preprint arXiv:2512.15068},
  year={2025}
}
```

---

## License

MIT License
