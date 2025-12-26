# The Semantic Illusion

**Full Title**: The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems

**Author**: Debu Sinha

---

## Abstract

We apply conformal prediction to RAG hallucination detection, demonstrating a fundamental dichotomy: on synthetic hallucinations, embedding methods achieve 95% coverage with 0% FPR; on real RLHF-model hallucinations (HaluEval), the same methods yield 100% FPR. GPT-4 as judge achieves 7% FPR, proving the task is solvable via reasoning but opaque to surface-level semantics.

---

## Key Results

| Dataset | Method | Coverage | FPR |
|---------|--------|----------|-----|
| Natural Questions | CRG Ensemble | 95.8% | 0% |
| HaluEval | CRG Ensemble | 94.5% | 100% |
| HaluEval | GPT-4o-mini Judge | 67% | 7% |

---

## Folder Structure

```
semantic_illusion/
├── paper/
│   ├── semantic_illusion.tex    # LaTeX source
│   ├── semantic_illusion.pdf    # Compiled paper
│   └── *.png                    # Figures (t-SNE, distributions)
├── code/
│   ├── crg_core.py              # Core CRG algorithm
│   ├── crg_hybrid_detector.py   # Hybrid detection pipeline
│   └── verify_claims.py         # Claims verification script
├── results/
│   ├── results_with_ci.json     # Main results with CIs
│   ├── deberta_roc_results.json # DeBERTa AUC results
│   └── claim_ledger.json        # Curated claims
├── submission/
│   └── arxiv_submission.zip     # ArXiv package
├── CONFERENCE_TARGETS.md        # Venue recommendations
├── SUBMISSION_NOTES.md          # Submission checklist
└── README.md                    # This file
```

---

## Verification Status

- **Claims verified**: 13/13 (100%)
- **Run**: `python code/verify_claims.py`

---

## Target Venues

1. **TMLR** (Primary) - Submit Jan 6-15, 2026
2. **NeurIPS 2026** (Backup) - ~May 2026

See `CONFERENCE_TARGETS.md` for details.

---

## Links

- **Code**: https://github.com/debu-sinha/conformal-rag-guardrails
