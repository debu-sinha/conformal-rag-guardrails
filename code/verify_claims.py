"""
Claims Verification Script for Semantic Illusion Paper
Verifies paper claims against experimental evidence files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = Path(r"C:\Users\dsinh\gemini-projects\vanguard\topics\conformal_rag_guardrails\results")

# Tolerance for numeric comparisons
TOLERANCE = 0.02  # 2% tolerance


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def verify_numeric(claimed: float, evidence: float, tolerance: float = TOLERANCE) -> Tuple[bool, float]:
    """Verify a numeric claim within tolerance."""
    discrepancy = abs(claimed - evidence)
    verified = discrepancy <= tolerance
    return verified, discrepancy


def verify_ci(claimed_ci: List[float], evidence_ci: List[float], tolerance: float = TOLERANCE) -> bool:
    """Verify confidence interval claim."""
    if len(claimed_ci) != 2 or len(evidence_ci) != 2:
        return False
    lower_ok = abs(claimed_ci[0] - evidence_ci[0]) <= tolerance
    upper_ok = abs(claimed_ci[1] - evidence_ci[1]) <= tolerance
    return lower_ok and upper_ok


class ClaimsVerifier:
    """Verify paper claims against experimental results."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results = self._load_all_results()
        self.verification_results = []

    def _load_all_results(self) -> Dict[str, Any]:
        """Load all result files."""
        results = {}

        # Main results with CI
        results_ci_path = self.results_dir / "results_with_ci.json"
        if results_ci_path.exists():
            results['main'] = load_json(results_ci_path)

        # DeBERTa results
        deberta_path = self.results_dir / "deberta_results.json"
        if deberta_path.exists():
            results['deberta'] = load_json(deberta_path)

        # DeBERTa ROC results (contains AUC)
        deberta_roc_path = self.results_dir / "deberta_roc_results.json"
        if deberta_roc_path.exists():
            results['deberta_roc'] = load_json(deberta_roc_path)

        # Hybrid experiment
        hybrid_path = self.results_dir / "hybrid_experiment_results.json"
        if hybrid_path.exists():
            results['hybrid'] = load_json(hybrid_path)

        # SOTA baselines
        sota_path = self.results_dir / "sota_baselines_results.json"
        if sota_path.exists():
            results['sota'] = load_json(sota_path)

        # Claim ledger (manually curated)
        claim_ledger_path = self.results_dir / "claim_ledger.json"
        if claim_ledger_path.exists():
            results['claim_ledger'] = load_json(claim_ledger_path)

        return results

    def verify_claim(self, claim_id: str, claim_text: str,
                     metric: str, dataset: str,
                     claimed_value: Any,
                     evidence_path: str,
                     notes: str = "") -> Dict:
        """Verify a single claim."""
        result = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "metric": metric,
            "dataset": dataset,
            "claimed_value": claimed_value,
            "evidence_path": evidence_path,
            "notes": notes
        }

        # Get evidence value
        evidence_value = self._get_evidence(metric, dataset, evidence_path)

        if evidence_value is None:
            result["status"] = "NO_EVIDENCE"
            result["evidence_value"] = None
            result["discrepancy"] = None
        elif isinstance(claimed_value, (int, float)):
            verified, discrepancy = verify_numeric(claimed_value, evidence_value)
            result["status"] = "VERIFIED" if verified else "MISMATCH"
            result["evidence_value"] = evidence_value
            result["discrepancy"] = discrepancy
        elif isinstance(claimed_value, list):
            # Confidence interval
            verified = verify_ci(claimed_value, evidence_value)
            result["status"] = "VERIFIED" if verified else "MISMATCH"
            result["evidence_value"] = evidence_value
            result["discrepancy"] = None
        else:
            result["status"] = "UNKNOWN_TYPE"
            result["evidence_value"] = evidence_value
            result["discrepancy"] = None

        self.verification_results.append(result)
        return result

    def _get_evidence(self, metric: str, dataset: str, evidence_path: str) -> Optional[Any]:
        """Extract evidence value from results."""
        try:
            parts = evidence_path.split('.')
            data = self.results
            for part in parts:
                if isinstance(data, dict):
                    data = data.get(part)
                elif isinstance(data, list) and part.isdigit():
                    data = data[int(part)]
                else:
                    return None
                if data is None:
                    return None
            return data
        except (KeyError, IndexError, TypeError):
            return None

    def verify_all_claims(self) -> List[Dict]:
        """Verify all paper claims."""

        # ===== MAIN CLAIMS =====

        # Claim 1: NQ coverage 95.8%
        self.verify_claim(
            "C1", "Natural Questions achieves 95.8% coverage",
            "coverage", "nq", 0.958,
            "main.nq.crg.coverage",
            "Table 1 main result"
        )

        # Claim 2: NQ FPR 0%
        self.verify_claim(
            "C2", "Natural Questions achieves 0% FPR",
            "fpr", "nq", 0.0,
            "main.nq.crg.fpr",
            "Table 1 main result"
        )

        # Claim 3: NQ FPR CI [0%, 1%]
        self.verify_claim(
            "C3", "NQ FPR 95% CI: [0%, 1%]",
            "fpr_ci", "nq", [0.0, 0.01],
            "main.nq.crg.fpr_ci",
            "Table 1 confidence interval"
        )

        # Claim 4: HaluEval coverage ~95%
        self.verify_claim(
            "C4", "HaluEval achieves ~95% coverage",
            "coverage", "halueval", 0.945,
            "main.halueval.crg.coverage",
            "Table 1 main result"
        )

        # Claim 5: HaluEval FPR 100%
        self.verify_claim(
            "C5", "HaluEval FPR 100%",
            "fpr", "halueval", 1.0,
            "main.halueval.crg.fpr",
            "Table 1 - the key negative result"
        )

        # Claim 6: HaluEval FPR CI [99%, 100%]
        self.verify_claim(
            "C6", "HaluEval FPR 95% CI: [99%, 100%]",
            "fpr_ci", "halueval", [0.99, 1.0],
            "main.halueval.crg.fpr_ci",
            "Table 1 confidence interval"
        )

        # ===== GPT-4 JUDGE CLAIMS =====

        # Claim 7: GPT-4o-mini achieves 7% FPR
        self.verify_claim(
            "C7", "GPT-4o-mini Judge achieves 7% FPR on HaluEval",
            "fpr", "halueval", 0.07,
            "main.gpt4_judge.halueval.fpr_corrected",
            "Section 4.5 - The Solution"
        )

        # Claim 8: GPT-4 FPR CI [3.4%, 13.7%]
        self.verify_claim(
            "C8", "GPT-4o-mini FPR 95% CI: [3.4%, 13.7%]",
            "fpr_ci", "halueval", [0.034, 0.137],
            "main.gpt4_judge.halueval.fpr_ci",
            "Section 4.5 confidence interval"
        )

        # ===== DEBERTA CLAIMS =====

        # Claim 9: DeBERTa AUC 0.81 (from deberta_roc_results.json)
        self.verify_claim(
            "C9", "DeBERTa achieves AUC 0.81",
            "auc", "halueval", 0.81,
            "deberta_roc.auc",
            "Section 4.4 - The DeBERTa Paradox"
        )

        # Claim 10: Hallucinated mean score 0.415
        self.verify_claim(
            "C10", "Hallucinated responses have mean entailment score 0.415",
            "mean_score", "halueval", 0.415,
            "deberta.hallucinated_scores.mean",
            "Table 2"
        )

        # Claim 11: Faithful mean score 0.933
        self.verify_claim(
            "C11", "Faithful responses have mean entailment score 0.933",
            "mean_score", "halueval", 0.933,
            "deberta.faithful_scores.mean",
            "Table 2"
        )

        # ===== CALIBRATION SET SIZE ABLATION =====
        # Note: n=300 ablation in paper uses data from claim_ledger.json (manually verified)

        # Claim 12: n=300 achieves 96.2% coverage (from claim_ledger ablation study)
        self.verify_claim(
            "C12", "Calibration set n=300 achieves 96.2% coverage",
            "coverage", "nq", 0.962,
            "claim_ledger.results.ablation_studies.0.results.0.coverage",
            "Table 3 - Ablation study (from claim_ledger)"
        )

        # Claim 13: n=600 achieves 95.8% coverage
        self.verify_claim(
            "C13", "Calibration set n=600 achieves 95.8% coverage",
            "coverage", "nq", 0.958,
            "main.nq.crg.coverage",
            "Table 3 - matches main result"
        )

        return self.verification_results

    def generate_report(self) -> Dict:
        """Generate verification report."""
        if not self.verification_results:
            self.verify_all_claims()

        total = len(self.verification_results)
        verified = sum(1 for r in self.verification_results if r["status"] == "VERIFIED")
        mismatches = sum(1 for r in self.verification_results if r["status"] == "MISMATCH")
        no_evidence = sum(1 for r in self.verification_results if r["status"] == "NO_EVIDENCE")

        report = {
            "total_claims": total,
            "verified": verified,
            "mismatches": mismatches,
            "no_evidence": no_evidence,
            "score": verified / total if total > 0 else 0,
            "results": self.verification_results
        }

        return report

    def print_report(self):
        """Print formatted verification report."""
        report = self.generate_report()

        print("\n" + "="*60)
        print("CLAIMS VERIFICATION REPORT")
        print("="*60)
        print(f"\nTotal Claims: {report['total_claims']}")
        print(f"Verified:     {report['verified']} ({report['verified']/report['total_claims']*100:.1f}%)")
        print(f"Mismatches:   {report['mismatches']}")
        print(f"No Evidence:  {report['no_evidence']}")
        print("\n" + "-"*60)

        for r in report['results']:
            status_icon = "OK" if r["status"] == "VERIFIED" else "FAIL" if r["status"] == "MISMATCH" else "??"
            print(f"\n[{status_icon}] {r['claim_id']}: {r['claim_text']}")
            print(f"    Claimed: {r['claimed_value']}")
            print(f"    Evidence: {r['evidence_value']}")
            if r['discrepancy'] is not None:
                print(f"    Discrepancy: {r['discrepancy']:.4f}")
            print(f"    Source: {r['evidence_path']}")

        print("\n" + "="*60)


def main():
    """Run verification."""
    verifier = ClaimsVerifier()
    verifier.print_report()

    # Save report
    report = verifier.generate_report()
    output_path = RESULTS_DIR / "claims_verification_report_v2.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
