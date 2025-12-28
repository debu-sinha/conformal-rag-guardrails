"""
Hybrid Detector for Semantic Illusion Problem

A simple, dependency-light approach that combines:
1. Existing CRG scores (embedding-based)
2. Sentence-level contradiction detection
3. Entity/number mismatch detection (rule-based)

No external dependencies beyond what's already installed.
"""
import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

from .core_algorithm import (
    CRGConfig,
    RAGExample,
    ConformalRAGGuardrails,
    NLIModel
)

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid detector."""
    # Weights for combining scores
    crg_weight: float = 0.3
    contradiction_weight: float = 0.4
    entity_weight: float = 0.3

    # Thresholds
    contradiction_threshold: float = 0.7
    entity_mismatch_threshold: float = 0.3


class EntityExtractor:
    """Simple rule-based entity/number extractor (no spacy needed)."""

    # Patterns for extracting entities
    NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|hundred|%|percent))?\b', re.IGNORECASE)
    DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b', re.IGNORECASE)
    CAPITALIZED_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        return {
            'numbers': self.NUMBER_PATTERN.findall(text),
            'dates': self.DATE_PATTERN.findall(text),
            'named_entities': self.CAPITALIZED_PATTERN.findall(text)
        }

    def compute_overlap(self, response_entities: Dict, doc_entities: Dict) -> float:
        """Compute entity overlap between response and documents."""
        total_response = 0
        matched = 0

        for entity_type in ['numbers', 'dates', 'named_entities']:
            resp_set = set(str(e).lower() for e in response_entities.get(entity_type, []))
            doc_set = set(str(e).lower() for e in doc_entities.get(entity_type, []))

            total_response += len(resp_set)
            matched += len(resp_set & doc_set)

        if total_response == 0:
            return 1.0  # No entities to check

        return matched / total_response


class ContradictionDetector:
    """Detects contradictions using NLI model."""

    def __init__(self, config: CRGConfig):
        self.nli_model = NLIModel(config)
        self.config = config

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @torch.no_grad()
    def detect_contradictions(self, response: str, documents: List[str]) -> Tuple[float, List[Dict]]:
        """
        Detect contradictions between response and documents.

        Returns:
            contradiction_score: 0-1 (higher = more contradictions)
            details: List of contradicting sentences
        """
        sentences = self._split_sentences(response)
        if not sentences:
            return 0.0, []

        doc_text = " ".join(documents)
        contradictions = []

        for sent in sentences:
            # Check if sentence contradicts any document
            # NLI labels: 0=contradiction, 1=neutral, 2=entailment
            inputs = self.nli_model.tokenizer(
                [doc_text[:512]],
                [sent],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(self.nli_model.device)
            attention_mask = inputs["attention_mask"].to(self.nli_model.device)

            outputs = self.nli_model.model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)

            # Get contradiction probability (label 0 for BART-MNLI)
            contradiction_prob = probs[0, 0].item()

            if contradiction_prob > 0.5:
                contradictions.append({
                    'sentence': sent,
                    'contradiction_prob': contradiction_prob
                })

        contradiction_score = len(contradictions) / len(sentences) if sentences else 0
        return contradiction_score, contradictions


class HybridDetector:
    """
    Hybrid hallucination detector combining multiple signals.

    The key insight: embeddings fail because hallucinations are semantically similar.
    We add:
    1. Contradiction detection (NLI)
    2. Entity mismatch detection (rule-based)
    """

    def __init__(self, config: CRGConfig = None, hybrid_config: HybridConfig = None):
        self.config = config or CRGConfig()
        self.hybrid_config = hybrid_config or HybridConfig()

        # Initialize components
        self.crg = ConformalRAGGuardrails(self.config)
        self.contradiction_detector = ContradictionDetector(self.config)
        self.entity_extractor = EntityExtractor()

        self._calibrated = False
        self._threshold = None

    def compute_hybrid_score(self, example: RAGExample) -> Dict[str, float]:
        """
        Compute hybrid score combining multiple signals.

        Returns dict with individual scores and combined score.
        """
        scores = {}

        # 1. CRG score (embedding-based)
        crg_scores = self.crg.compute_individual_scores([example])
        crg_ensemble = self.crg.compute_ensemble_score(crg_scores)[0].item()
        scores['crg'] = crg_ensemble

        # 2. Contradiction score
        contradiction_score, _ = self.contradiction_detector.detect_contradictions(
            example.response, example.documents
        )
        scores['contradiction'] = contradiction_score

        # 3. Entity mismatch score
        response_entities = self.entity_extractor.extract(example.response)
        doc_entities = self.entity_extractor.extract(" ".join(example.documents))
        entity_overlap = self.entity_extractor.compute_overlap(response_entities, doc_entities)
        scores['entity_mismatch'] = 1 - entity_overlap  # Higher = more mismatch

        # Combined score (weighted)
        combined = (
            self.hybrid_config.crg_weight * scores['crg'] +
            self.hybrid_config.contradiction_weight * scores['contradiction'] +
            self.hybrid_config.entity_weight * scores['entity_mismatch']
        )
        scores['combined'] = combined

        return scores

    def score(self, example: RAGExample) -> float:
        """Get single combined score for an example."""
        return self.compute_hybrid_score(example)['combined']

    def score_batch(self, examples: List[RAGExample]) -> torch.Tensor:
        """Score a batch of examples."""
        scores = []
        for ex in examples:
            scores.append(self.score(ex))
        return torch.tensor(scores)

    def calibrate(self, calibration_examples: List[RAGExample], alpha: float = 0.05):
        """
        Calibrate threshold using conformal prediction.

        Args:
            calibration_examples: Examples with label=1 (hallucinations)
            alpha: Target miscoverage rate
        """
        # Compute scores on calibration set
        cal_scores = self.score_batch(calibration_examples)

        # Find threshold
        n = len(cal_scores)
        quantile_idx = int(np.ceil((n + 1) * alpha))
        quantile_idx = min(quantile_idx, n) - 1

        sorted_scores = torch.sort(cal_scores)[0]
        self._threshold = sorted_scores[quantile_idx].item()
        self._calibrated = True

        logger.info(f"Hybrid detector calibrated. Threshold: {self._threshold:.4f}")
        return self._threshold

    def predict(self, examples: List[RAGExample]) -> torch.Tensor:
        """
        Predict hallucinations.

        Returns:
            Tensor of predictions (1 = hallucination, 0 = faithful)
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        scores = self.score_batch(examples)
        return (scores >= self._threshold).float()

    def evaluate(self, test_examples: List[RAGExample]) -> Dict[str, float]:
        """
        Evaluate on test set.

        Returns metrics dict with coverage, FPR, accuracy.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before evaluate()")

        predictions = self.predict(test_examples)
        labels = torch.tensor([ex.label for ex in test_examples])

        hall_mask = labels == 1
        faith_mask = labels == 0

        coverage = (predictions[hall_mask] == 1).float().mean().item() if hall_mask.sum() > 0 else 0
        fpr = (predictions[faith_mask] == 1).float().mean().item() if faith_mask.sum() > 0 else 0
        accuracy = (predictions == labels.float()).float().mean().item()

        return {
            'coverage': coverage,
            'fpr': fpr,
            'accuracy': accuracy,
            'threshold': self._threshold
        }
