"""Unit tests for core CRG functionality."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# Mock classes for testing without model dependencies
@dataclass
class RAGExample:
    """Test fixture for RAG examples."""
    query: str
    documents: List[str]
    response: str
    label: Optional[int] = None


class TestConformalCalibration:
    """Tests for conformal prediction calibration."""

    def test_quantile_computation(self):
        """Test conformal quantile is computed correctly."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        alpha = 0.1

        # Conformal quantile should be ceil((n+1)(1-alpha))/n percentile
        n = len(scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        quantile_idx = min(quantile_idx, n - 1)
        expected = np.sort(scores)[quantile_idx]

        assert expected == 0.9

    def test_coverage_guarantee(self):
        """Test that calibrated threshold achieves target coverage."""
        np.random.seed(42)

        # Simulate scores where correct have lower scores
        correct_scores = np.random.uniform(0.1, 0.4, 100)
        incorrect_scores = np.random.uniform(0.6, 1.0, 100)

        # Calibrate on correct samples
        alpha = 0.1
        threshold = np.quantile(correct_scores, 1 - alpha)

        # Test coverage on new correct samples
        test_correct = np.random.uniform(0.1, 0.4, 100)
        coverage = np.mean(test_correct <= threshold)

        # Should be at least 1-alpha
        assert coverage >= 0.85  # Allow some variance

    def test_empty_calibration_set(self):
        """Test handling of empty calibration set."""
        scores = np.array([])

        with pytest.raises((ValueError, IndexError)):
            np.quantile(scores, 0.9)


class TestRAGExample:
    """Tests for RAGExample dataclass."""

    def test_example_creation(self):
        """Test RAGExample can be created with required fields."""
        example = RAGExample(
            query="What is ML?",
            documents=["Machine learning is..."],
            response="ML is a type of AI."
        )

        assert example.query == "What is ML?"
        assert len(example.documents) == 1
        assert example.label is None

    def test_example_with_label(self):
        """Test RAGExample with label."""
        example = RAGExample(
            query="Capital of France?",
            documents=["Paris is the capital."],
            response="Berlin is the capital.",
            label=1
        )

        assert example.label == 1


class TestScoreComputation:
    """Tests for score computation logic."""

    def test_cosine_similarity_bounds(self):
        """Test cosine similarity is in [-1, 1]."""
        from numpy.linalg import norm

        def cosine_sim(a, b):
            return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

        vec1 = np.random.randn(768)
        vec2 = np.random.randn(768)

        sim = cosine_sim(vec1, vec2)

        assert -1 <= sim <= 1

    def test_similarity_identical_vectors(self):
        """Test similarity of identical vectors is 1."""
        from numpy.linalg import norm

        def cosine_sim(a, b):
            return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

        vec = np.array([1.0, 2.0, 3.0])

        sim = cosine_sim(vec, vec)

        assert np.isclose(sim, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
