import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Tensor = torch.Tensor
Device = Union[str, torch.device]


@dataclass
class RAGExample:
    query: str
    documents: List[str]
    response: str
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.documents) > 0
        if self.label is not None:
            assert self.label in {0, 1}


@dataclass
class CRGConfig:
    sentence_encoder_name: str = "BAAI/bge-base-en-v1.5"
    nli_model_name: str = "facebook/bart-large-mnli"
    grounding_threshold: float = 0.3
    temperature: float = 0.1
    max_seq_length: int = 512
    use_weighted_rad: bool = True
    use_sentence_level_sec: bool = True
    ensemble_weights: Optional[List[float]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.ensemble_weights is not None:
            assert len(self.ensemble_weights) == 3
            total = sum(self.ensemble_weights)
            self.ensemble_weights = [w / total for w in self.ensemble_weights]


class NonconformityScore(ABC):
    @abstractmethod
    def compute(self, example: RAGExample) -> float:
        pass

    @abstractmethod
    def compute_batch(self, examples: List[RAGExample]) -> Tensor:
        pass


class SentenceEncoder(nn.Module):
    def __init__(self, config: CRGConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._encoder = None
        self._tokenizer = None

    def _load_model(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading sentence encoder: {self.config.sentence_encoder_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.sentence_encoder_name)
            self._encoder = AutoModel.from_pretrained(self.config.sentence_encoder_name)

            if self.config.gradient_checkpointing:
                self._encoder.gradient_checkpointing_enable()

            self._encoder = self._encoder.to(self.device)

            if self.config.use_fp16 and self.device.type == "cuda":
                self._encoder = self._encoder.half()

            self._encoder.eval()

    @property
    def encoder(self):
        self._load_model()
        return self._encoder

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    def mean_pooling(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts: List[str], normalize: bool = True) -> Tensor:
        if not texts:
            raise ValueError("Cannot encode empty list")

        self._load_model()

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if self.config.gradient_checkpointing and self.training:
            outputs = checkpoint(
                self._encoder,
                input_ids,
                attention_mask=attention_mask,
                use_reentrant=False
            )
        else:
            outputs = self._encoder(input_ids, attention_mask=attention_mask)

        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class NLIModel(nn.Module):
    def __init__(self, config: CRGConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading NLI model: {self.config.nli_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.config.nli_model_name)

            if self.config.gradient_checkpointing:
                if hasattr(self._model, 'gradient_checkpointing_enable'):
                    self._model.gradient_checkpointing_enable()

            self._model = self._model.to(self.device)

            if self.config.use_fp16 and self.device.type == "cuda":
                self._model = self._model.half()

            self._model.eval()

            self._entailment_idx = 2
            if hasattr(self._model.config, 'label2id'):
                label2id = self._model.config.label2id
                for key in ['entailment', 'ENTAILMENT', 'Entailment']:
                    if key in label2id:
                        self._entailment_idx = label2id[key]
                        break

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    @property
    def entailment_idx(self):
        self._load_model()
        return self._entailment_idx

    @torch.no_grad()
    def compute_entailment_prob(self, premises: List[str], hypotheses: List[str]) -> Tensor:
        assert len(premises) == len(hypotheses)

        if not premises:
            return torch.tensor([], device=self.device)

        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        return probs[:, self.entailment_idx]


class RetrievalAttributionDivergence(NonconformityScore):
    def __init__(self, encoder: SentenceEncoder, config: CRGConfig):
        self.encoder = encoder
        self.config = config

    def compute(self, example: RAGExample) -> float:
        return self.compute_batch([example])[0].item()

    def compute_batch(self, examples: List[RAGExample]) -> Tensor:
        if not examples:
            return torch.tensor([], device=self.encoder.device)

        scores = []

        for ex in examples:
            response_emb = self.encoder.encode([ex.response])
            doc_embs = self.encoder.encode(ex.documents)
            similarities = torch.matmul(response_emb, doc_embs.T).squeeze(0)

            # Calculate cosine similarity between response and each document
            response_emb_np = response_emb.cpu().numpy()
            doc_embs_np = doc_embs.cpu().numpy()
            cosine_similarities = cosine_similarity(response_emb_np, doc_embs_np).flatten()

            if self.config.use_weighted_rad:
                query_emb = self.encoder.encode([ex.query])
                query_doc_sim = torch.matmul(query_emb, doc_embs.T).squeeze(0)
                weights = F.softmax(query_doc_sim / self.config.temperature, dim=0)
                weighted_sim = torch.sum(weights * similarities)
                score = 1.0 - weighted_sim
            else:
                score = 1.0 - similarities.max()

            # Integrate cosine similarity into the final score
            combined_score = 0.5 * score + 0.5 * (1.0 - cosine_similarities.max())
            scores.append(torch.tensor(combined_score, device=self.encoder.device).clamp(0.0, 1.0))

        return torch.stack(scores)


class SemanticEntailmentCalibration(NonconformityScore):
    def __init__(self, nli_model: NLIModel, config: CRGConfig):
        self.nli_model = nli_model
        self.config = config

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences if sentences else [text]

    def compute(self, example: RAGExample) -> float:
        return self.compute_batch([example])[0].item()

    def compute_batch(self, examples: List[RAGExample]) -> Tensor:
        if not examples:
            return torch.tensor([], device=self.nli_model.device)

        scores = []

        for ex in examples:
            if self.config.use_sentence_level_sec:
                sentences = self._split_sentences(ex.response)
                sentence_scores = []

                for sent in sentences:
                    premises = ex.documents
                    hypotheses = [sent] * len(ex.documents)
                    entail_probs = self.nli_model.compute_entailment_prob(premises, hypotheses)
                    max_entail = entail_probs.max()
                    sentence_scores.append(1.0 - max_entail)

                score = torch.stack(sentence_scores).max()
            else:
                premises = ex.documents
                hypotheses = [ex.response] * len(ex.documents)
                entail_probs = self.nli_model.compute_entailment_prob(premises, hypotheses)
                score = 1.0 - entail_probs.max()

            scores.append(score.clamp(0.0, 1.0))

        return torch.stack(scores)


class TokenLevelFactualGrounding(NonconformityScore):
    def __init__(self, encoder: SentenceEncoder, config: CRGConfig):
        self.encoder = encoder
        self.config = config

    def _get_token_embeddings(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        self.encoder._load_model()

        encoded = self.encoder._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.encoder.device)
        attention_mask = encoded["attention_mask"].to(self.encoder.device)

        with torch.no_grad():
            outputs = self.encoder._encoder(input_ids, attention_mask=attention_mask)

        token_embs = outputs.last_hidden_state
        token_embs = F.normalize(token_embs, p=2, dim=-1)
        lengths = attention_mask.sum(dim=1)

        return token_embs, lengths, attention_mask

    def compute(self, example: RAGExample) -> float:
        return self.compute_batch([example])[0].item()

    def compute_batch(self, examples: List[RAGExample]) -> Tensor:
        if not examples:
            return torch.tensor([], device=self.encoder.device)

        scores = []

        for ex in examples:
            resp_embs, resp_lens, resp_mask = self._get_token_embeddings([ex.response])
            resp_embs = resp_embs[0]
            resp_len = resp_lens[0].item()
            resp_mask = resp_mask[0]

            doc_embs, doc_lens, doc_masks = self._get_token_embeddings(ex.documents)

            grounding_scores = []

            for t_idx in range(resp_len):
                if resp_mask[t_idx] == 0:
                    continue

                t_emb = resp_embs[t_idx]
                max_sim = 0.0

                for d_idx, (d_emb, d_mask) in enumerate(zip(doc_embs, doc_masks)):
                    sims = torch.matmul(t_emb, d_emb.T)
                    sims = sims * d_mask.float()
                    max_sim = max(max_sim, sims.max().item())

                grounding_scores.append(max_sim)

            if not grounding_scores:
                scores.append(torch.tensor(1.0, device=self.encoder.device))
                continue

            grounding_tensor = torch.tensor(grounding_scores, device=self.encoder.device)
            poorly_grounded = (grounding_tensor < self.config.grounding_threshold).float()
            scores.append(poorly_grounded.mean())

        return torch.stack(scores)


class ConformalRAGGuardrails(nn.Module):
    def __init__(self, config: Optional[CRGConfig] = None):
        super().__init__()
        self.config = config or CRGConfig()

        self.sentence_encoder = SentenceEncoder(self.config)
        self.nli_model = NLIModel(self.config)

        self.rad_scorer = RetrievalAttributionDivergence(self.sentence_encoder, self.config)
        self.sec_scorer = SemanticEntailmentCalibration(self.nli_model, self.config)
        self.tfg_scorer = TokenLevelFactualGrounding(self.sentence_encoder, self.config)

        if self.config.ensemble_weights:
            self.ensemble_weights = torch.tensor(self.config.ensemble_weights, dtype=torch.float32)
            self._weights_fixed = True
        else:
            self.ensemble_weights = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
            self._weights_fixed = False

        self._calibrated = False
        self._calibration_scores: Optional[Tensor] = None
        self._threshold: Optional[float] = None
        self._alpha: Optional[float] = None

    def compute_individual_scores(self, examples: List[RAGExample]) -> Dict[str, Tensor]:
        logger.info(f"Computing nonconformity scores for {len(examples)} examples...")

        rad_scores = self.rad_scorer.compute_batch(examples)
        logger.info("  RAD scores computed")

        sec_scores = self.sec_scorer.compute_batch(examples)
        logger.info("  SEC scores computed")

        tfg_scores = self.tfg_scorer.compute_batch(examples)
        logger.info("  TFG scores computed")

        return {'rad': rad_scores, 'sec': sec_scores, 'tfg': tfg_scores}

    def compute_ensemble_score(self, individual_scores: Dict[str, Tensor]) -> Tensor:
        rad = individual_scores['rad']
        sec = individual_scores['sec']
        tfg = individual_scores['tfg']

        stacked = torch.stack([rad, sec, tfg], dim=1)
        weights = self.ensemble_weights.to(stacked.device)
        return torch.matmul(stacked, weights)

    def learn_ensemble_weights(self, examples: List[RAGExample], labels: List[int]) -> Tensor:
        from sklearn.metrics import roc_auc_score

        scores = self.compute_individual_scores(examples)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        best_auroc = 0.0
        best_weights = [1/3, 1/3, 1/3]

        for w1 in np.arange(0.1, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.8 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0.1:
                    continue

                weights = torch.tensor([w1, w2, w3], dtype=torch.float32)
                stacked = torch.stack([scores['rad'], scores['sec'], scores['tfg']], dim=1)
                ensemble = torch.matmul(stacked, weights.to(stacked.device))

                try:
                    auroc = roc_auc_score(labels_tensor.cpu().numpy(), ensemble.cpu().numpy())
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = [w1, w2, w3]
                except:
                    pass

        logger.info(f"Learned weights: RAD={best_weights[0]:.2f}, SEC={best_weights[1]:.2f}, TFG={best_weights[2]:.2f}")
        self.ensemble_weights = torch.tensor(best_weights, dtype=torch.float32)
        return self.ensemble_weights

    def calibrate(self, calibration_examples: List[RAGExample], alpha: float = 0.05) -> float:
        assert 0 < alpha < 1

        logger.info(f"Calibrating on {len(calibration_examples)} examples at alpha={alpha}")

        individual_scores = self.compute_individual_scores(calibration_examples)
        calibration_scores = self.compute_ensemble_score(individual_scores)

        self._calibration_scores = calibration_scores
        self._alpha = alpha

        n = len(calibration_scores)
        quantile_idx = int(np.ceil((n + 1) * alpha))
        quantile_idx = min(quantile_idx, n) - 1

        sorted_scores = torch.sort(calibration_scores)[0]
        self._threshold = sorted_scores[quantile_idx].item()

        self._calibrated = True

        logger.info(f"Calibration complete. Threshold: {self._threshold:.4f}")

        return self._threshold

    def predict(self, examples: List[RAGExample], return_scores: bool = False) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        individual_scores = self.compute_individual_scores(examples)
        ensemble_scores = self.compute_ensemble_score(individual_scores)
        predictions = (ensemble_scores >= self._threshold).float()

        if return_scores:
            return predictions, {'ensemble': ensemble_scores, **individual_scores}
        return predictions

    def evaluate(self, test_examples: List[RAGExample], labels: List[int]) -> Dict[str, float]:
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before evaluate()")

        predictions, scores = self.predict(test_examples, return_scores=True)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        hallucinated_mask = labels_tensor == 1
        faithful_mask = labels_tensor == 0

        if hallucinated_mask.sum() > 0:
            coverage = (predictions[hallucinated_mask] == 1).float().mean().item()
        else:
            coverage = float('nan')

        if faithful_mask.sum() > 0:
            fpr = (predictions[faithful_mask] == 1).float().mean().item()
        else:
            fpr = float('nan')

        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(labels, scores['ensemble'].cpu().numpy())
        except:
            auroc = float('nan')

        return {
            'coverage': coverage,
            'coverage_gap': (1 - self._alpha) - coverage,
            'false_positive_rate': fpr,
            'flag_rate': predictions.mean().item(),
            'auroc': auroc,
            'threshold': self._threshold,
        }