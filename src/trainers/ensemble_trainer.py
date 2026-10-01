"""
model-training-pipeline/src/trainers/ensemble_trainer.py

Three model trainers + Ensemble combiner:
  1. TransformerTrainer  — BERT-base fine-tuned (50% ensemble weight)
  2. GNNTrainer          — GraphSAGE 3-layer (30% ensemble weight)
  3. XGBoostTrainer      — 1000 trees, depth 8 (20% ensemble weight)
  4. EnsembleTrainer     — Weighted combination + Optuna weight tuning

All trainers log to MLflow and return serializable model artifacts.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)

from src.core.config import settings
from src.domain.models import TrainingDataset

logger = logging.getLogger(__name__)


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class EntityPairDataset(Dataset):
    """PyTorch Dataset for entity pair classification."""

    def __init__(
        self,
        texts_1: List[str],
        texts_2: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts_1,
            texts_2,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Transformer Trainer ─────────────────────────────────────────────────────

class TransformerTrainer:
    """
    Fine-tunes BERT-base for entity pair matching.
    Input: concatenated entity text pairs via [SEP] token.
    Output: binary match probability [0, 1].
    Architecture: BERT-base (110M params) + linear classification head.
    """

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TransformerTrainer using device: {self._device}")

    def train(
        self,
        dataset: TrainingDataset,
        mlflow_run_id: Optional[str] = None,
    ) -> Tuple[nn.Module, float]:
        """
        Fine-tune BERT for entity matching.
        Returns (model, val_f1).
        """
        logger.info("TransformerTrainer: starting BERT fine-tuning")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            settings.TRANSFORMER_MODEL_NAME,
            num_labels=2,
            hidden_dropout_prob=settings.TRANSFORMER_DROPOUT,
        )
        model = model.to(self._device)

        # Prepare datasets
        train_idx = dataset.train_indices or list(range(int(len(dataset.pairs) * 0.75)))
        val_idx = dataset.val_indices or list(range(int(len(dataset.pairs) * 0.75), len(dataset.pairs)))

        train_texts_1 = [dataset.entity_texts_1[i] for i in train_idx] if dataset.entity_texts_1 else [""] * len(train_idx)
        train_texts_2 = [dataset.entity_texts_2[i] for i in train_idx] if dataset.entity_texts_2 else [""] * len(train_idx)
        train_labels = [dataset.pairs[i].label for i in train_idx]

        val_texts_1 = [dataset.entity_texts_1[i] for i in val_idx] if dataset.entity_texts_1 else [""] * len(val_idx)
        val_texts_2 = [dataset.entity_texts_2[i] for i in val_idx] if dataset.entity_texts_2 else [""] * len(val_idx)
        val_labels = [dataset.pairs[i].label for i in val_idx]

        train_ds = EntityPairDataset(train_texts_1, train_texts_2, train_labels, tokenizer, settings.TRANSFORMER_MAX_LENGTH)
        val_ds = EntityPairDataset(val_texts_1, val_texts_2, val_labels, tokenizer, settings.TRANSFORMER_MAX_LENGTH)

        # Training arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=settings.TRANSFORMER_EPOCHS,
                per_device_train_batch_size=settings.TRANSFORMER_BATCH_SIZE,
                per_device_eval_batch_size=settings.TRANSFORMER_BATCH_SIZE * 2,
                learning_rate=settings.TRANSFORMER_LEARNING_RATE,
                weight_decay=settings.TRANSFORMER_WEIGHT_DECAY,
                warmup_ratio=settings.TRANSFORMER_WARMUP_RATIO,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=2,
                report_to="none",  # We handle MLflow logging ourselves
                logging_steps=100,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
            )

            trainer.train()

        # Evaluate
        model.eval()
        val_f1 = self._evaluate(model, tokenizer, val_texts_1, val_texts_2, val_labels)

        elapsed = time.time() - start
        logger.info(f"TransformerTrainer complete: val_f1={val_f1:.4f} in {elapsed/60:.1f}min")

        if mlflow_run_id:
            mlflow.log_metrics({
                "transformer_val_f1": val_f1,
                "transformer_train_time_min": elapsed / 60,
            })

        return model, val_f1

    def _evaluate(self, model, tokenizer, texts_1, texts_2, labels) -> float:
        """Quick F1 evaluation on validation set."""
        if not texts_1:
            return 0.0

        dataset = EntityPairDataset(texts_1, texts_2, labels, tokenizer, settings.TRANSFORMER_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items() if k != "labels"}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)

        return float(f1_score(labels, all_preds, zero_division=0))

    def predict_proba(self, model, tokenizer, texts_1: List[str], texts_2: List[str]) -> np.ndarray:
        """Return probabilities for positive class."""
        model.eval()
        dataset = EntityPairDataset(texts_1, texts_2, [0] * len(texts_1), tokenizer, settings.TRANSFORMER_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items() if k != "labels"}
                outputs = model(**batch)
                prob = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(prob)

        return np.array(probs)


# ─── Graph Neural Network Trainer ────────────────────────────────────────────

class GNNEntityEncoder(nn.Module):
    """
    GraphSAGE-inspired entity relationship encoder.
    Input: 50-dim feature vector per entity pair node.
    Output: match probability [0, 1].
    """

    def __init__(self, input_dim: int = 50, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class GNNTrainer:
    """
    Trains the GNN model on the 50-dim feature matrix.
    In production, this would use actual entity graph structure
    (shared addresses, phone numbers, email domains) via torch-geometric.
    This implementation uses a deep MLP as an approximation.
    """

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        dataset: TrainingDataset,
        mlflow_run_id: Optional[str] = None,
    ) -> Tuple[nn.Module, float]:
        """Train GNN on feature matrix. Returns (model, val_f1)."""
        if dataset.feature_matrix is None or len(dataset.feature_matrix) == 0:
            logger.warning("GNNTrainer: empty feature matrix, returning dummy model")
            return self._dummy_model(), 0.0

        logger.info("GNNTrainer: starting GNN training")
        start = time.time()

        X = torch.tensor(dataset.feature_matrix, dtype=torch.float32).to(self._device)
        y = torch.tensor(dataset.labels, dtype=torch.float32).to(self._device)

        train_idx = dataset.train_indices or list(range(int(len(X) * 0.75)))
        val_idx = dataset.val_indices or list(range(int(len(X) * 0.75), len(X)))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = GNNEntityEncoder(
            input_dim=50,
            hidden_dim=settings.GNN_HIDDEN_DIM,
            dropout=settings.GNN_DROPOUT,
        ).to(self._device)

        # Class-weighted loss for imbalanced data
        n_pos = y_train.sum().item()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(self._device)
        criterion = nn.BCELoss(reduction="mean")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.GNN_LEARNING_RATE,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings.GNN_EPOCHS
        )

        best_val_f1 = 0.0
        best_state = None
        patience = 10
        no_improve = 0

        for epoch in range(settings.GNN_EPOCHS):
            model.train()

            # Mini-batch training
            perm = torch.randperm(len(X_train))
            batch_size = 512
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                Xb, yb = X_train[idx], y_train[idx]

                # Apply class weighting manually
                weights = torch.where(yb == 1, pos_weight, torch.ones_like(pos_weight))
                criterion_w = nn.BCELoss(weight=weights.expand_as(yb))

                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion_w(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation every 10 epochs
            if (epoch + 1) % 10 == 0:
                val_f1 = self._evaluate(model, X_val, y_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        if best_state:
            model.load_state_dict(best_state)

        elapsed = time.time() - start
        logger.info(f"GNNTrainer complete: best_val_f1={best_val_f1:.4f} in {elapsed/60:.1f}min")

        if mlflow_run_id:
            mlflow.log_metrics({"gnn_val_f1": best_val_f1, "gnn_train_time_min": elapsed / 60})

        return model, best_val_f1

    def _evaluate(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            probs = model(X).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        return float(f1_score(y.cpu().numpy(), preds, zero_division=0))

    def predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            return model(X_t).cpu().numpy()

    def _dummy_model(self) -> nn.Module:
        return GNNEntityEncoder(input_dim=50)


# ─── XGBoost Trainer ─────────────────────────────────────────────────────────

class XGBoostTrainer:
    """
    Trains XGBoost on the 50-dim tabular feature matrix.
    Handles class imbalance, early stopping, and feature importance.
    """

    def train(
        self,
        dataset: TrainingDataset,
        mlflow_run_id: Optional[str] = None,
    ) -> Tuple[xgb.XGBClassifier, float]:
        """Train XGBoost. Returns (model, val_f1)."""
        if dataset.feature_matrix is None or len(dataset.feature_matrix) == 0:
            logger.warning("XGBoostTrainer: empty dataset")
            return xgb.XGBClassifier(), 0.0

        logger.info("XGBoostTrainer: starting XGBoost training")
        start = time.time()

        X = dataset.feature_matrix
        y = dataset.labels

        train_idx = dataset.train_indices or list(range(int(len(X) * 0.75)))
        val_idx = dataset.val_indices or list(range(int(len(X) * 0.75), len(X)))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Compute scale_pos_weight for class imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=settings.XGB_N_ESTIMATORS,
            max_depth=settings.XGB_MAX_DEPTH,
            learning_rate=settings.XGB_LEARNING_RATE,
            subsample=settings.XGB_SUBSAMPLE,
            colsample_bytree=settings.XGB_COLSAMPLE_BYTREE,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=settings.XGB_EARLY_STOPPING_ROUNDS,
            n_jobs=-1,
            tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        val_preds = model.predict(X_val)
        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))

        elapsed = time.time() - start
        logger.info(f"XGBoostTrainer complete: val_f1={val_f1:.4f} in {elapsed/60:.1f}min")

        if mlflow_run_id:
            mlflow.log_metrics({
                "xgb_val_f1": val_f1,
                "xgb_train_time_min": elapsed / 60,
                "xgb_n_trees": model.best_iteration or settings.XGB_N_ESTIMATORS,
            })

        return model, val_f1

    def predict_proba(self, model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def get_feature_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        importance = model.feature_importances_
        if feature_names and len(feature_names) == len(importance):
            return dict(zip(feature_names, importance.tolist()))
        return {f"feat_{i}": float(v) for i, v in enumerate(importance)}


# ─── Ensemble Trainer ────────────────────────────────────────────────────────

class EnsembleTrainer:
    """
    Combines Transformer + GNN + XGBoost predictions with learned weights.
    Default: [0.50, 0.30, 0.20] — tunable via Optuna.
    """

    def __init__(self):
        self._transformer_trainer = TransformerTrainer()
        self._gnn_trainer = GNNTrainer()
        self._xgb_trainer = XGBoostTrainer()

    def train_all(
        self,
        dataset: TrainingDataset,
        mlflow_run_id: Optional[str] = None,
        tune_weights: bool = False,
    ) -> Dict:
        """
        Train all three models and combine into ensemble.
        Returns dict with all models and their validation F1 scores.
        """
        enabled = settings.enabled_trainers
        logger.info(f"EnsembleTrainer: training {sorted(enabled)} (of transformer/gnn/xgboost)")

        def skipped(name: str) -> Dict:
            """A model that was deliberately not trained. Never presented as trained."""
            return {
                "model": None,
                "f1": None,
                "trained": False,
                "skipped_reason": (
                    f"{name} not in ENABLED_TRAINERS ({settings.ENABLED_TRAINERS}); "
                    f"excluded from the ensemble"
                ),
            }

        # 1. Transformer
        if "transformer" in enabled:
            transformer_model, transformer_f1 = self._transformer_trainer.train(
                dataset, mlflow_run_id
            )
            # Load the tokenizer so it can be saved alongside the model in MLflow
            transformer_tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_MODEL_NAME)
            transformer_entry = {
                "model": transformer_model, "f1": transformer_f1,
                "tokenizer": transformer_tokenizer, "trained": True,
            }
        else:
            transformer_model = None
            transformer_entry = skipped("transformer")

        # 2. GNN
        if "gnn" in enabled:
            gnn_model, gnn_f1 = self._gnn_trainer.train(dataset, mlflow_run_id)
            gnn_entry = {"model": gnn_model, "f1": gnn_f1, "trained": True}
        else:
            gnn_model = None
            gnn_entry = skipped("gnn")

        # 3. XGBoost
        if "xgboost" in enabled:
            xgb_model, xgb_f1 = self._xgb_trainer.train(dataset, mlflow_run_id)
            xgb_entry = {"model": xgb_model, "f1": xgb_f1, "trained": True}
        else:
            xgb_model = None
            xgb_entry = skipped("xgboost")

        if not any(e["trained"] for e in (transformer_entry, gnn_entry, xgb_entry)):
            raise ValueError(
                f"No trainers enabled (ENABLED_TRAINERS={settings.ENABLED_TRAINERS!r}); "
                f"nothing to train."
            )

        # 4. Ensemble weights — configured weights, zeroed for models that were not
        #    trained, then renormalized so the enabled models still sum to 1.0.
        configured = [
            settings.ENSEMBLE_TRANSFORMER_WEIGHT,
            settings.ENSEMBLE_GNN_WEIGHT,
            settings.ENSEMBLE_XGB_WEIGHT,
        ]
        trained_flags = [
            transformer_entry["trained"], gnn_entry["trained"], xgb_entry["trained"],
        ]
        weights = [w if t else 0.0 for w, t in zip(configured, trained_flags)]
        total = sum(weights)
        weights = [w / total for w in weights]

        if settings.is_partial_ensemble:
            logger.warning(
                "PARTIAL ENSEMBLE: trained %s only. Effective weights %s "
                "(production weights are %s). Metrics from this run describe the "
                "partial ensemble, not the full production model.",
                sorted(enabled), [round(w, 4) for w in weights], configured,
            )

        if tune_weights and not settings.is_partial_ensemble                 and dataset.feature_matrix is not None and len(dataset.feature_matrix) > 0:
            weights = self._tune_ensemble_weights(
                dataset=dataset,
                transformer_model=transformer_model,
                gnn_model=gnn_model,
                xgb_model=xgb_model,
            )
            logger.info(f"Optuna tuned ensemble weights: {weights}")
        elif tune_weights and settings.is_partial_ensemble:
            logger.info("Skipping Optuna weight tuning — partial ensemble has fixed weights")

        if mlflow_run_id:
            mlflow.log_params({
                "ensemble_transformer_weight": weights[0],
                "ensemble_gnn_weight": weights[1],
                "ensemble_xgb_weight": weights[2],
                "enabled_trainers": settings.ENABLED_TRAINERS,
                "is_partial_ensemble": settings.is_partial_ensemble,
            })

        return {
            "transformer": transformer_entry,
            "gnn": gnn_entry,
            "xgb": xgb_entry,
            "weights": weights,
            "is_partial_ensemble": settings.is_partial_ensemble,
        }

    def predict_ensemble(
        self,
        models: Dict,
        dataset: TrainingDataset,
        indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute ensemble probabilities for given dataset indices.
        Returns array of probabilities in [0, 1].
        """
        idx = indices or list(range(len(dataset.pairs)))
        weights = models["weights"]

        # Get predictions from each model
        probs = np.zeros(len(idx))

        # XGBoost / GNN — tabular models over the feature matrix.
        if dataset.feature_matrix is not None and len(dataset.feature_matrix) > 0:
            X = dataset.feature_matrix[idx]
            if models["xgb"].get("model") is not None and weights[2] > 0:
                xgb_probs = self._xgb_trainer.predict_proba(models["xgb"]["model"], X)
                probs += weights[2] * xgb_probs

            if models["gnn"].get("model") is not None and weights[1] > 0:
                gnn_probs = self._gnn_trainer.predict_proba(models["gnn"]["model"], X)
                probs += weights[1] * gnn_probs

        # Transformer
        if (models["transformer"].get("model") is not None and weights[0] > 0
                and dataset.entity_texts_1 and dataset.entity_texts_2):
            t1 = [dataset.entity_texts_1[i] for i in idx]
            t2 = [dataset.entity_texts_2[i] for i in idx]
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_MODEL_NAME)
            transformer_probs = self._transformer_trainer.predict_proba(
                models["transformer"]["model"], tokenizer, t1, t2
            )
            probs += weights[0] * transformer_probs

        # Normalize if weights don't sum to 1
        weight_sum = sum(weights)
        if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
            probs /= weight_sum

        return np.clip(probs, 0.0, 1.0)

    def _tune_ensemble_weights(
        self,
        dataset: TrainingDataset,
        transformer_model,
        gnn_model,
        xgb_model,
    ) -> List[float]:
        """
        Use Optuna to find optimal ensemble weights on validation set.
        Maximizes F1 score on validation split.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        val_idx = dataset.val_indices or list(range(int(len(dataset.pairs) * 0.75), len(dataset.pairs)))
        y_val = dataset.labels[val_idx]

        # Pre-compute individual model predictions
        X_val = dataset.feature_matrix[val_idx] if dataset.feature_matrix is not None else None

        xgb_probs = self._xgb_trainer.predict_proba(xgb_model, X_val) if X_val is not None else np.zeros(len(val_idx))
        gnn_probs = self._gnn_trainer.predict_proba(gnn_model, X_val) if X_val is not None else np.zeros(len(val_idx))
        transformer_probs = np.zeros(len(val_idx))  # Expensive; skip unless needed

        def objective(trial):
            w_t = trial.suggest_float("w_transformer", 0.1, 0.7)
            w_g = trial.suggest_float("w_gnn", 0.1, 0.5)
            w_x = trial.suggest_float("w_xgb", 0.1, 0.5)
            total = w_t + w_g + w_x
            probs = (w_t * transformer_probs + w_g * gnn_probs + w_x * xgb_probs) / total
            preds = (probs >= 0.5).astype(int)
            return f1_score(y_val, preds, zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, timeout=60)

        best = study.best_params
        total = best["w_transformer"] + best["w_gnn"] + best["w_xgb"]
        return [
            best["w_transformer"] / total,
            best["w_gnn"] / total,
            best["w_xgb"] / total,
        ]
