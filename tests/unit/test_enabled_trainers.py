"""
Tests for ENABLED_TRAINERS — the execution-profile switch that lets the local
laptop profile train XGBoost only.

Dependency type: TEST DOUBLE (trainers are mocked; no model is actually trained).

The guarantee under test is honesty: a model that was not trained must never be
presented as if it had been. It must carry model=None, f1=None, trained=False and
a skip reason, and must be excluded from the ensemble weights.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import settings
from src.domain.models import TrainingDataset


def _trainer_with_mocks():
    from src.trainers.ensemble_trainer import EnsembleTrainer

    trainer = EnsembleTrainer.__new__(EnsembleTrainer)
    trainer._transformer_trainer = MagicMock()
    trainer._gnn_trainer = MagicMock()
    trainer._xgb_trainer = MagicMock()
    trainer._transformer_trainer.train.return_value = (MagicMock(), 0.92)
    trainer._gnn_trainer.train.return_value = (MagicMock(), 0.90)
    trainer._xgb_trainer.train.return_value = (MagicMock(), 0.88)
    return trainer


def _dataset():
    dataset = MagicMock(spec=TrainingDataset)
    dataset.feature_matrix = None
    return dataset


class TestEnabledTrainers:
    def test_xgboost_only_does_not_train_the_others(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "xgboost")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            result = trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

        trainer._xgb_trainer.train.assert_called_once()
        trainer._transformer_trainer.train.assert_not_called()
        trainer._gnn_trainer.train.assert_not_called()

    def test_skipped_models_are_marked_not_trained(self, monkeypatch):
        """The critical anti-faking guarantee."""
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "xgboost")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            result = trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

        for name in ("transformer", "gnn"):
            assert result[name]["trained"] is False
            assert result[name]["model"] is None
            # f1 must be None, NOT 0.0 — a zero would read as "trained and scored badly".
            assert result[name]["f1"] is None
            assert "not in ENABLED_TRAINERS" in result[name]["skipped_reason"]

        assert result["xgb"]["trained"] is True
        assert result["xgb"]["f1"] == 0.88

    def test_weights_renormalize_over_trained_models_only(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "xgboost")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            result = trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

        assert result["weights"] == [0.0, 0.0, 1.0]
        assert sum(result["weights"]) == pytest.approx(1.0)
        assert result["is_partial_ensemble"] is True

    def test_two_of_three_renormalizes_proportionally(self, monkeypatch):
        """transformer 0.5 + xgboost 0.2 -> 0.714 / 0.286, preserving their ratio."""
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "transformer,xgboost")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            result = trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

        t, g, x = result["weights"]
        assert g == 0.0
        assert sum(result["weights"]) == pytest.approx(1.0)
        assert t / x == pytest.approx(
            settings.ENSEMBLE_TRANSFORMER_WEIGHT / settings.ENSEMBLE_XGB_WEIGHT
        )

    def test_full_ensemble_is_not_partial_and_keeps_configured_weights(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "transformer,gnn,xgboost")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            result = trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

        assert result["is_partial_ensemble"] is False
        assert result["weights"] == pytest.approx([
            settings.ENSEMBLE_TRANSFORMER_WEIGHT,
            settings.ENSEMBLE_GNN_WEIGHT,
            settings.ENSEMBLE_XGB_WEIGHT,
        ])
        assert all(result[n]["trained"] for n in ("transformer", "gnn", "xgb"))

    def test_no_enabled_trainers_is_an_error(self, monkeypatch):
        """Must fail loudly rather than return an empty 'successful' ensemble."""
        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "")
        trainer = _trainer_with_mocks()

        with patch("src.trainers.ensemble_trainer.AutoTokenizer"):
            with pytest.raises(ValueError, match="No trainers enabled"):
                trainer.train_all(dataset=_dataset(), mlflow_run_id=None, tune_weights=False)

    def test_production_default_is_the_full_ensemble(self):
        """The shipped default must remain the full production ensemble."""
        field_default = type(settings).model_fields["ENABLED_TRAINERS"].default
        assert field_default == "transformer,gnn,xgboost"
