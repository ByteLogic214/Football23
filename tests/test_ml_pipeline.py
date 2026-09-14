"""tests/test_ml_pipeline.py — pruebas unitarias e integración del modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.ml_pipeline import (CLASSES, MatchOutcomeModel,
                                build_model_matrix, validate_probabilities)
from engine.thestats_features import MODEL_FEATURES


# --------------------------------------------------------------------------- #
# Dimensiones y rangos de datos
# --------------------------------------------------------------------------- #
def test_model_matrix_dimensiones(model_matrix):
    X, y, meta = model_matrix
    assert X.shape[1] == 2 * len(MODEL_FEATURES), (
        f"Se esperaban 2 x {len(MODEL_FEATURES)} columnas, hay {X.shape[1]}")
    assert len(X) == len(y) == len(meta)
    assert set(np.unique(y)).issubset({0, 1, 2})
    assert not pd.to_datetime(meta["date"], utc=True).isna().any()


def test_model_matrix_orden_temporal(model_matrix):
    _, _, meta = model_matrix
    assert pd.to_datetime(meta["date"], utc=True).is_monotonic_increasing


def test_target_corresponde_al_marcador(team_frame):
    X, y, meta = build_model_matrix(team_frame)
    homes = team_frame[team_frame.venue == "home"].set_index("match_id")
    gf = homes.loc[meta["match_id"], "goals_for"].to_numpy()
    ga = homes.loc[meta["match_id"], "goals_against"].to_numpy()
    esperado = np.select([gf > ga, gf == ga], [0, 1], default=2)
    assert np.array_equal(y, esperado)


# --------------------------------------------------------------------------- #
# Validación cruzada temporal: sin fuga hacia el futuro
# --------------------------------------------------------------------------- #
def test_timeseries_sin_fuga_de_futuro(model_matrix):
    X, y, meta = model_matrix
    model = MatchOutcomeModel(n_splits=4, backend="lightgbm")
    Xs, ys, ds = model.sort_by_date(X, y, meta["date"])
    for tr, te in model.time_splits(Xs, ys, ds):
        assert tr.max() < te.min()
        assert ds.iloc[tr].max() < ds.iloc[te].min()
        assert len(np.intersect1d(tr, te)) == 0


def test_sort_by_date_reordena(model_matrix):
    X, y, meta = model_matrix
    model = MatchOutcomeModel(backend="lightgbm")
    Xs, ys, ds = model.sort_by_date(X.iloc[::-1], y[::-1],
                                    meta["date"].iloc[::-1])
    assert ds.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Contrato de probabilidades [0, 1]
# --------------------------------------------------------------------------- #
def test_validate_probabilities_rechaza_rangos_invalidos():
    for mal in (np.array([[1.2, 0.0, 0.0]]),           # > 1
                np.array([[-0.1, 0.6, 0.5]]),          # < 0
                np.array([[0.5, 0.5, 0.5]]),           # no suma 1
                np.array([[np.nan, 0.5, 0.5]])):       # no finito
        try:
            validate_probabilities(mal)
            raise AssertionError(f"Debió rechazar {mal}")
        except ValueError:
            pass
    ok = validate_probabilities(np.array([[0.7, 0.2, 0.1]]))
    assert ok.shape == (1, 3)


def test_predecir_sin_entrenar_lanza():
    model = MatchOutcomeModel(backend="lightgbm")
    try:
        model.predict_proba(np.zeros((2, 6)))
        raise AssertionError("Debió lanzar RuntimeError")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# Integración end-to-end (rápida: malla 'fast', folds reducidos)
# --------------------------------------------------------------------------- #
def _fast_model() -> MatchOutcomeModel:
    return MatchOutcomeModel(backend="lightgbm", n_splits=3, hpo="grid",
                             search_space="fast", holdout_frac=0.2,
                             random_state=42)


def test_end_to_end_entrena_y_reporta(model_matrix, tmp_path):
    X, y, meta = model_matrix
    model = _fast_model().fit(X, y, meta["date"])
    rep = model.report_

    assert model.model_ is not None and model.best_params_ is not None
    assert np.array(rep["confusion_matrix"]).shape == (3, 3)
    assert len(rep["feature_importance"]) == X.shape[1]
    assert 0.0 <= rep["holdout"]["accuracy"] <= 1.0
    assert 0.0 < rep["holdout"]["log_loss"] < np.log(3) + 0.5  # mejor que ruido puro
    assert np.isfinite(rep["holdout"]["brier"])
    assert rep["n_train"] + rep["n_holdout"] == len(X)
    # probabilidades del holdout dentro de [0,1] y normalizadas
    proba = validate_probabilities(model.predict_proba(
        X.iloc[-rep["n_holdout"]:]))
    assert proba.shape == (rep["n_holdout"], 3)


def test_modelo_supera_al_azar_en_datos_estructurados(model_matrix):
    """Con datos sintéticos con señal real (fuerzas latentes), la accuracy
    debe superar claramente el azar multinomial (~33%)."""
    X, y, meta = model_matrix
    model = _fast_model().fit(X, y, meta["date"])
    assert model.report_["holdout"]["accuracy"] > 0.45


def test_persistencia_predicciones_identicas(model_matrix, tmp_path):
    X, y, meta = model_matrix
    model = _fast_model().fit(X, y, meta["date"])
    path = tmp_path / "model.joblib"
    model.save(path)
    cargado = MatchOutcomeModel.load(path)
    np.testing.assert_allclose(cargado.predict_proba(X.tail(10)),
                               model.predict_proba(X.tail(10)), atol=1e-8)


def test_report_json_escribible(model_matrix, tmp_path):
    import json
    X, y, meta = model_matrix
    model = _fast_model().fit(X, y, meta["date"])
    out = tmp_path / "report.json"
    model.write_report(out)
    data = json.loads(out.read_text())
    assert set(data) >= {"confusion_matrix", "feature_importance",
                         "holdout", "cv_time_series", "best_params"}
