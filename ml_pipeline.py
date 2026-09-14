"""
engine/ml_pipeline.py
=====================
Pipeline ML para predicción de resultado de partido (Victoria local /
Empate / Victoria visitante) sobre la matriz de features de
engine/thestats_features.py.

- Backend de gradient boosting con auto-detección: XGBoost > LightGBM >
  HistGradientBoosting (los tres gestionan NaN de forma nativa, lo cual
  respeta la política de no imputación del pipeline de features).
- HPO: Optuna (TPE, minimiza LogLoss) con GridSearch temporal como
  fallback. NUNCA se usa validación aleatoria: TimeSeriesSplit puro.
- Reporte: LogLoss / Accuracy / Brier en holdout temporal, matriz de
  confusión, métricas por fold e importancia (gain) de variables.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from engine.thestats_features import MODEL_FEATURES

logger = logging.getLogger(__name__)

CLASSES = np.array([0, 1, 2])          # 0=home win, 1=draw, 2=away win
CLASS_NAMES = ["home_win", "draw", "away_win"]

# Espacios de hiperparámetros (diccionario común; se mapea a cada backend)
PARAM_SETS_FULL: list[dict[str, Any]] = [
    {"learning_rate": 0.03, "n_estimators": 600, "max_depth": 4,
     "num_leaves": 31, "min_child_samples": 20, "subsample": 0.9,
     "colsample_bytree": 0.9, "reg_lambda": 1.0},
    {"learning_rate": 0.05, "n_estimators": 400, "max_depth": 5,
     "num_leaves": 31, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_lambda": 3.0},
    {"learning_rate": 0.05, "n_estimators": 400, "max_depth": 6,
     "num_leaves": 63, "min_child_samples": 20, "subsample": 1.0,
     "colsample_bytree": 0.9, "reg_lambda": 1.0},
    {"learning_rate": 0.08, "n_estimators": 300, "max_depth": 4,
     "num_leaves": 31, "min_child_samples": 40, "subsample": 0.9,
     "colsample_bytree": 1.0, "reg_lambda": 5.0},
    {"learning_rate": 0.03, "n_estimators": 600, "max_depth": 7,
     "num_leaves": 63, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.9, "reg_lambda": 3.0},
    {"learning_rate": 0.05, "n_estimators": 500, "max_depth": 5,
     "num_leaves": 47, "min_child_samples": 30, "subsample": 0.9,
     "colsample_bytree": 0.8, "reg_lambda": 5.0},
]
PARAM_SETS_FAST: list[dict[str, Any]] = PARAM_SETS_FULL[:2]  # para tests/CI


# --------------------------------------------------------------------------- #
# Matriz de modelado a nivel PARTIDO (home_/away_ ensanchados + target)
# --------------------------------------------------------------------------- #
def build_model_matrix(team_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Convierte el frame largo equipo-partido en matriz de modelado:
    una fila por partido, columnas home_{feat} / away_{feat} para cada
    feature de MODEL_FEATURES, más target desde la perspectiva del local
    (0=victoria local, 1=empate, 2=victoria visitante).
    """
    required = {"match_id", "date", "venue", "team_id",
                "goals_for", "goals_against"}
    missing = required - set(team_frame.columns)
    if missing:
        raise ValueError(f"team_frame sin columnas requeridas: {missing}")

    homes = team_frame[team_frame["venue"] == "home"].set_index("match_id")
    aways = team_frame[team_frame["venue"] == "away"].set_index("match_id")
    common = homes.index.intersection(aways.index)
    if len(common) == 0:
        raise ValueError("Sin partidos con ambos lados presentes")

    feats = [f for f in MODEL_FEATURES if f in team_frame.columns]
    h = homes.loc[common, feats].add_prefix("home_")
    a = aways.loc[common, feats].add_prefix("away_")
    X = pd.concat([h, a], axis=1)

    meta = pd.DataFrame({
        "match_id": common,
        "date": pd.to_datetime(homes.loc[common, "date"].values, utc=True),
        "home_team_id": homes.loc[common, "team_id"].values,
        "away_team_id": aways.loc[common, "team_id"].values,
    }).sort_values("date").reset_index(drop=True)
    X = X.loc[meta["match_id"]].reset_index(drop=True)

    gf = homes.loc[common, "goals_for"].astype(float)
    ga = homes.loc[common, "goals_against"].astype(float)
    y = pd.Series(np.select(
        [gf > ga, gf == ga, gf < ga], [0, 1, 2], default=-1
    ), index=common)
    y = y.loc[meta["match_id"]].to_numpy()

    valid = y >= 0
    if not valid.all():
        logger.warning("Descartados %d partidos sin marcador real", (~valid).sum())
    return X[valid], y[valid], meta[valid].reset_index(drop=True)


def validate_probabilities(proba: Any, n_classes: int = 3) -> np.ndarray:
    """Contrato de salida del modelo: shape (n, k), valores en [0, 1],
    filas que suman 1. Lanza ValueError si se viola."""
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"Shape inválido {p.shape}; esperaba (n, {n_classes})")
    if not np.isfinite(p).all():
        raise ValueError("Probabilidades con valores no finitos")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("Probabilidades fuera del rango [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Las probabilidades no suman 1 por fila")
    return p


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class MatchOutcomeModel:
    """
    Clasificador 3-clases con validación temporal estricta.

    Uso:
        model = MatchOutcomeModel(backend="auto", n_splits=5,
                                  hpo="auto", n_trials=30)
        model.fit(X, y, meta["date"])
        proba = model.predict_proba(X_new)      # (n, 3) validado en [0,1]
        model.write_report("report.json")
    """

    def __init__(
        self,
        backend: str = "auto",           # "auto" | "xgboost" | "lightgbm" | "sklearn"
        n_splits: int = 5,
        hpo: str = "auto",               # "auto" | "optuna" | "grid"
        n_trials: int = 30,
        holdout_frac: float = 0.2,
        search_space: str = "full",      # "full" | "fast" (tests/CI)
        random_state: int = 42,
    ) -> None:
        self.backend = self._resolve_backend(backend)
        self.n_splits = n_splits
        self.hpo = hpo
        self.n_trials = n_trials
        self.holdout_frac = holdout_frac
        self.search_space = search_space
        self.random_state = random_state
        self.model_: Any = None
        self.best_params_: Optional[dict[str, Any]] = None
        self.report_: dict[str, Any] = {}

    # -- backend ----------------------------------------------------------- #
    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            # Los extras son opcionales. El fallback garantiza que la CI base
            # es reproducible sin convertir un ImportError en fallo tardío.
            modules = {"xgboost": "xgboost", "lightgbm": "lightgbm", "sklearn": "sklearn"}
            if backend not in modules:
                raise ValueError(f"Backend no soportado: {backend}")
            try:
                __import__(modules[backend])
                return backend
            except ImportError:
                logger.warning("%s no instalado; se usará sklearn", backend)
                return "sklearn"
        for candidate, module in (("xgboost", "xgboost"),
                                  ("lightgbm", "lightgbm"),
                                  ("sklearn", "sklearn")):
            try:
                __import__(module)
                return candidate
            except ImportError:
                continue
        raise ImportError("Ningún backend de boosting disponible")

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        common = dict(params)
        if self.backend == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                objective="multi:softprob", num_class=3,
                max_depth=int(common["max_depth"]),
                learning_rate=common["learning_rate"],
                n_estimators=int(common["n_estimators"]),
                min_child_weight=common["min_child_samples"],
                subsample=common["subsample"],
                colsample_bytree=common["colsample_bytree"],
                reg_lambda=common["reg_lambda"],
                eval_metric="mlogloss", tree_method="hist",
                random_state=self.random_state, n_jobs=4)
        if self.backend == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                objective="multiclass", num_class=3,
                num_leaves=int(common["num_leaves"]),
                learning_rate=common["learning_rate"],
                n_estimators=int(common["n_estimators"]),
                min_child_samples=int(common["min_child_samples"]),
                subsample=common["subsample"],
                colsample_bytree=common["colsample_bytree"],
                reg_lambda=common["reg_lambda"],
                verbosity=-1, random_state=self.random_state, n_jobs=4)
        # sklearn HistGradientBoosting (fallback, NaN nativo)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            loss="log_loss", max_iter=int(common["n_estimators"]) // 10,
            learning_rate=common["learning_rate"],
            max_leaf_nodes=int(common["num_leaves"]),
            l2_regularization=common["reg_lambda"],
            random_state=self.random_state, early_stopping=False)

    # -- orden temporal y splits ------------------------------------------- #
    @staticmethod
    def sort_by_date(X: pd.DataFrame, y: np.ndarray,
                     dates: pd.Series) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        d = pd.to_datetime(dates, utc=True)
        if not d.is_monotonic_increasing:
            order = np.argsort(d.values, kind="stable")
            X, y, d = X.iloc[order], y[order], d.iloc[order]
        return X.reset_index(drop=True), np.asarray(y), d.reset_index(drop=True)

    def time_splits(self, X: pd.DataFrame, y: np.ndarray,
                    dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """TimeSeriesSplit sobre datos ORDENADOS por fecha. Garantiza que
        todo el train es estrictamente anterior al test (sin fuga)."""
        d = pd.to_datetime(dates, utc=True)
        if not d.is_monotonic_increasing:
            raise ValueError("Los datos deben estar ordenados por fecha")
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in splitter.split(X):
            yield train_idx, test_idx

    # -- validación cruzada temporal ---------------------------------------- #
    def _cv_score(self, params: dict[str, Any], X: pd.DataFrame,
                  y: np.ndarray, dates: pd.Series) -> dict[str, float]:
        losses, accs = [], []
        for tr, te in self.time_splits(X, y, dates):
            est = self._make_estimator(params)
            sw = compute_sample_weight("balanced", y[tr])
            est.fit(X.iloc[tr], y[tr], sample_weight=sw)
            proba = validate_probabilities(est.predict_proba(X.iloc[te]))
            losses.append(log_loss(y[te], proba, labels=CLASSES))
            accs.append(accuracy_score(y[te], proba.argmax(axis=1)))
        return {"log_loss": float(np.mean(losses)),
                "accuracy": float(np.mean(accs))}

    # -- búsqueda de hiperparámetros ---------------------------------------- #
    def _hyperparameter_search(self, X: pd.DataFrame, y: np.ndarray,
                               dates: pd.Series) -> dict[str, Any]:
        history: list[dict[str, Any]] = []

        def _record(params: dict[str, Any]) -> float:
            score = self._cv_score(params, X, y, dates)
            history.append({"params": params, **score})
            logger.info("HPO logloss=%.4f acc=%.3f %s", score["log_loss"],
                        score["accuracy"], params)
            return score["log_loss"]

        use_optuna = self.hpo in ("auto", "optuna")
        if use_optuna:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial: Any) -> float:
                    base = PARAM_SETS_FULL[0]
                    params = {
                        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                        "max_depth": trial.suggest_int("max_depth", 3, 8),
                        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    }
                    _ = base
                    return _record(params)

                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
                study.optimize(objective, n_trials=self.n_trials,
                               show_progress_bar=False)
                self.search_history_ = history
                return dict(study.best_params)
            except ImportError:
                if self.hpo == "optuna":
                    raise

        # Fallback: GridSearch sobre malla predefinida con TimeSeriesSplit
        grid = (PARAM_SETS_FAST if self.search_space == "fast"
                else PARAM_SETS_FULL)
        best, best_loss = None, np.inf
        for params in grid:
            loss = _record(params)
            if loss < best_loss:
                best, best_loss = params, loss
        self.search_history_ = history
        if best is None:
            raise RuntimeError("GridSearch vacío: no hay combinaciones")
        return dict(best)

    # -- entrenamiento ------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            dates: pd.Series) -> "MatchOutcomeModel":
        X, y, d = self.sort_by_date(X, y, dates)
        n = len(X)
        n_holdout = max(self.n_splits, int(round(n * self.holdout_frac)))
        if n - n_holdout < self.n_splits * 3:
            raise ValueError(
                f"Muestra insuficiente para {self.n_splits} folds "
                f"temporales con holdout (n={n})")

        X_tr, y_tr, d_tr = X.iloc[:-n_holdout], y[:-n_holdout], d.iloc[:-n_holdout]
        X_ho, y_ho, d_ho = X.iloc[-n_holdout:], y[-n_holdout:], d.iloc[-n_holdout:]

        self.best_params_ = self._hyperparameter_search(X_tr, y_tr, d_tr)
        self.model_ = self._make_estimator(self.best_params_)
        sw = compute_sample_weight("balanced", y_tr)
        self.model_.fit(X_tr, y_tr, sample_weight=sw)

        proba = self.predict_proba(X_ho)
        pred = proba.argmax(axis=1)
        self.report_ = self._build_report(X_tr, y_tr, d_tr, X_ho, y_ho, d_ho,
                                          proba, pred)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("El modelo no está entrenado (fit)")
        X_in = X if isinstance(X, pd.DataFrame) else np.asarray(X)
        return validate_probabilities(self.model_.predict_proba(X_in))

    # -- reporte ------------------------------------------------------------ #
    def _feature_importance(self, X_ho: pd.DataFrame, y_ho: np.ndarray) -> dict[str, float]:
        names = list(X_ho.columns)
        if hasattr(self.model_, "booster_"):  # LightGBM: importancia por ganancia
            gains = self.model_.booster_.feature_importance("gain")
        elif hasattr(self.model_, "feature_importances_"):
            gains = self.model_.feature_importances_
        else:
            gains = np.zeros(len(names))
        gains = np.asarray(gains, dtype=float)
        if gains.sum() > 0:
            gains = gains / gains.sum()
        return {name: float(g) for name, g in zip(names, gains)}

    def _build_report(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
                      d_tr: pd.Series, X_ho: pd.DataFrame, y_ho: np.ndarray,
                      d_ho_dates: pd.Series,
                      proba: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        cm = confusion_matrix(y_ho, pred, labels=CLASSES)
        onehot = np.eye(3)[y_ho]
        cv_best = self._cv_score(self.best_params_, X_tr, y_tr, d_tr)
        return {
            "backend": self.backend,
            "n_classes": 3,
            "class_names": CLASS_NAMES,
            "n_train": int(len(X_tr)),
            "n_holdout": int(len(X_ho)),
            "train_period": [str(d_tr.iloc[0].date()), str(d_tr.iloc[-1].date())],
            "holdout_period": [str(d_ho_dates.iloc[0].date()),
                               str(d_ho_dates.iloc[-1].date())],
            "best_params": self.best_params_,
            "cv_time_series": cv_best,
            "holdout": {
                "log_loss": float(log_loss(y_ho, proba, labels=CLASSES)),
                "accuracy": float(accuracy_score(y_ho, pred)),
                "brier": float(np.mean((proba - onehot) ** 2)),
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_ho, pred, labels=CLASSES, target_names=CLASS_NAMES,
                output_dict=True, zero_division=0),
            "feature_importance": self._feature_importance(X_ho, y_ho),
            "search_history": getattr(self, "search_history_", []),
        }

    def write_report(self, path: str | Path) -> None:
        def _default(o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            raise TypeError(str(type(o)))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.report_, fh, indent=2, default=_default)
        logger.info("Reporte escrito en %s", path)

    # -- persistencia ------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MatchOutcomeModel":
        return joblib.load(path)
