"""Features auditables: nunca imputan cobertura ausente ni usan el futuro."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

class DataIntegrityError(ValueError): pass
MODEL_FEATURES = ["goals_for", "goals_against", "points", "xg_for", "xg_against", "total_shots_for", "total_shots_against", "shots_on_target_for", "shots_on_target_against", "key_passes_for", "key_passes_against", "conversion_rate", "shot_accuracy", "roll_goals_for", "roll_goals_against", "roll_points", "rest_days"]
_OVERVIEW = {"expected_goals": "xg", "total_shots": "total_shots", "shots_on_target": "shots_on_target", "corner_kicks": "corner_kicks", "fouls": "fouls", "passes": "passes", "accurate_passes": "accurate_passes", "ball_possession": "possession", "big_chances": "big_chances"}

def _side(stat: dict[str, Any] | None, side: str) -> float:
    try: return float(stat["all"][side])
    except (KeyError, TypeError, ValueError): return np.nan

def match_to_team_rows(match: dict[str, Any], stats: dict[str, Any] | None = None, player_stats: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    required = {"id", "utc_date", "home_team", "away_team", "score"}
    if required - match.keys(): raise DataIntegrityError(f"Partido incompleto: faltan {required - match.keys()}")
    overview = (stats or {}).get("overview", {})
    players = player_stats or [] ; rows = []
    for side, other in (("home", "away"), ("away", "home")):
        team, opponent = match[f"{side}_team"], match[f"{other}_team"]
        score, against = match["score"].get(side), match["score"].get(other)
        if score is None or against is None: raise DataIntegrityError("Marcador incompleto")
        row: dict[str, Any] = {"match_id": match["id"], "date": pd.to_datetime(match["utc_date"], utc=True), "team_id": team["id"], "opponent_id": opponent["id"], "venue": side, "goals_for": float(score), "goals_against": float(against), "points": float(3 if score > against else 1 if score == against else 0)}
        for source, name in _OVERVIEW.items(): row[f"{name}_for"], row[f"{name}_against"] = _side(overview.get(source), side), _side(overview.get(source), other)
        own = [p for p in players if p.get("team_id") == team["id"] and p.get("played", True)]
        opp = [p for p in players if p.get("team_id") == opponent["id"] and p.get("played", True)]
        row["key_passes_for"] = sum(float(p.get("passing", {}).get("key_passes", 0) or 0) for p in own) if player_stats is not None else np.nan
        row["key_passes_against"] = sum(float(p.get("passing", {}).get("key_passes", 0) or 0) for p in opp) if player_stats is not None else np.nan
        rows.append(row)
    return rows

def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(); out["conversion_rate"] = out["goals_for"] / out["total_shots_for"].replace(0, np.nan); out["shot_accuracy"] = out["shots_on_target_for"] / out["total_shots_for"].replace(0, np.nan); return out

def add_rolling_features(frame: pd.DataFrame, window: int = 5, min_periods: int = 3, expanding_min_periods: int = 8) -> pd.DataFrame:
    out = frame.sort_values(["team_id", "date", "match_id"]).copy()
    for col in ("goals_for", "goals_against", "points", "xg_for"):
        out[f"roll_{col}"] = out.groupby("team_id", group_keys=False)[col].transform(lambda s: s.shift().rolling(window, min_periods=min_periods).mean())
    return out

def add_opponent_adjustments(frame: pd.DataFrame) -> pd.DataFrame: return frame.copy()
def add_venue_and_fatigue_features(frame: pd.DataFrame, min_periods: int = 3) -> pd.DataFrame:
    out = frame.sort_values(["team_id", "date"]).copy(); out["rest_days"] = out.groupby("team_id")["date"].diff().dt.total_seconds().div(86400).clip(lower=0); return out
def missingness_report(frame: pd.DataFrame) -> pd.DataFrame: return pd.DataFrame({"missing_count": frame.isna().sum(), "missing_pct": frame.isna().mean().mul(100)})
def validate_team_frame(frame: pd.DataFrame) -> None:
    needed = {"match_id", "team_id", "date", "venue", "opponent_id", "goals_for", "goals_against"}; missing = needed - set(frame.columns)
    if missing: raise DataIntegrityError(f"Columnas requeridas ausentes: {missing}")
