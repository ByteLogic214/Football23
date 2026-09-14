"""tests/test_features.py — pruebas unitarias del feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.thestats_features import (
    DataIntegrityError, add_derived_metrics, match_to_team_rows,
    missingness_report, validate_team_frame)
from tests.conftest import MATCH_PAYLOAD, STATS_PAYLOAD, PLAYER_STATS_PAYLOAD


def _base_match(mid="mt_x", date="2025-08-01T15:00:00.000Z"):
    return {"id": mid, "status": "finished", "utc_date": date,
            "home_team": {"id": "tm_A", "name": "A"},
            "away_team": {"id": "tm_B", "name": "B"},
            "score": {"home": 2, "away": 1}}


# --------------------------------------------------------------------------- #
# Forma y esquema
# --------------------------------------------------------------------------- #
def test_match_genera_exactamente_dos_filas():
    rows = match_to_team_rows(_base_match(), STATS_PAYLOAD,
                       PLAYER_STATS_PAYLOAD["data"])
    assert len(rows) == 2
    venues = {r["venue"] for r in rows}
    assert venues == {"home", "away"}
    home = next(r for r in rows if r["venue"] == "home")
    assert home["goals_for"] == 2 and home["goals_against"] == 1
    assert home["points"] == 3.0
    away = next(r for r in rows if r["venue"] == "away")
    assert away["goals_for"] == 1 and away["points"] == 0.0


def test_payloads_mock_respetan_contrato_de_api():
    """Los mocks deben parecerse a la API real: claves home/away en
    overview 'all' y envelope 'data'."""
    for payload in (MATCH_PAYLOAD, STATS_PAYLOAD, PLAYER_STATS_PAYLOAD):
        assert "data" in payload
    ov = STATS_PAYLOAD["data"]["overview"]
    assert ov["expected_goals"]["all"]["home"] >= 0
    assert set(ov) >= {"expected_goals", "total_shots", "corner_kicks",
                       "fouls", "ball_possession", "accurate_passes"}


def test_esquema_roto_aborta():
    roto = _base_match()
    del roto["home_team"]
    try:
        match_to_team_rows(roto)
        raise AssertionError("Debió lanzar DataIntegrityError")
    except DataIntegrityError:
        pass


# --------------------------------------------------------------------------- #
# Política de nulos (integridad)
# --------------------------------------------------------------------------- #
def test_gap_de_stats_deja_xg_nan_pero_conserva_goles():
    rows = match_to_team_rows(_base_match(), stats=None, player_stats=None)
    home = next(r for r in rows if r["venue"] == "home")
    assert pd.isna(home["xg_for"])           # cobertura ausente -> NaN
    assert home["goals_for"] == 2            # dato real del detalle -> intacto
    assert pd.isna(home["key_passes_for"])   # sin player-stats -> NaN


def test_tasa_de_conversion_nan_con_denominador_cero():
    df = pd.DataFrame([{
        "match_id": "m1", "team_id": "t1", "date": pd.Timestamp("2025-01-01", tz="UTC"),
        "venue": "home", "opponent_id": "t2",
        "goals_for": 0.0, "goals_against": 0.0, "points": 1.0,
        "total_shots_for": 0.0, "shots_on_target_for": 0.0,
        "accurate_passes_for": np.nan, "passes_for": np.nan, "xg_for": np.nan}])
    out = add_derived_metrics(df)
    assert pd.isna(out.loc[0, "conversion_rate"])   # 0/0 -> NaN, nunca 0 ni inf
    assert pd.isna(out.loc[0, "shot_accuracy"])


def test_media_movil_sin_leakage():
    """La media móvil del partido t NO debe incluir el partido t."""
    from engine.thestats_features import add_rolling_features
    fechas = pd.date_range("2025-01-01", periods=5, freq="7D", tz="UTC")
    base = {"team_id": "t1", "venue": "home", "opponent_id": "t2",
            "goals_against": 1.0, "points": 3.0}
    rows = []
    goles = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, g in enumerate(goles):
        r = dict(base); r.update({"match_id": f"m{i}",
                                  "date": fechas[i],
                                  "goals_for": g})
        rows.append(r)
    df = add_rolling_features(pd.DataFrame(rows), window=3, min_periods=1,
                              expanding_min_periods=1)
    df = df.sort_values("date").reset_index(drop=True)
    # fila 2: previos [1,2] -> media 1.5 (sin incluir el 3 del propio partido)
    assert df.loc[2, "roll_goals_for"] == 1.5
    # fila 4: previos [2,3,4] -> media 3.0 (sin incluir el 5)
    assert df.loc[4, "roll_goals_for"] == 3.0


def test_missingness_report_detecta_cobertura():
    df = pd.DataFrame({"match_id": ["a"], "team_id": ["t"],
                       "date": [pd.Timestamp("2025-01-01", tz="UTC")],
                       "venue": ["home"], "opponent_id": ["x"],
                       "goals_for": [1.0], "goals_against": [0.0],
                       "xg_for": [np.nan]})
    rep = missingness_report(df)
    assert "xg_for" in rep.index and rep.loc["xg_for", "missing_pct"] == 100.0
    validate_team_frame(df)  # no lanza: gap de cobertura no es corrupción
