"""
tests/conftest.py
=================
Fixtures compartidas. Los datos sintéticos aquí generados SON SOLO PARA
TESTS: reproducen la forma estadística de datos reales de TheStatsAPI
para que la suite corra rápido y sin red. El pipeline de producción
nunca usa estos generadores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.thestats_features import (
    add_derived_metrics, add_opponent_adjustments,
    add_venue_and_fatigue_features, add_rolling_features, match_to_team_rows)
from engine.ml_pipeline import build_model_matrix

# --------------------------------------------------------------------------- #
# Payloads con la forma EXACTA de las respuestas de TheStatsAPI (para mocks)
# --------------------------------------------------------------------------- #
MATCH_PAYLOAD = {
    "data": {
        "id": "mt_838955483",
        "competition_id": "comp_3039",
        "competition_name": "Premier League",
        "season_id": "sn_6125938",
        "matchday": 20,
        "status": "finished",
        "utc_date": "2024-01-15T15:00:00.000Z",
        "home_team": {"id": "tm_0406", "name": "Liverpool"},
        "away_team": {"id": "tm_1002", "name": "Aston Villa"},
        "score": {"home": 2, "away": 1,
                  "half_time_home": 1, "half_time_away": 0},
        "venue": {"name": "Anfield", "city": "Liverpool"},
        "xg_available": True,
    }
}

STATS_PAYLOAD = {
    "data": {
        "match_id": "mt_838955483",
        "overview": {
            "ball_possession": {"all": {"home": 58, "away": 42}},
            "expected_goals": {"all": {"home": 1.87, "away": 0.92}},
            "big_chances": {"all": {"home": 3, "away": 1}},
            "total_shots": {"all": {"home": 14, "away": 9}},
            "shots_on_target": {"all": {"home": 6, "away": 3}},
            "corner_kicks": {"all": {"home": 7, "away": 4}},
            "fouls": {"all": {"home": 10, "away": 13}},
            "yellow_cards": {"all": {"home": 1, "away": 2}},
            "passes": {"all": {"home": 612, "away": 445}},
            "accurate_passes": {"all": {"home": 540, "away": 370}},
        },
    }
}

PLAYER_STATS_PAYLOAD = {
    "data": [
        {"player_id": "pl_29627593", "player_name": "Mohamed Salah",
         "team_id": "tm_0406", "position": "F", "rating": 8.2,
         "minutes_played": 90, "started": True, "played": True,
         "passing": {"total_passes": 34, "accurate_passes": 28,
                     "key_passes": 4, "assists": 1},
         "shooting": {"goals": 1, "total_shots": 4, "shots_on_target": 2,
                      "expected_goals": 0.71, "expected_assists": 0.42}},
        {"player_id": "pl_10020001", "player_name": "Ollie Watkins",
         "team_id": "tm_1002", "position": "F", "rating": 6.9,
         "minutes_played": 90, "started": True, "played": True,
         "passing": {"total_passes": 21, "accurate_passes": 15,
                     "key_passes": 2, "assists": 0},
         "shooting": {"goals": 1, "total_shots": 3, "shots_on_target": 1,
                      "expected_goals": 0.55, "expected_assists": 0.10}},
    ]
}


# --------------------------------------------------------------------------- #
# Generador determinista de un frame equipo-partido con estadística realista
# --------------------------------------------------------------------------- #
def make_team_frame(seed: int = 7, n_teams: int = 10, n_rounds: int = 30) -> pd.DataFrame:
    """
    Liga sintética round-robin con fuerzas de ataque/defensa latentes y
    goles ~ Poisson. Produce el frame a través del pipeline REAL de
    features (con umbrales reducidos para tener señal con pocos datos).
    """
    rng = np.random.default_rng(seed)
    attack = np.exp(rng.normal(0.0, 0.30, n_teams))     # ~1.0 = promedio
    defense_weak = np.exp(rng.normal(0.0, 0.25, n_teams))  # >1 = defensa débil
    team_ids = [f"tm_{i:04d}" for i in range(n_teams)]

    rows: list[dict] = []
    dates = pd.date_range("2025-08-01", periods=n_rounds, freq="7D")
    for r in range(n_rounds):
        # emparejamiento circular: cada equipo juega una vez por jornada
        order = [(i + r) % n_teams for i in range(n_teams)]
        for k in range(n_teams // 2):
            hi, ai = order[k], order[n_teams - 1 - k]
            lam_h = 1.45 * attack[hi] * defense_weak[ai]
            lam_a = 1.10 * attack[ai] * defense_weak[hi]
            gh = int(rng.poisson(max(lam_h, 0.05)))
            ga = int(rng.poisson(max(lam_a, 0.05)))
            mid = f"mt_{r:03d}_{k:02d}"
            iso = dates[r].strftime("%Y-%m-%d") + "T15:00:00.000Z"

            def side_stats(lam: float, gf: int) -> dict:
                shots = int(max(0, rng.poisson(8 + 4 * lam)))
                on_t = int(rng.binomial(shots, 0.36))
                xg = float(max(0.05, lam + rng.normal(0, 0.22)))
                poss = float(np.clip(50 + (lam - 1.25) * 8 + rng.normal(0, 5), 25, 75))
                corners = int(max(0, rng.poisson(4 + 1.5 * lam)))
                fouls = int(max(0, rng.poisson(11 + rng.normal(0, 2))))
                passes = int(max(100, rng.poisson(380 + 3.0 * poss)))
                acc = int(rng.binomial(passes, 0.84))
                return {"shots": shots, "on": on_t, "xg": xg, "poss": poss,
                        "corners": corners, "fouls": fouls,
                        "passes": passes, "acc": acc}

            hs, as_ = side_stats(lam_h, gh), side_stats(lam_a, ga)
            match = {"id": mid, "status": "finished", "utc_date": iso,
                     "competition_id": "comp_999", "season_id": "sn_999",
                     "matchday": r + 1,
                     "home_team": {"id": team_ids[hi], "name": team_ids[hi]},
                     "away_team": {"id": team_ids[ai], "name": team_ids[ai]},
                     "score": {"home": gh, "away": ga}}
            def ov(home_v, away_v):
                return {"all": {"home": home_v, "away": away_v}}
            stats = {"overview": {
                "ball_possession": ov(hs["poss"], as_["poss"]),
                "expected_goals": ov(hs["xg"], as_["xg"]),
                "big_chances": ov(int(hs["xg"] > 1.2), int(as_["xg"] > 1.2)),
                "total_shots": ov(hs["shots"], as_["shots"]),
                "shots_on_target": ov(hs["on"], as_["on"]),
                "corner_kicks": ov(hs["corners"], as_["corners"]),
                "fouls": ov(hs["fouls"], as_["fouls"]),
                "yellow_cards": ov(int(rng.random() < 0.4), int(rng.random() < 0.5)),
                "passes": ov(hs["passes"], as_["passes"]),
                "accurate_passes": ov(hs["acc"], as_["acc"])}}
            players = [
                {"player_id": f"pl_h_{mid}", "team_id": team_ids[hi],
                 "played": True, "passing": {"key_passes": int(rng.poisson(3 + 2 * lam_h))},
                 "shooting": {"expected_goals": round(hs["xg"] * 0.4, 3)}},
                {"player_id": f"pl_a_{mid}", "team_id": team_ids[ai],
                 "played": True, "passing": {"key_passes": int(rng.poisson(3 + 2 * lam_a))},
                 "shooting": {"expected_goals": round(as_["xg"] * 0.4, 3)}},
            ]
            rows.extend(match_to_team_rows(match, stats, players))

    df = pd.DataFrame(rows)
    df = add_derived_metrics(df)
    df = add_rolling_features(df, min_periods=2, expanding_min_periods=8)
    df = add_opponent_adjustments(df)
    df = add_venue_and_fatigue_features(df, min_periods=2)
    return df


# --------------------------------------------------------------------------- #
# Fixtures pytest
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def team_frame() -> pd.DataFrame:
    return make_team_frame()


@pytest.fixture(scope="session")
def model_matrix(team_frame):
    return build_model_matrix(team_frame)


@pytest.fixture(scope="session")
def api_payloads():
    return {"match": MATCH_PAYLOAD, "stats": STATS_PAYLOAD,
            "player_stats": PLAYER_STATS_PAYLOAD}
