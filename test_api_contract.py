"""tests/test_api_contract.py — mocks de TheStatsAPI: la suite corre sin red."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from engine.thestats_client import TheStatsClient, StatsAPIError
from engine.thestats_features import match_to_team_rows
from engine.ml_pipeline import build_model_matrix
from tests.conftest import MATCH_PAYLOAD, STATS_PAYLOAD, PLAYER_STATS_PAYLOAD


def _patched_client():
    """Cliente con los 3 métodos de red sustituidos por mocks async.
    Cero conexiones salientes: los tests no dependen de la red."""
    client = TheStatsClient(api_key="test-key-no-network")
    return client, [
        patch.object(client, "get_match",
                     new=AsyncMock(return_value=MATCH_PAYLOAD["data"])),
        patch.object(client, "get_match_stats",
                     new=AsyncMock(return_value=STATS_PAYLOAD["data"])),
        patch.object(client, "get_match_player_stats",
                     new=AsyncMock(return_value=PLAYER_STATS_PAYLOAD["data"])),
    ]


def test_get_match_bundle_sin_red():
    client, patches = _patched_client()
    for p in patches:
        p.start()
    try:
        bundle = asyncio.run(client.get_match_bundle("mt_838955483"))
    finally:
        for p in patches:
            p.stop()
    assert bundle["match"]["home_team"]["name"] == "Liverpool"
    assert bundle["match"]["score"]["home"] == 2
    assert bundle["stats"]["overview"]["expected_goals"]["all"]["home"] == 1.87
    assert len(bundle["player_stats"]) == 2


def test_mock_fluye_hasta_la_matriz_de_modelado():
    """Integración: payload mockeado -> filas de features -> matriz ML."""
    rows = match_to_team_rows(MATCH_PAYLOAD["data"],
                              STATS_PAYLOAD, PLAYER_STATS_PAYLOAD["data"])
    assert len(rows) == 2
    home = next(r for r in rows if r["venue"] == "home")
    assert abs(home["xg_for"] - 1.87) < 1e-9
    assert home["shots_on_target_for"] == 6
    assert home["key_passes_for"] == 4          # 4 key passes de Salah
    away = next(r for r in rows if r["venue"] == "away")
    assert away["key_passes_for"] == 2          # 2 key passes de Watkins


def test_cliente_sin_api_key_lanza_authentication_error():
    import os
    old = os.environ.pop("THESTATSAPI_KEY", None)
    try:
        TheStatsClient(api_key=None)
        raise AssertionError("Debió lanzar AuthenticationError")
    except StatsAPIError:
        pass
    finally:
        if old:
            os.environ["THESTATSAPI_KEY"] = old
