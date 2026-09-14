"""Compatibilidad síncrona para scripts heredados.

El código nuevo debe importar :class:`engine.thestats_client.TheStatsClient`.
Esta fachada conserva las utilidades de consola sin ocultar fallos: los errores
de red y cuota se propagan en lugar de convertirse silenciosamente en ``None``.
"""
from __future__ import annotations
import asyncio
from typing import Any
from engine.thestats_client import TheStatsClient

class FootballDataClient:
    def __init__(self) -> None: self._client = TheStatsClient()
    def _run(self, coroutine: Any) -> Any: return asyncio.run(coroutine)
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]: return self._run(self._client.request(endpoint, params=params))
    def get_match_details(self, match_id: str) -> dict[str, Any]: return self.get(f"/football/matches/{match_id}")
    def get_matches_by_date(self, date_str: str) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        async def collect() -> None:
            async for page in self._client.pages("/football/matches", params={"date_from": date_str, "date_to": date_str}): data.extend(page.get("data", []))
        self._run(collect()); return data
    def get_historical_team_data(self, team_id: str, match_date: str) -> list[dict[str, Any]]:
        payload = self.get("/football/matches", {"team_id": team_id, "date_to": match_date, "status": "finished", "per_page": 10})
        return payload.get("data", [])
    def search_team_name(self, query: str) -> dict[str, list[dict[str, Any]]]:
        return {"thestats": self.get("/football/teams", {"search": query}).get("data", []), "isports": []}
