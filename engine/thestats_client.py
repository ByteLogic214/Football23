"""Cliente asíncrono, tipado y seguro para TheStatsAPI v1/v2.

El registro ``ENDPOINTS`` es la superficie completa (36 GET) publicada por
https://api.thestatsapi.com/llms.txt a 2026-09-14. ``request`` permite usar
cualquiera sin esperar una nueva versión del SDK; los métodos de conveniencia
mantienen el código de producto legible.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

import httpx

BASE_URL = "https://api.thestatsapi.com/api"
ENDPOINTS = (
    "/health", "/football/competitions", "/football/competitions/{competition_id}",
    "/football/competitions/{competition_id}/seasons", "/football/competitions/{competition_id}/seasons/{season_id}/groups",
    "/football/competitions/{competition_id}/seasons/{season_id}/standings", "/football/teams", "/football/teams/{team_id}/players",
    "/football/teams/{team_id}/injuries-suspensions", "/football/teams/{team_id}", "/football/teams/{team_id}/stats",
    "/football/teams/{team_id}/standings", "/football/matches", "/football/matches/{match_id}",
    "/football/matches/{match_id}/referee", "/football/matches/{match_id}/stats", "/football/matches/{match_id}/live-stats",
    "/football/matches/{match_id}/player-stats", "/football/matches/{match_id}/live-player-stats", "/football/matches/{match_id}/lineups",
    "/football/matches/{match_id}/players/{player_id}/heatmap", "/football/matches/{match_id}/shotmap", "/football/matches/{match_id}/timeline",
    "/football/matches/{match_id}/live-timeline", "/football/players", "/football/players/{player_id}",
    "/football/players/{player_id}/injuries-suspensions", "/football/players/{player_id}/stats",
    "/football/players/{player_id}/competitions/{competition_id}/seasons/{season_id}/heatmap", "/football/matches/{match_id}/odds",
    "/football/matches/{match_id}/odds/live", "/football/matches/{match_id}/odds/players", "/v2/football/matches/{match_id}/odds/players",
    "/coverage/leagues", "/coverage/leagues/{competition_id}", "/coverage/summary",
)

class StatsAPIError(RuntimeError):
    """Error de API con estado, código de proveedor y petición correlacionable."""
    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message); self.status_code, self.code = status_code, code

class AuthenticationError(StatsAPIError): pass
class RateLimitError(StatsAPIError): pass

@dataclass(frozen=True)
class QuotaState:
    rate_limit: int | None; rate_remaining: int | None; rate_reset: int | None
    monthly_limit: int | None; monthly_remaining: int | None; monthly_reset: int | None

def _integer(headers: Mapping[str, str], key: str) -> int | None:
    try: return int(headers[key])
    except (KeyError, ValueError): return None

class TheStatsClient:
    def __init__(self, api_key: str | None = None, *, timeout: float = 20.0, max_retries: int = 3, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.api_key = api_key or os.getenv("THESTATSAPI_KEY")
        if not self.api_key: raise AuthenticationError("THESTATSAPI_KEY no está configurada")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}, transport=transport)
        self.quota: QuotaState | None = None

    async def __aenter__(self) -> "TheStatsClient": return self
    async def __aexit__(self, *args: object) -> None: await self.aclose()
    async def aclose(self) -> None: await self._client.aclose()

    def _record_quota(self, response: httpx.Response) -> None:
        h = response.headers
        self.quota = QuotaState(*(_integer(h, key) for key in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "X-Monthly-Quota-Limit", "X-Monthly-Quota-Remaining", "X-Monthly-Quota-Reset")))

    async def request(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not path.startswith("/") or ".." in path: raise ValueError("Ruta de API inválida")
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.get(path, params=params)
                self._record_quota(response)
                if response.status_code == 429:
                    if attempt == self.max_retries: raise RateLimitError("Límite de TheStatsAPI agotado", status_code=429)
                    retry_after = float(response.headers.get("Retry-After", min(60, 2 ** attempt)))
                    await asyncio.sleep(retry_after + random.uniform(0, .25)); continue
                if response.status_code >= 500 and attempt < self.max_retries:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, .25)); continue
                if response.status_code >= 400:
                    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    raise StatsAPIError(body.get("message", response.text), status_code=response.status_code, code=body.get("code"))
                return response.json()
            except httpx.TransportError as exc:
                if attempt == self.max_retries: raise StatsAPIError("Error de red hacia TheStatsAPI") from exc
                await asyncio.sleep((2 ** attempt) + random.uniform(0, .25))
        raise AssertionError("bucle de reintentos inalcanzable")

    async def pages(self, path: str, *, params: Mapping[str, Any] | None = None) -> AsyncIterator[dict[str, Any]]:
        query = dict(params or {}); query.setdefault("per_page", 100); query.setdefault("page", 1)
        while True:
            payload = await self.request(path, params=query); yield payload
            meta = payload.get("meta", {})
            if int(query["page"]) >= int(meta.get("total_pages", query["page"])): break
            query["page"] = int(query["page"]) + 1

    async def get_match(self, match_id: str) -> dict[str, Any]: return (await self.request(f"/football/matches/{match_id}"))["data"]
    async def get_match_stats(self, match_id: str) -> dict[str, Any]: return (await self.request(f"/football/matches/{match_id}/stats"))["data"]
    async def get_match_player_stats(self, match_id: str) -> list[dict[str, Any]]: return (await self.request(f"/football/matches/{match_id}/player-stats"))["data"]
    async def get_match_bundle(self, match_id: str) -> dict[str, Any]:
        match, stats, players = await asyncio.gather(self.get_match(match_id), self.get_match_stats(match_id), self.get_match_player_stats(match_id))
        return {"match": match, "stats": stats, "player_stats": players}
