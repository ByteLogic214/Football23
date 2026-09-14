"""Comprueba que el SDK no haya omitido endpoints de la especificación oficial."""
from __future__ import annotations
import re
from urllib.request import Request, urlopen
from engine.thestats_client import ENDPOINTS

SPEC_URL = "https://api.thestatsapi.com/llms.txt"
def main() -> None:
    request = Request(SPEC_URL, headers={"User-Agent": "football-analytics-api-surface-check/0.1"})
    spec = urlopen(request, timeout=20).read().decode("utf-8")
    documented = set(re.findall(r"^#### GET `([^`]+)`", spec, re.MULTILINE))
    registered = set(ENDPOINTS)
    if documented != registered:
        raise SystemExit(f"Diferencia de endpoints. Faltan={sorted(documented-registered)}; obsoletos={sorted(registered-documented)}")
    print(f"OK: {len(registered)} endpoints documentados y registrados")
if __name__ == "__main__": main()
