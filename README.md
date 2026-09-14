# Football Analytics

Base profesional y auditable para analítica de fútbol con **TheStatsAPI**. No
promete ganancias ni convierte una heurística en una recomendación de apuesta:
las predicciones sólo se publican después de validación temporal, calibración y
backtesting reproducible.

## Qué incluye

- Cliente asíncrono autenticado, timeout, reintentos con *jitter*, paginación y
  seguimiento de límites por minuto y mensuales.
- Registro verificable de los **36 endpoints GET** de la especificación oficial
  (salud, competiciones, equipos, partidos, jugadores, lesiones, alineaciones,
  eventos, mapas, cuotas v1/v2 y cobertura). `scripts/verify_api_surface.py`
  falla si la fuente oficial incorpora o retira alguno.
- Feature engineering con faltantes como faltantes (no se inventa xG), ventanas
  desplazadas para evitar fuga de información y matriz de modelado temporal.
- CI con Ruff y pytest, configuración centralizada en `pyproject.toml` y una
  plantilla de entorno sin secretos.

## Inicio rápido

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
cp .env.example .env
export THESTATSAPI_KEY='tu-clave'     # no la añadas a Git
pytest
python scripts/verify_api_surface.py
```

Ejemplo de cliente:

```python
import asyncio
from engine.thestats_client import TheStatsClient

async def main():
    async with TheStatsClient() as api:
        match = await api.get_match("mt_838955483")
        print(match["home_team"]["name"])

asyncio.run(main())
```

## Calidad y operación

`THESTATSAPI_KEY` se lee sólo desde el entorno. Las respuestas 429 se reintentan
respetando `Retry-After`; los restantes errores se propagan como `StatsAPIError`
con estado y código. Cada respuesta conserva `client.quota` para que el producto
pueda frenar ingestas antes de agotar presupuesto.

Los datos de cobertura no son universales por liga, temporada o mercado. Antes
de entrenar, persiste el origen, hora de ingesta, cobertura y versión del modelo;
separa cronológicamente entrenamiento/validación/prueba y mide log-loss, Brier,
calibración y rendimiento fuera de muestra. No uses el sistema como consejo
financiero ni publicites rentabilidad sin resultados auditables.

## Desarrollo

```bash
ruff check .
pytest
python scripts/verify_api_surface.py
```

La especificación contractual es https://api.thestatsapi.com/llms.txt. El script
de verificación mantiene el SDK alineado sin copiar una documentación de 11.968
líneas que podría quedar obsoleta.
