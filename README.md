# Filtro de Kalman para Control de Tráfico Aéreo

Simulación de seguimiento de trayectorias aéreas mediante filtros de Kalman lineales con modelos CV y CA. Las medidas provienen de un radar en coordenadas polares (rho, theta).

## Características

- **Modelos dinámicos**: CV (4 estados) y CA (6 estados)
- **Debiasing de medidas**: Fórmulas exactas de Lerro & Bar-Shalom (Ec. 12, 13a-13c)
- **Transformaciones geodésicas**: WGS84 con conversión a sistema local ENU
- **Visualización**: Gráficas estilo seaborn con exportación PNG y SVG
- **Waypoints reales**: RATAS, NUBLO, ROVAK con radar VALDES

## Estructura

```
Filtro-kalman/
├── src/
│   ├── config.py          # Parámetros de simulación
│   ├── transformaciones.py # DMS, geocéntricas, locales
│   ├── generar_datos.py   # Trayectorias y medidas radar
│   ├── medidas.py         # Debiasing Lerro & Bar-Shalom
│   ├── kalman.py          # Filtro de Kalman lineal
│   ├── objetivos.py       # Orquestación CV/CA
│   └── graficas.py        # Visualización y exportación
├── main.py
└── pyproject.toml
```
> [!NOTE]
> - CV ➜ velocidad constante
> - CA ➜ aceleración constante

---
## Ejecución

Para gestionar dependencias:
```bash
uv sync
```
Para ejecutar:
```bash
uv run main.py
```

Las gráficas se exportan automáticamente a `resultados/png/` y `resultados/svg/`.

## Parámetros

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| V1_MS | 210.92 m/s | Velocidad inicial (410 kt) |
| V2_MS | 262.36 m/s | Velocidad máxima (510 kt) |
| ACCEL | 10.0 m/s² | Aceleración |
| DT | 4.0 s | Paso de tiempo |
| SIGMA_RHO | 30.0 m | Error de distancia radar |
| SIGMA_THETA | 0.068° | Error angular radar |

## Requisitos

- Python >= 3.13
- numpy, seaborn, matplotlib

## Autores

- Alejandro Gil Getino - agilge00@estudiantes.unileon.es - Universidad de León
- Raúl Castedo Flórez - rcastf02@estudiantes.unileon.es - Universidad de León
