# ENTORNO Y CÓDIGO DEL TRABAJO DE GRADUACIÓN

> Repositorio público del prototipo del Trabajo de Graduación
> (https://github.com/signaidy/recomendador-fdm). Para ejecutar el prototipo basta con
> `docker compose up --build` en la raíz de este repositorio y abrir http://localhost:8501. Las rutas
> `proyecto/...` que aparecen en este documento corresponden a la carpeta en que se desarrolló; en este
> repositorio se omite el prefijo `proyecto/`.

Repositorio reproducible comprometido en el anteproyecto aprobado.

## Entorno

Se usa un entorno virtual con **Python 3.12**, no con el 3.14 del sistema. El motivo es la
disponibilidad de ruedas precompiladas para toda la pila científica: en un calendario de seis
semanas no conviene depender de compilaciones locales ni de soporte recién publicado.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r proyecto/requirements.txt
```

Ejecutar siempre con el intérprete del entorno:

```bash
.venv/Scripts/python.exe proyecto/src/<script>.py
```

`proyecto/requirements.txt` contiene las versiones exactas verificadas el 17 de agosto de 2026.
Versiones principales: numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, shap 0.52.0, pymoo 0.6.2,
streamlit 1.61.1, xgboost 3.4.1, matplotlib 3.11.1.

El entorno virtual está excluido de git; `requirements.txt` sí se versiona.

## Scripts

| Script | Función |
|---|---|
| `src/extraer_patil_l16.py` | Convierte el xlsx crudo de Patil et al. a `datos/procesados/patil_l16_pla.csv` |
| `src/auditoria_dispersion.py` | Dispersión de las respuestas y línea base sin destreza. Solo biblioteca estándar |
| `src/verificar_entorno.py` | Comprueba que scikit-learn, SHAP, pymoo, matplotlib y streamlit funcionan juntos |
| `src/extraer_aktepe_ergun.py` | Convierte el xlsx crudo de Aktepe y Ergün (500 corridas, 10 materiales) a `datos/procesados/aktepe_ergun_fdm_500.csv`, normalizando patrón de relleno y marcando la rugosidad centinela |
| `src/extraer_erkan_ra.py` | Renombra y valida el CSV crudo de Erkan, Sürer y Yaman (Ra en PLA, Box-Behnken) a `datos/procesados/erkan_ra_pla_box_behnken.csv` |
| `src/comparar_modelos.py` | Compara línea base, ridge, SVR, kNN, bosque aleatorio y gradient boosting sobre todas las campañas con validación cruzada anidada y agrupada; escribe `resultados/tablas/comparacion_modelos.csv` |
| `src/entrenar_sustituto.py` | Entrena la red ReLU sustituta sobre Aktepe y Ergün (destreza reportada por validación cruzada anidada: la arquitectura se elige con 3 pliegues internos dentro de cada uno de los 5 × 3 pliegues externos, `sustituto_mlp_seleccion_anidada.csv`; el modelo final se elige por validación cruzada sobre todos los datos; destreza por material) y exporta pesos a `resultados/modelos/` (unos 8 minutos) |
| `src/estimar_tiempo_orca.py` | Toma la probeta ISO 527-2 tipo 1A de `datos/crudos/probetas/` (Printables 653017, CC BY-NC 4.0; `--stl generar` produce en su lugar la ASTM D638 tipo IV propia), escribe un perfil de proceso de OrcaSlicer por combinación de parámetros, lamina (900 combinaciones: densidad de 10 a 100 % en pasos de 10 %, 3 alturas, 3 velocidades y 10 patrones) por línea de comandos para la Creality K1C con hasta seis procesos en paralelo (`--paralelo`), reanudable, y extrae tiempo y filamento de cada 3MF a `datos/procesados/orca_k1c_tiempo_material.csv`. `--limite N` prueba la cadena; `--solo-preparar` no invoca el laminador |
| `src/optimizar_milp.py` | Codifica la red como programa lineal entero mixto en cvxpy y lo resuelve con HiGHS. En modo discreto integra la tabla del laminador (tiempo y filamento exactos por combinación; `--tiempo-max`, `--material-max`, `--peso-tiempo`, `--peso-material`, `--potencia-w`; la densidad se explora de 10 a 100 % en pasos de 10 %: 30, 40, 60 y 70 % son interpolación del modelo y `--extrapolar` admite 10, 90 y 100 %, fuera del rango de entrenamiento y marcados como extrapolación), maximiza la puntuación ponderada S = w_r·R_n + w_q·Q_n − w_m·M_n − w_t·T_n con respuestas normalizadas al dominio (`--perfil` Equilibrado/Económico/Acabado/Resistencia máxima o `--w-resistencia`, `--w-calidad`, `--w-material`, `--w-tiempo`), reporta la solución de compromiso (rodilla) y la estabilidad ante ±10 % en los pesos, verifica contra enumeración exhaustiva, calcula el conjunto de Pareto exacto en cuatro objetivos, traza la frontera por método epsilon y evalúa la hipótesis contra `--referencia` (unos 5 minutos con barrido; `--sin-barrido` para una consulta) |
| `src/explicar.py` | Explicabilidad con SHAP (explicador por permutaciones): importancia global, contribución por nivel y explicaciones locales de la referencia, el óptimo y la candidata; tablas y figuras en `resultados/` (unos 20 s) |
| `src/app.py` | Prototipo de interfaz en Streamlit (perfiles de prioridad, puntuación S, solución de compromiso, estabilidad, Pareto, SHAP local): `.venv/Scripts/python.exe -m streamlit run proyecto/src/app.py`. Con Docker: `docker compose up --build` desde la raíz del repositorio (imagen autocontenida, cualquier sistema). Manual en `proyecto/MANUAL_PROTOTIPO.md` |

Los extractores y la auditoría de dispersión no requieren el entorno virtual: usan solo la
biblioteca estándar, de modo que la auditoría de datos es reproducible en cualquier instalación
de Python. `comparar_modelos.py` sí requiere el entorno virtual.

## Datos

```
datos/crudos/         sin modificar, con huella SHA-256 y fecha de descarga
datos/procesados/     derivados reproducibles a partir de los crudos
datos/literatura/     tablas transcritas de artículos, una campaña por archivo (véase su README)
```

La procedencia, licencia y veredicto de cada fuente está en
`../referencias/fuentes_y_datasets.md`. **Los datos crudos no se modifican nunca**: toda
depuración ocurre en código, para que sea auditable.

## Verificación del entorno

`src/verificar_entorno.py` ejecuta una cadena mínima de extremo a extremo —validación cruzada,
SHAP, NSGA-II, gráfica— sobre datos reales. No produce resultados del trabajo: existe para que una
incompatibilidad de versiones aparezca ahora y no a mitad del calendario.
