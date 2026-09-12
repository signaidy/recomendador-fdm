# ENTORNO Y CÓDIGO DEL TRABAJO DE GRADUACIÓN

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

Los extractores y la auditoría de dispersión no requieren el entorno virtual: usan solo la
biblioteca estándar, de modo que la auditoría de datos es reproducible en cualquier instalación
de Python. `comparar_modelos.py` sí requiere el entorno virtual.

## Datos

```
datos/crudos/         sin modificar, con huella SHA-256 y fecha de descarga
datos/procesados/     derivados reproducibles a partir de los crudos
```

La procedencia, licencia y veredicto de cada fuente está en
`../referencias/fuentes_y_datasets.md`. **Los datos crudos no se modifican nunca**: toda
depuración ocurre en código, para que sea auditable.

## Verificación del entorno

`src/verificar_entorno.py` ejecuta una cadena mínima de extremo a extremo —validación cruzada,
SHAP, NSGA-II, gráfica— sobre datos reales. No produce resultados del trabajo: existe para que una
incompatibilidad de versiones aparezca ahora y no a mitad del calendario.
