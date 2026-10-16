# Prototipo de interfaz del Trabajo de Graduación: recomendador FDM PLA+.
# Imagen autocontenida: incluye el sustituto entrenado, la tabla del laminador y el Pareto exacto,
# de modo que quien evalúe el prototipo no necesita Python, OrcaSlicer ni Windows.
#
#   docker compose up --build        → http://localhost:8501
#
# La imagen reproduce dentro de /app la disposición del repositorio del trabajo (este repositorio es
# su submódulo «proyecto»), por eso los archivos se copian a /app/proyecto/...
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencias primero, para aprovechar la caché de capas.
COPY requirements-app.txt /app/proyecto/requirements-app.txt
RUN pip install -r /app/proyecto/requirements-app.txt

# Código, modelo entrenado, tablas y datos procesados que la interfaz lee.
COPY src/entrenar_sustituto.py src/optimizar_milp.py src/explicar.py src/app.py /app/proyecto/src/
COPY resultados/modelos/ /app/proyecto/resultados/modelos/
COPY resultados/tablas/sustituto_mlp_validacion.csv \
     resultados/tablas/milp_pareto_exacto_discreto_closed.csv \
     resultados/tablas/milp_pareto_exacto_discreto_closed_extrapolar.csv /app/proyecto/resultados/tablas/
COPY resultados/figuras/shap_global.png /app/proyecto/resultados/figuras/
COPY datos/procesados/aktepe_ergun_fdm_500.csv \
     datos/procesados/orca_k1c_tiempo_material.csv /app/proyecto/datos/procesados/

RUN useradd --create-home usuario && chown -R usuario /app
USER usuario

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "proyecto/src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
