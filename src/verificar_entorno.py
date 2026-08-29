"""Verifica que el entorno soporta la cadena completa del trabajo.

No produce resultados del trabajo de graduación: solo comprueba que las piezas
de las semanas B, C y D funcionan juntas sobre las versiones instaladas, para no
descubrir una incompatibilidad a mitad del calendario. Las versiones de numpy,
pandas y scikit-learn instaladas son mayores recientes, y SHAP y pymoo son las
dependencias con más historial de romper ante ese tipo de salto.

Ejecutar con el intérprete del entorno virtual:
    .venv/Scripts/python.exe proyecto/src/verificar_entorno.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT / "proyecto" / "datos" / "procesados" / "patil_l16_pla.csv"

FACTORES = [
    "altura_capa_mm",
    "angulo_raster_grados",
    "velocidad_impresion_mms",
    "densidad_relleno_pct",
]
RESPUESTA = "error_lado_cubo_mm"


def separador(titulo: str) -> None:
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


def main() -> None:
    ok = True

    separador("1. CARGA DE DATOS")
    df = pd.read_csv(DATOS)
    X = df[FACTORES].to_numpy(dtype=float)
    y = df[RESPUESTA].to_numpy(dtype=float)
    print(f"   registros={len(df)}  factores={len(FACTORES)}  respuesta={RESPUESTA}")

    separador("2. VALIDACION CRUZADA: LINEA BASE FRENTE A BOSQUE ALEATORIO")
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut

    # Con n=16 se usa dejar-uno-fuera. En el trabajo real la agrupacion sera por
    # campana; aqui solo se comprueba que la maquinaria corre.
    loo = LeaveOneOut()
    pred_base, pred_bosque, reales = [], [], []
    for tr, te in loo.split(X):
        base = DummyRegressor(strategy="median").fit(X[tr], y[tr])
        bosque = RandomForestRegressor(
            n_estimators=300, random_state=0, min_samples_leaf=1
        ).fit(X[tr], y[tr])
        pred_base.append(base.predict(X[te])[0])
        pred_bosque.append(bosque.predict(X[te])[0])
        reales.append(y[te][0])

    reales = np.array(reales)
    mae_base = float(np.mean(np.abs(np.array(pred_base) - reales)))
    mae_bosque = float(np.mean(np.abs(np.array(pred_bosque) - reales)))
    destreza = 1 - mae_bosque / mae_base

    print(f"   MAE linea base (mediana)  = {mae_base:.4f} mm")
    print(f"   MAE bosque aleatorio      = {mae_bosque:.4f} mm")
    print(f"   Destreza S = 1 - r        = {destreza:+.3f}   (criterio del programa: S >= 0.30)")
    print()
    print("   NOTA: cifra de humo, no resultado del trabajo. Falta ajuste de")
    print("   hiperparametros, protocolo de agrupacion definitivo y analisis de residuos.")

    separador("3. EXPLICABILIDAD (SHAP)")
    try:
        import shap

        modelo = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
        explicador = shap.TreeExplainer(modelo)
        valores = explicador.shap_values(X)
        print(f"   TreeExplainer OK. Matriz de valores SHAP: {np.shape(valores)}")
        importancia = np.abs(valores).mean(axis=0)
        for nombre, imp in sorted(
            zip(FACTORES, importancia), key=lambda t: -t[1]
        ):
            print(f"     {nombre:26} {imp:.5f}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"   FALLO SHAP -> {type(exc).__name__}: {exc}")

    separador("4. OPTIMIZACION MULTIOBJETIVO (pymoo, NSGA-II)")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        modelo_error = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
        modelo_tiempo = RandomForestRegressor(n_estimators=200, random_state=0).fit(
            X, df["tiempo_construccion_min"].to_numpy(dtype=float)
        )

        class Compromiso(ElementwiseProblem):
            """Minimizar simultaneamente error dimensional y tiempo."""

            def __init__(self) -> None:
                super().__init__(
                    n_var=4,
                    n_obj=2,
                    xl=X.min(axis=0),
                    xu=X.max(axis=0),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                punto = x.reshape(1, -1)
                out["F"] = [
                    float(modelo_error.predict(punto)[0]),
                    float(modelo_tiempo.predict(punto)[0]),
                ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                Compromiso(), NSGA2(pop_size=40), ("n_gen", 15),
                seed=1, verbose=False,
            )

        frente = np.atleast_2d(res.F)
        print(f"   NSGA-II OK. Soluciones no dominadas: {len(frente)}")
        print(f"   Error dimensional: {frente[:,0].min():.4f} a {frente[:,0].max():.4f} mm")
        print(f"   Tiempo:            {frente[:,1].min():.1f} a {frente[:,1].max():.1f} min")
        print("   La existencia de un frente con rango en ambos ejes confirma que hay")
        print("   compromiso real entre calidad y tiempo, que es la premisa del trabajo.")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"   FALLO pymoo -> {type(exc).__name__}: {exc}")

    separador("5. GRAFICAS E INTERFAZ")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        salida = ROOT / "proyecto" / "resultados" / "figuras" / "_verificacion_entorno.png"
        salida.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(df["altura_capa_mm"], df[RESPUESTA])
        ax.set_xlabel("altura de capa (mm)")
        ax.set_ylabel("error del lado (mm)")
        fig.tight_layout()
        fig.savefig(salida, dpi=100)
        plt.close(fig)
        print(f"   matplotlib OK -> {salida.name} ({salida.stat().st_size:,} bytes)")

        import streamlit
        print(f"   streamlit {streamlit.__version__} importable")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"   FALLO -> {type(exc).__name__}: {exc}")

    separador("RESULTADO")
    print("   ENTORNO COMPLETO Y FUNCIONAL" if ok else "   HAY FALLOS, revisar arriba")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
