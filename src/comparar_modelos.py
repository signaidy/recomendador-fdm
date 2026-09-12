"""Compara familias de modelos sobre cada campaña disponible, bajo validación cruzada agrupada.

Responde a la pregunta de si una máquina de vectores de soporte para regresión (SVR) es la
mejor opción frente a las alternativas previstas en el programa definitivo (línea base
regularizada, bosque aleatorio y un tercer modelo no lineal). Todos los modelos se evalúan con
el mismo protocolo y se comparan contra el predictor constante, de modo que la métrica
principal es la destreza S = 1 − MAE_modelo / MAE_base (numeral 4.4 del programa).

Protocolo:
- Campañas pequeñas (n ≤ 40): dejar-uno-fuera.
- Campañas medianas sin réplicas identificables: K-fold repetido (5 × 3).
- Campaña Erkan: K-fold agrupado por probeta, porque las 18 inclinaciones de una misma
  probeta comparten parámetros de proceso.
- Los hiperparámetros de SVR, kNN y ridge se ajustan en validación cruzada interna dentro de
  cada pliegue externo (validación anidada), para que la comparación no favorezca al modelo
  con más perillas.

Son cifras de comparación de familias, no el resultado final del trabajo: sirven para elegir
el modelo, no para reportar su desempeño definitivo.

Ejecutar con el intérprete del entorno virtual:
    .venv/Scripts/python.exe proyecto/src/comparar_modelos.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SEMILLA = 20260903
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "proyecto" / "datos" / "procesados"
CRUDOS = ROOT / "proyecto" / "datos" / "crudos"
SALIDA = ROOT / "proyecto" / "resultados" / "tablas" / "comparacion_modelos.csv"


# --------------------------------------------------------------------------- campañas
def campanas() -> list[dict]:
    lista: list[dict] = []

    patil = pd.read_csv(PROC / "patil_l16_pla.csv")
    x_patil = ["altura_capa_mm", "angulo_raster_grados", "velocidad_impresion_mms", "densidad_relleno_pct"]
    for y, unidad in [
        ("rugosidad_ra_um_cara_superior", "µm"),
        ("tiempo_construccion_min", "min"),
        ("error_lado_cubo_mm", "mm"),
    ]:
        lista.append(dict(campana="Patil L16 PLA", respuesta=y, unidad=unidad, df=patil,
                          num=x_patil, cat=[], grupos=None))

    okudan = pd.read_csv(CRUDOS / "okudan_3dprinter_manufacturing.csv")
    okudan = okudan[(okudan.material == "pla") & (okudan.roughness >= 0)
                    & (okudan.bed_temperature <= 120) & (okudan.print_speed <= 200)].copy()
    lista.append(dict(campana="Okudan PLA depurado", respuesta="tension_strength", unidad="MPa", df=okudan,
                      num=["layer_height", "wall_thickness", "infill_density", "nozzle_temperature",
                           "bed_temperature", "print_speed", "fan_speed"],
                      cat=["infill_pattern"], grupos=None))

    aktepe = pd.read_csv(PROC / "aktepe_ergun_fdm_500.csv")
    x_akt_num = ["altura_capa_mm", "densidad_relleno_pct", "velocidad_impresion_mms"]
    x_akt_cat = ["patron_relleno", "tipo_impresora"]
    pla_plus = aktepe[aktepe.material == "PLA+"]
    for y, unidad in [("tension_maxima_mpa", "MPa"), ("rugosidad_ra_um", "µm"), ("dureza_shore", "Shore")]:
        lista.append(dict(campana="Aktepe PLA+", respuesta=y, unidad=unidad, df=pla_plus.dropna(subset=[y]),
                          num=x_akt_num, cat=x_akt_cat, grupos=None))
    pla = aktepe[aktepe.material == "PLA"]
    lista.append(dict(campana="Aktepe PLA", respuesta="tension_maxima_mpa", unidad="MPa", df=pla,
                      num=x_akt_num, cat=["patron_relleno"], grupos=None))

    erkan = pd.read_csv(PROC / "erkan_ra_pla_box_behnken.csv")
    x_erkan = ["altura_capa_mm", "temperatura_extrusion_c", "velocidad_pared_exterior_mms",
               "densidad_relleno_pct", "espesor_pared_mm", "temperatura_cama_c", "ventilador_pct"]
    lista.append(dict(campana="Erkan PLA, todas las caras", respuesta="rugosidad_ra_um", unidad="µm", df=erkan,
                      num=x_erkan + ["inclinacion_superficie_grados"], cat=[], grupos="probeta"))
    # Sin la inclinación como entrada: mide cuánta señal aportan los parámetros de proceso por sí solos.
    lista.append(dict(campana="Erkan PLA, todas las caras, sin inclinación", respuesta="rugosidad_ra_um",
                      unidad="µm", df=erkan, num=x_erkan, cat=[], grupos="probeta"))
    lista.append(dict(campana="Erkan PLA, cara superior 0°", respuesta="rugosidad_ra_um", unidad="µm",
                      df=erkan[erkan.inclinacion_superficie_grados == 0], num=x_erkan, cat=[], grupos=None))
    lista.append(dict(campana="Erkan PLA, pared vertical 90°", respuesta="rugosidad_ra_um", unidad="µm",
                      df=erkan[erkan.inclinacion_superficie_grados == 90], num=x_erkan, cat=[], grupos=None))
    return lista


# --------------------------------------------------------------------------- modelos
def preprocesador(num: list[str], cat: list[str]) -> ColumnTransformer:
    partes = [("num", StandardScaler(), num)]
    if cat:
        partes.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(partes)


def modelos(num: list[str], cat: list[str], n: int) -> dict[str, object]:
    interno = KFold(n_splits=3, shuffle=True, random_state=SEMILLA) if n < 60 else 5

    def tubo(estimador):
        return Pipeline([("prep", preprocesador(num, cat)), ("modelo", estimador)])

    svr = GridSearchCV(
        TransformedTargetRegressor(regressor=tubo(SVR(kernel="rbf")), transformer=StandardScaler()),
        param_grid={
            "regressor__modelo__C": [0.3, 1, 3, 10, 30, 100],
            "regressor__modelo__epsilon": [0.02, 0.1, 0.3],
            "regressor__modelo__gamma": ["scale", 0.1, 0.5],
        },
        scoring="neg_mean_absolute_error", cv=interno, n_jobs=-1,
    )
    knn = GridSearchCV(
        tubo(KNeighborsRegressor(weights="distance")),
        param_grid={"modelo__n_neighbors": [k for k in (2, 3, 5, 8) if k < max(3, n // 3)]},
        scoring="neg_mean_absolute_error", cv=interno,
    )
    return {
        "Línea base (mediana)": DummyRegressor(strategy="median"),
        "Ridge (efectos principales)": tubo(RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "SVR RBF (anidado)": svr,
        "kNN (anidado)": knn,
        "Bosque aleatorio": tubo(RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEMILLA)),
        "Gradient boosting": tubo(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2,
                                                            subsample=0.8, random_state=SEMILLA)),
    }


def particion(n: int, grupos):
    if grupos is not None:
        return GroupKFold(n_splits=5)
    if n <= 40:
        return LeaveOneOut()
    return RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)


# --------------------------------------------------------------------------- evaluación
def evaluar(c: dict) -> list[dict]:
    df = c["df"].reset_index(drop=True)
    x = df[c["num"] + c["cat"]]
    y = df[c["respuesta"]].to_numpy(dtype=float)
    grupos = df[c["grupos"]].to_numpy() if c["grupos"] else None
    n = len(df)
    cv = particion(n, grupos)
    filas = []
    mae_base = None
    for nombre, modelo in modelos(c["num"], c["cat"], n).items():
        pred = np.full(n, np.nan)
        errores_pliegue = []
        for tr, te in cv.split(x, y, grupos):
            modelo.fit(x.iloc[tr], y[tr])
            p = modelo.predict(x.iloc[te])
            errores_pliegue.append(mean_absolute_error(y[te], p))
            # Con K-fold repetido cada punto se predice varias veces: se promedia.
            pred[te] = np.where(np.isnan(pred[te]), p, (pred[te] + p) / 2)
        mae = float(np.mean(errores_pliegue))
        if nombre.startswith("Línea base"):
            mae_base = mae
        filas.append(dict(campana=c["campana"], respuesta=c["respuesta"], unidad=c["unidad"], n=n,
                          validacion=type(cv).__name__, modelo=nombre, mae=mae,
                          mae_sd_pliegues=float(np.std(errores_pliegue))))
    for f in filas:
        f["destreza_S"] = 1 - f["mae"] / mae_base
    return filas


def main() -> None:
    resultados: list[dict] = []
    for c in campanas():
        print(f"\n{c['campana']} — {c['respuesta']} (n = {len(c['df'])})")
        filas = evaluar(c)
        filas_orden = sorted(filas, key=lambda f: f["mae"])
        for f in filas_orden:
            marca = " ◄" if f is filas_orden[0] else ""
            print(f"  {f['modelo']:30} MAE {f['mae']:9.4f} {f['unidad']:5}  S {f['destreza_S']:+.3f}{marca}")
        resultados.extend(filas)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)
    print(f"\nEscrito: {SALIDA}")

    # Resumen: ranking medio de cada familia entre campañas con señal (S del mejor ≥ 0.10).
    tabla = pd.DataFrame(resultados)
    con_senal = tabla.groupby(["campana", "respuesta"]).destreza_S.max()
    con_senal = con_senal[con_senal >= 0.10].index
    sub = tabla.set_index(["campana", "respuesta"]).loc[con_senal].reset_index()
    sub["rango"] = sub.groupby(["campana", "respuesta"]).mae.rank()
    resumen = sub[sub.modelo != "Línea base (mediana)"].groupby("modelo").agg(
        rango_medio=("rango", "mean"), destreza_media=("destreza_S", "mean"),
        veces_mejor=("rango", lambda r: int((r == 1).sum())))
    print("\nRESUMEN entre campañas con señal aprendible (S máx ≥ 0.10):")
    print(resumen.sort_values("rango_medio").round(3).to_string())


if __name__ == "__main__":
    main()
