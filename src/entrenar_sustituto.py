"""Entrena el modelo sustituto neuronal sobre el conjunto de Aktepe y Ergün.

Decisión del 22 de septiembre de 2026 (reunión con el asesor): el entrenamiento se hace
únicamente sobre el conjunto de Aktepe y Ergün (Mendeley 10.17632/zd6td6svd6.2), por ser el
más completo, y el modelo se acopla después a un optimizador exacto (véase optimizar_milp.py).

El sustituto es un perceptrón multicapa con activación ReLU y salida lineal. Se elige ReLU a
propósito: una red ReLU es una función lineal a trozos y puede codificarse **exactamente** como
un programa lineal entero mixto, de modo que el optimizador no aproxima la red, sino que la
resuelve tal cual. Ese es el enfoque híbrido acordado.

Entradas: altura de capa, densidad de relleno, velocidad (numéricas, estandarizadas) y patrón
de relleno, tipo de impresora y material (categóricas, one-hot). Salidas: tensión máxima (MPa),
rugosidad Ra (µm) y dureza Shore, estandarizadas durante el entrenamiento.

Protocolo: selección de arquitectura por validación cruzada 5 × 3 sobre las 483 filas con las
tres respuestas completas; destreza S = 1 − MAE/MAE_base contra la mediana, reportada global y
por material. Luego se reentrena la arquitectura elegida sobre todas las filas y se guardan los
pesos con los parámetros de escala en resultados/modelos/.

Ejecutar con el intérprete del entorno virtual:
    .venv/Scripts/python.exe proyecto/src/entrenar_sustituto.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEMILLA = 20260922
ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT / "proyecto" / "datos" / "procesados" / "aktepe_ergun_fdm_500.csv"
DIR_MODELOS = ROOT / "proyecto" / "resultados" / "modelos"
DIR_TABLAS = ROOT / "proyecto" / "resultados" / "tablas"

NUMERICAS = ["altura_capa_mm", "densidad_relleno_pct", "velocidad_impresion_mms"]
CATEGORICAS = ["patron_relleno", "tipo_impresora", "material"]
RESPUESTAS = ["tension_maxima_mpa", "rugosidad_ra_um", "dureza_shore"]
UNIDADES = {"tension_maxima_mpa": "MPa", "rugosidad_ra_um": "µm", "dureza_shore": "Shore D"}

ARQUITECTURAS = [(16,), (32,), (16, 8), (32, 16), (48, 24), (64, 32, 16)]
ALPHAS = [1e-4, 1e-3, 1e-2]


class Sustituto:
    """Red ReLU con escalado de entradas y salidas, y exportación de pesos para el MILP."""

    def __init__(self, capas: tuple[int, ...], alpha: float):
        self.capas = capas
        self.alpha = alpha
        self.esc_x = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.esc_y = StandardScaler()
        self.red = MLPRegressor(
            hidden_layer_sizes=capas, activation="relu", alpha=alpha, solver="lbfgs",
            max_iter=4000, random_state=SEMILLA,
        )

    def _matriz(self, df: pd.DataFrame, ajustar: bool) -> np.ndarray:
        if ajustar:
            xn = self.esc_x.fit_transform(df[NUMERICAS])
            xc = self.ohe.fit_transform(df[CATEGORICAS])
        else:
            xn = self.esc_x.transform(df[NUMERICAS])
            xc = self.ohe.transform(df[CATEGORICAS])
        return np.hstack([xn, xc])

    def fit(self, df: pd.DataFrame) -> "Sustituto":
        x = self._matriz(df, ajustar=True)
        y = self.esc_y.fit_transform(df[RESPUESTAS])
        self.red.fit(x, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.esc_y.inverse_transform(self.red.predict(self._matriz(df, ajustar=False)))

    def exportar(self) -> dict:
        """Todo lo que el MILP necesita para reproducir la red sin scikit-learn."""
        return {
            "numericas": NUMERICAS,
            "categoricas": CATEGORICAS,
            "categorias": {c: list(map(str, cats)) for c, cats in zip(CATEGORICAS, self.ohe.categories_)},
            "respuestas": RESPUESTAS,
            "x_media": self.esc_x.mean_.tolist(),
            "x_escala": self.esc_x.scale_.tolist(),
            "y_media": self.esc_y.mean_.tolist(),
            "y_escala": self.esc_y.scale_.tolist(),
            "pesos": [w.tolist() for w in self.red.coefs_],
            "sesgos": [b.tolist() for b in self.red.intercepts_],
            "capas": list(self.capas),
            "alpha": self.alpha,
        }


def destreza(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    mae = mean_absolute_error(y, pred)
    mae_base = mean_absolute_error(y, base)
    return mae, 1 - mae / mae_base


def validar(df: pd.DataFrame, capas, alpha) -> tuple[np.ndarray, np.ndarray]:
    """Predicción fuera de muestra (promedio de las repeticiones) y línea base por pliegue."""
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
    pred = np.zeros((len(df), len(RESPUESTAS)))
    base = np.zeros_like(pred)
    cuenta = np.zeros(len(df))
    for tr, te in cv.split(df):
        m = Sustituto(capas, alpha).fit(df.iloc[tr])
        pred[te] += m.predict(df.iloc[te])
        base[te] += df.iloc[tr][RESPUESTAS].median().to_numpy()
        cuenta[te] += 1
    return pred / cuenta[:, None], base / cuenta[:, None]


def main() -> None:
    df = pd.read_csv(DATOS).dropna(subset=RESPUESTAS).reset_index(drop=True)
    print(f"Filas con las tres respuestas: {len(df)} (de 500)")

    # --- selección de arquitectura: MAE estandarizado medio de las tres respuestas
    print("\nSelección de arquitectura (5 × 3 pliegues, MAE relativo medio sobre las tres respuestas):")
    resultados = []
    for capas in ARQUITECTURAS:
        for alpha in ALPHAS:
            pred, base = validar(df, capas, alpha)
            s = [destreza(df[r], pred[:, i], base[:, i])[1] for i, r in enumerate(RESPUESTAS)]
            resultados.append(dict(capas=capas, alpha=alpha, S_tension=s[0], S_ra=s[1], S_dureza=s[2], S_media=np.mean(s)))
            print(f"  capas={str(capas):14} alpha={alpha:<7} S tensión {s[0]:+.3f} | Ra {s[1]:+.3f} | dureza {s[2]:+.3f} | media {np.mean(s):+.3f}")
    mejor = max(resultados, key=lambda r: r["S_media"])
    print(f"\nElegida: capas={mejor['capas']} alpha={mejor['alpha']} (S media {mejor['S_media']:+.3f})")

    # --- métricas detalladas de la arquitectura elegida
    pred, base = validar(df, mejor["capas"], mejor["alpha"])
    filas = []
    for i, r in enumerate(RESPUESTAS):
        mae, s = destreza(df[r], pred[:, i], base[:, i])
        filas.append(dict(ambito="todas", material="todos", respuesta=r, unidad=UNIDADES[r], n=len(df), mae=mae, destreza_S=s))
        for mat, g in df.groupby("material"):
            if len(g) < 15:
                continue
            idx = g.index.to_numpy()
            mae_m, s_m = destreza(df.loc[idx, r], pred[idx, i], base[idx, i])
            filas.append(dict(ambito="por material", material=mat, respuesta=r, unidad=UNIDADES[r], n=len(idx), mae=mae_m, destreza_S=s_m))
    tabla = pd.DataFrame(filas)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DIR_TABLAS / "sustituto_mlp_validacion.csv", index=False)
    print("\nDestreza fuera de muestra del sustituto elegido (modelo único, todos los materiales):")
    print(tabla[tabla.ambito == "todas"][["respuesta", "n", "mae", "destreza_S"]].round(3).to_string(index=False))
    print("\nPor material (n ≥ 15):")
    print(tabla[tabla.ambito == "por material"].pivot(index="material", columns="respuesta", values="destreza_S").round(3).to_string())

    # --- comparación: ¿conviene un modelo solo de PLA+?
    pla = df[df.material == "PLA+"].reset_index(drop=True)
    pred_p, base_p = validar(pla, mejor["capas"], mejor["alpha"])
    print("\nContraste: mismo sustituto entrenado solo con PLA+ (n = %d):" % len(pla))
    for i, r in enumerate(RESPUESTAS):
        mae, s = destreza(pla[r], pred_p[:, i], base_p[:, i])
        fila_global = tabla[(tabla.material == "PLA+") & (tabla.respuesta == r)].iloc[0]
        print(f"  {r:20} solo PLA+: S {s:+.3f} (MAE {mae:.3f}) | modelo conjunto evaluado en PLA+: S {fila_global.destreza_S:+.3f} (MAE {fila_global.mae:.3f})")

    # --- modelo final
    final = Sustituto(mejor["capas"], mejor["alpha"]).fit(df)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, DIR_MODELOS / "sustituto_mlp.joblib")
    with (DIR_MODELOS / "sustituto_mlp.json").open("w", encoding="utf-8") as fh:
        json.dump(final.exportar(), fh, ensure_ascii=False, indent=1)
    print(f"\nModelo guardado en {DIR_MODELOS} (joblib para Python, json con pesos para el MILP)")


if __name__ == "__main__":
    main()
