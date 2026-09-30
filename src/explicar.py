"""Explicabilidad global y local del sustituto neuronal con valores de Shapley (SHAP).

Las explicaciones se calculan en el espacio de características del modelo (entradas
estandarizadas más indicadores de las categóricas) con el explicador por permutaciones de SHAP,
que es independiente del tipo de modelo, y se agregan de vuelta a las seis variables originales
sumando las contribuciones de los indicadores de cada categórica. Las contribuciones se
desestandarizan para expresarse en unidades físicas (MPa, µm, Shore D) respecto del valor base,
que es la predicción media sobre el conjunto de fondo.

Fondo: las combinaciones de PLA+ en impresora cerrada, que es el dominio de recomendación; así
la explicación local de una configuración responde a «por qué este punto difiere de la
combinación típica del dominio».

Como módulo: `explicar_local(df_fila)` devuelve las contribuciones por variable y respuesta, y
la interfaz lo usa para cada recomendación.

Como script: calcula la importancia global (media de |SHAP| por variable y respuesta), la guarda
en resultados/tablas/shap_global.csv con su figura, y explica tres configuraciones de interés:
la referencia proyectada, el óptimo de resistencia y la mejor candidata de la hipótesis.

    .venv/Scripts/python.exe proyecto/src/explicar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_sustituto import CATEGORICAS, NUMERICAS, RESPUESTAS, UNIDADES, Sustituto  # noqa: E402

DIR_MODELOS = ROOT / "proyecto" / "resultados" / "modelos"
DIR_TABLAS = ROOT / "proyecto" / "resultados" / "tablas"
DIR_FIGURAS = ROOT / "proyecto" / "resultados" / "figuras"
DATOS = ROOT / "proyecto" / "datos" / "procesados" / "aktepe_ergun_fdm_500.csv"
NOMBRES = {"altura_capa_mm": "Altura de capa", "densidad_relleno_pct": "Densidad de relleno", "velocidad_impresion_mms": "Velocidad",
           "patron_relleno": "Patrón de relleno", "tipo_impresora": "Tipo de impresora", "material": "Material"}
CONFIGS = {
    "referencia proyectada": dict(altura_capa_mm=0.2, densidad_relleno_pct=20.0, velocidad_impresion_mms=70.0, patron_relleno="grid", tipo_impresora="closed", material="PLA+"),
    "óptimo de resistencia": dict(altura_capa_mm=0.3, densidad_relleno_pct=80.0, velocidad_impresion_mms=30.0, patron_relleno="cubic", tipo_impresora="closed", material="PLA+"),
    "candidata de la hipótesis": dict(altura_capa_mm=0.1, densidad_relleno_pct=20.0, velocidad_impresion_mms=70.0, patron_relleno="cubic_subdivision", tipo_impresora="closed", material="PLA+"),
}


class Explicador:
    def __init__(self, modelo: Sustituto, fondo: pd.DataFrame, semilla: int = 20260929):
        self.m = modelo
        self.fondo_df = fondo.reset_index(drop=True)
        self.X_fondo = modelo._matriz(self.fondo_df, ajustar=False)
        # nombres de columnas del espacio del modelo y a qué variable original pertenece cada una
        cats = modelo.ohe.categories_
        self.columnas = list(NUMERICAS) + [f"{c}={v}" for c, vs in zip(CATEGORICAS, cats) for v in vs]
        self.origen = list(NUMERICAS) + [c for c, vs in zip(CATEGORICAS, cats) for _ in vs]
        self.explainer = shap.PermutationExplainer(modelo.red.predict, shap.maskers.Independent(self.X_fondo, max_samples=200), seed=semilla)
        self.base = modelo.esc_y.inverse_transform(modelo.red.predict(self.X_fondo).mean(axis=0, keepdims=True))[0]

    def valores(self, df: pd.DataFrame) -> np.ndarray:
        """SHAP en unidades físicas: (n, características, respuestas)."""
        X = self.m._matriz(df, ajustar=False)
        sv = self.explainer(X, max_evals=2 * X.shape[1] + 1).values      # (n, f, r) en unidades estandarizadas
        return sv * self.m.esc_y.scale_[None, None, :]

    def agregar(self, sv: np.ndarray) -> pd.DataFrame:
        """Suma las contribuciones de los indicadores de cada categórica: (n, 6 variables, r)."""
        vars_ = list(NUMERICAS) + list(CATEGORICAS)
        out = np.zeros((sv.shape[0], len(vars_), sv.shape[2]))
        for j, o in enumerate(self.origen):
            out[:, vars_.index(o), :] += sv[:, j, :]
        return out, vars_


def cargar() -> tuple[Sustituto, Explicador, pd.DataFrame]:
    modelo: Sustituto = joblib.load(DIR_MODELOS / "sustituto_mlp.joblib")
    datos = pd.read_csv(DATOS).dropna(subset=RESPUESTAS)
    fondo = datos[(datos.material == "PLA+") & (datos.tipo_impresora == "closed")]
    return modelo, Explicador(modelo, fondo), datos


def explicar_local(expl: Explicador, fila: dict) -> pd.DataFrame:
    """Contribución de cada variable a cada respuesta para una configuración, en unidades físicas."""
    df = pd.DataFrame([fila])
    sv, vars_ = expl.agregar(expl.valores(df))
    tabla = pd.DataFrame(sv[0], index=[NOMBRES[v] for v in vars_], columns=RESPUESTAS)
    tabla.loc["Valor base (media del dominio)"] = expl.base
    tabla.loc["Predicción"] = expl.base + sv[0].sum(axis=0)
    return tabla


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modelo, expl, datos = cargar()
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # ---- global: media de |SHAP| sobre el dominio de recomendación
    dom = expl.fondo_df
    sv, vars_ = expl.agregar(expl.valores(dom))
    imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=[NOMBRES[v] for v in vars_], columns=RESPUESTAS)
    imp_pct = 100 * imp / imp.sum(axis=0)
    salida = imp.copy()
    for r in RESPUESTAS:
        salida[f"{r}_pct"] = imp_pct[r]
    salida.to_csv(DIR_TABLAS / "shap_global.csv")
    print(f"IMPORTANCIA GLOBAL (media de |SHAP| sobre {len(dom)} combinaciones de PLA+ en cámara cerrada), unidades físicas y % del total:")
    print(pd.concat([imp.round(2), imp_pct.round(1).add_suffix(" %")], axis=1).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, r in zip(axes, RESPUESTAS):
        s = imp[r].sort_values()
        ax.barh(s.index, s.values, color="#4C72B0")
        ax.set_title({"tension_maxima_mpa": "Tensión máxima", "rugosidad_ra_um": "Rugosidad Ra", "dureza_shore": "Dureza"}[r])
        ax.set_xlabel(f"media de |SHAP| ({UNIDADES[r]})")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Importancia global de las variables en el sustituto, PLA+ en cámara cerrada")
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "shap_global.png", dpi=160)

    # ---- dependencia: cómo cambia la contribución con el nivel de cada numérica (para la discusión)
    dep = []
    for i, v in enumerate(NUMERICAS):
        for nivel, g in dom.groupby(v):
            idx = g.index.to_numpy()
            pos = dom.index.get_indexer(idx)
            dep.append(dict(variable=NOMBRES[v], nivel=nivel, n=len(pos), **{r: float(sv[pos, vars_.index(v), k].mean()) for k, r in enumerate(RESPUESTAS)}))
    pd.DataFrame(dep).to_csv(DIR_TABLAS / "shap_dependencia_niveles.csv", index=False)
    print("\nCONTRIBUCIÓN MEDIA POR NIVEL (unidades físicas):")
    print(pd.DataFrame(dep).round(2).to_string(index=False))

    # ---- local: tres configuraciones de interés
    fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(12, 3.2 * len(CONFIGS)))
    for fi, (nombre, cfg) in enumerate(CONFIGS.items()):
        tabla = explicar_local(expl, cfg)
        tabla.to_csv(DIR_TABLAS / f"shap_local_{nombre.split()[0]}.csv")
        print(f"\nEXPLICACIÓN LOCAL — {nombre}: capa {cfg['altura_capa_mm']} mm, relleno {cfg['densidad_relleno_pct']:.0f} %, {cfg['patron_relleno']}, {cfg['velocidad_impresion_mms']:.0f} mm/s")
        print(tabla.round(2).to_string())
        for k, r in enumerate(RESPUESTAS):
            ax = axes[fi, k]
            c = tabla[r].iloc[:6].sort_values()
            ax.barh(c.index, c.values, color=np.where(c.values >= 0, "#2A9D8F", "#E76F51"))
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{nombre}: {r.split('_')[0]} = {tabla[r]['Predicción']:.1f} {UNIDADES[r]} (base {expl.base[k]:.1f})", fontsize=9)
            ax.set_xlabel(f"contribución ({UNIDADES[r]})", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(DIR_FIGURAS / "shap_local.png", dpi=160)
    print(f"\nFiguras: {DIR_FIGURAS / 'shap_global.png'} y {DIR_FIGURAS / 'shap_local.png'}")


if __name__ == "__main__":
    main()
