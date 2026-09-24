"""Optimización exacta de parámetros de impresión sobre el sustituto neuronal (enfoque híbrido).

La red ReLU entrenada por entrenar_sustituto.py es lineal a trozos. Cada unidad oculta
h = max(z, 0) se codifica con una variable binaria a y cotas [L, U] de z obtenidas por
aritmética de intervalos:

    h >= 0,  h >= z,  h <= z - L·(1 - a),  h <= U·a

de modo que a = 1 fuerza h = z (unidad activa) y a = 0 fuerza h = 0. Con esto el problema
«maximizar la resistencia predicha sujeto a restricciones sobre las variables relevantes y sobre
las demás respuestas predichas» es un programa lineal entero mixto (MILP) que HiGHS resuelve
por simplex con ramificación y acotamiento hasta el óptimo global de la red. No hay aproximación
del modelo: el óptimo del MILP es el óptimo de la red.

Variables de decisión:
- Numéricas: altura de capa, densidad de relleno, velocidad. En modo «discreto» toman los
  niveles del diseño experimental (selección con binarias); en modo «continuo» se mueven dentro
  del rango de los datos.
- Categóricas: patrón de relleno y tipo de impresora (one-hot con suma 1), fijables. La impresora
  se fija por defecto en «closed»: la Creality K1C disponible para la validación es de cámara
  cerrada, igual que la K1 con la que Aktepe y Ergün produjeron 453 de las 500 combinaciones.
- Material: fijado (por defecto PLA+, el filamento disponible para la validación).

Verificación: en modo discreto el espacio es enumerable (3 × 3 × 3 × patrones × impresoras), así
que el script evalúa la red de scikit-learn en todas las combinaciones y comprueba que el MILP
devuelve el mismo óptimo. Es la prueba de que la codificación es exacta.

Salidas en resultados/tablas/ y resultados/figuras/: óptimo puntual, verificación y frontera
resistencia–rugosidad–material por barrido de restricciones (método epsilon).

Uso:
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py --ra-max 4 --densidad-max 50 --impresora closed
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py --continuo
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_sustituto import Sustituto  # noqa: E402  (necesario para cargar el joblib)

DIR_MODELOS = ROOT / "proyecto" / "resultados" / "modelos"
DIR_TABLAS = ROOT / "proyecto" / "resultados" / "tablas"
DIR_FIGURAS = ROOT / "proyecto" / "resultados" / "figuras"
DATOS = ROOT / "proyecto" / "datos" / "procesados" / "aktepe_ergun_fdm_500.csv"

NIVELES = {
    "altura_capa_mm": [0.1, 0.2, 0.3],
    "densidad_relleno_pct": [20.0, 50.0, 80.0],
    "velocidad_impresion_mms": [30.0, 50.0, 70.0],
}


def tiempo_relativo(altura: float, velocidad: float, densidad: float) -> float:
    """Proxy declarado de tiempo de impresión, normalizado a 1 en (0.2 mm, 50 mm/s, 50 %).

    Tiempo ∝ número de capas × longitud de trayectoria / velocidad. Sustituye a la estimación del
    laminador hasta que esta se mida sobre la geometría definitiva.
    """
    return (0.2 / altura) * (50.0 / velocidad) * (0.5 + densidad / 100.0)


class RedExportada:
    def __init__(self, ruta_json: Path):
        d = json.loads(ruta_json.read_text(encoding="utf-8"))
        self.numericas = d["numericas"]
        self.categoricas = d["categoricas"]
        self.categorias = d["categorias"]
        self.respuestas = d["respuestas"]
        self.x_media = np.array(d["x_media"])
        self.x_escala = np.array(d["x_escala"])
        self.y_media = np.array(d["y_media"])
        self.y_escala = np.array(d["y_escala"])
        self.pesos = [np.array(w) for w in d["pesos"]]
        self.sesgos = [np.array(b) for b in d["sesgos"]]
        self.n_num = len(self.numericas)
        self.n_cat = sum(len(v) for v in self.categorias.values())


def construir_milp(red: RedExportada, args) -> tuple[cp.Problem, dict]:
    v: dict = {}

    # ---- entradas numéricas
    if args.continuo:
        x = cp.Variable(red.n_num, name="x")
        cotas_x = np.array([[min(NIVELES[c]), max(NIVELES[c])] for c in red.numericas])
        restr = [x >= cotas_x[:, 0], x <= cotas_x[:, 1]]
        v["x"] = x
    else:
        x_expr, restr, cotas_x = [], [], []
        for c in red.numericas:
            niveles = np.array(NIVELES[c])
            b = cp.Variable(len(niveles), boolean=True, name=f"nivel_{c}")
            restr.append(cp.sum(b) == 1)
            x_expr.append(niveles @ b)
            cotas_x.append([niveles.min(), niveles.max()])
            v[f"nivel_{c}"] = b
        x = cp.hstack(x_expr)
        cotas_x = np.array(cotas_x)

    # restricciones sobre las variables relevantes
    idx = {c: i for i, c in enumerate(red.numericas)}
    if args.densidad_max is not None:
        restr.append(x[idx["densidad_relleno_pct"]] <= args.densidad_max)
    if args.altura_min is not None:
        restr.append(x[idx["altura_capa_mm"]] >= args.altura_min)
    if args.velocidad_min is not None:
        restr.append(x[idx["velocidad_impresion_mms"]] >= args.velocidad_min)

    # ---- categóricas one-hot
    onehot, cotas_c = [], []
    fijos = {"material": args.material, "tipo_impresora": args.impresora, "patron_relleno": args.patron}
    for c in red.categoricas:
        cats = red.categorias[c]
        b = cp.Variable(len(cats), boolean=True, name=f"cat_{c}")
        restr.append(cp.sum(b) == 1)
        if fijos.get(c) is not None:
            if fijos[c] not in cats:
                raise SystemExit(f"'{fijos[c]}' no es una categoría de {c}: {cats}")
            restr.append(b[cats.index(fijos[c])] == 1)
            cotas_c += [[float(k == cats.index(fijos[c]))] * 2 for k in range(len(cats))]
        else:
            cotas_c += [[0.0, 1.0]] * len(cats)
        onehot.append(b)
        v[f"cat_{c}"] = b

    # ---- estandarización y propagación de cotas (aritmética de intervalos)
    s = cp.hstack([cp.multiply(x - red.x_media, 1.0 / red.x_escala)] + onehot)
    lo = np.concatenate([(cotas_x[:, 0] - red.x_media) / red.x_escala, np.array(cotas_c)[:, 0]])
    hi = np.concatenate([(cotas_x[:, 1] - red.x_media) / red.x_escala, np.array(cotas_c)[:, 1]])

    h = s
    n_binarias_relu = 0
    for capa, (w, b) in enumerate(zip(red.pesos[:-1], red.sesgos[:-1])):
        z = h @ w + b
        wp, wn = np.maximum(w, 0), np.minimum(w, 0)
        z_lo = lo @ wp + hi @ wn + b
        z_hi = hi @ wp + lo @ wn + b
        a = cp.Variable(w.shape[1], boolean=True, name=f"relu_{capa}")
        hh = cp.Variable(w.shape[1], name=f"h_{capa}")
        for j in range(w.shape[1]):
            if z_hi[j] <= 0:            # unidad siempre inactiva
                restr.append(hh[j] == 0)
            elif z_lo[j] >= 0:          # unidad siempre activa
                restr.append(hh[j] == z[j])
            else:
                n_binarias_relu += 1
                restr += [hh[j] >= 0, hh[j] >= z[j],
                          hh[j] <= z[j] - z_lo[j] * (1 - a[j]),
                          hh[j] <= z_hi[j] * a[j]]
        h = hh
        lo, hi = np.maximum(z_lo, 0), np.maximum(z_hi, 0)
        v[f"relu_{capa}"] = a

    # cp.multiply es elemento a elemento; con «*» cvxpy haría un producto matricial.
    y = cp.multiply(h @ red.pesos[-1] + red.sesgos[-1], red.y_escala) + red.y_media
    r = {nombre: y[i] for i, nombre in enumerate(red.respuestas)}
    v.update(x=x, y=y, n_binarias_relu=n_binarias_relu)

    if args.ra_max is not None:
        restr.append(r["rugosidad_ra_um"] <= args.ra_max)
    if args.dureza_min is not None:
        restr.append(r["dureza_shore"] >= args.dureza_min)

    objetivo = r["tension_maxima_mpa"] - args.peso_ra * r["rugosidad_ra_um"] - args.peso_densidad * x[idx["densidad_relleno_pct"]]
    return cp.Problem(cp.Maximize(objetivo), restr), v


def leer_solucion(red: RedExportada, v: dict) -> dict:
    sol = {c: float(np.round(v["x"].value[i], 4)) for i, c in enumerate(red.numericas)}
    for c in red.categoricas:
        sol[c] = red.categorias[c][int(np.argmax(v[f"cat_{c}"].value))]
    for i, nombre in enumerate(red.respuestas):
        sol[f"pred_{nombre}"] = float(v["y"].value[i])
    sol["tiempo_relativo"] = tiempo_relativo(sol["altura_capa_mm"], sol["velocidad_impresion_mms"], sol["densidad_relleno_pct"])
    return sol


def resolver(red: RedExportada, args) -> dict | None:
    prob, v = construir_milp(red, args)
    t0 = time.perf_counter()
    prob.solve(solver=cp.HIGHS)
    dt = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    sol = leer_solucion(red, v)
    sol.update(objetivo=float(prob.value), estado=prob.status, segundos=dt, binarias_relu=v["n_binarias_relu"])
    return sol


def enumerar(modelo: Sustituto, red: RedExportada, args) -> pd.DataFrame:
    """Evalúa la red de scikit-learn en todas las combinaciones discretas admisibles."""
    cats = {c: ([f] if f is not None else red.categorias[c])
            for c, f in zip(red.categoricas, [args.patron, args.impresora, args.material])}
    filas = [dict(zip(red.numericas + red.categoricas, combo))
             for combo in itertools.product(*[NIVELES[c] for c in red.numericas], *[cats[c] for c in red.categoricas])]
    df = pd.DataFrame(filas)
    pred = modelo.predict(df)
    for i, nombre in enumerate(red.respuestas):
        df[f"pred_{nombre}"] = pred[:, i]
    df["tiempo_relativo"] = [tiempo_relativo(a, vel, d) for a, vel, d in zip(df.altura_capa_mm, df.velocidad_impresion_mms, df.densidad_relleno_pct)]
    df["objetivo"] = df.pred_tension_maxima_mpa - args.peso_ra * df.pred_rugosidad_ra_um - args.peso_densidad * df.densidad_relleno_pct
    adm = pd.Series(True, index=df.index)
    if args.ra_max is not None:
        adm &= df.pred_rugosidad_ra_um <= args.ra_max
    if args.densidad_max is not None:
        adm &= df.densidad_relleno_pct <= args.densidad_max
    if args.altura_min is not None:
        adm &= df.altura_capa_mm >= args.altura_min
    if args.velocidad_min is not None:
        adm &= df.velocidad_impresion_mms >= args.velocidad_min
    if args.dureza_min is not None:
        adm &= df.pred_dureza_shore >= args.dureza_min
    df["admisible"] = adm
    return df


def soporte(sol: dict, datos: pd.DataFrame) -> dict:
    """Cuántas filas de entrenamiento respaldan la combinación categórica del óptimo.

    El optimizador explota el sustituto donde este sea más alto, incluidas regiones sin datos.
    Un óptimo cuya combinación material–impresora–patrón no fue observada es extrapolación y
    debe declararse como tal (regla de dominio del anteproyecto).
    """
    d = datos[datos.material == sol["material"]]
    mismo_imp = d[d.tipo_impresora == sol["tipo_impresora"]]
    mismo_pat = mismo_imp[mismo_imp.patron_relleno == sol["patron_relleno"]]
    exacto = mismo_pat[(mismo_pat.altura_capa_mm == sol["altura_capa_mm"])
                       & (mismo_pat.densidad_relleno_pct == sol["densidad_relleno_pct"])
                       & (mismo_pat.velocidad_impresion_mms == sol["velocidad_impresion_mms"])]
    return dict(soporte_material=len(d), soporte_impresora=len(mismo_imp),
                soporte_impresora_patron=len(mismo_pat), soporte_exacto=len(exacto),
                tension_observada_exacta=float(exacto.tension_maxima_mpa.mean()) if len(exacto) else float("nan"))


def imprimir(sol: dict, titulo: str) -> None:
    print(f"\n{titulo}")
    print(f"  capa {sol['altura_capa_mm']} mm | relleno {sol['densidad_relleno_pct']} % | velocidad {sol['velocidad_impresion_mms']} mm/s | "
          f"patrón {sol['patron_relleno']} | impresora {sol['tipo_impresora']} | material {sol['material']}")
    print(f"  predicho: tensión {sol['pred_tension_maxima_mpa']:.2f} MPa | Ra {sol['pred_rugosidad_ra_um']:.2f} µm | "
          f"dureza {sol['pred_dureza_shore']:.1f} Shore D | tiempo relativo {sol['tiempo_relativo']:.2f}")
    if "soporte_impresora_patron" in sol:
        aviso = "" if sol["soporte_impresora_patron"] > 0 else "  ← EXTRAPOLACIÓN: combinación impresora–patrón sin datos"
        obs = f", observada {sol['tension_observada_exacta']:.2f} MPa" if sol["soporte_exacto"] else ""
        print(f"  soporte en datos: {sol['soporte_impresora']} filas con esa impresora, {sol['soporte_impresora_patron']} con ese patrón, "
              f"{sol['soporte_exacto']} idénticas{obs}{aviso}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", default="PLA+")
    p.add_argument("--impresora", default="closed",
                   help="closed (por defecto: la Creality K1C disponible es de cámara cerrada, como la K1 de Aktepe y Ergün) | open | libre")
    p.add_argument("--patron", default=None, help="fija el patrón de relleno; libre si se omite")
    p.add_argument("--ra-max", type=float, default=None, help="cota superior de Ra predicha (µm)")
    p.add_argument("--densidad-max", type=float, default=None, help="cota superior de densidad de relleno (%%), proxy de material")
    p.add_argument("--altura-min", type=float, default=None, help="cota inferior de altura de capa (mm), proxy de tiempo")
    p.add_argument("--velocidad-min", type=float, default=None, help="cota inferior de velocidad (mm/s), proxy de tiempo")
    p.add_argument("--dureza-min", type=float, default=None)
    p.add_argument("--peso-ra", type=float, default=0.0, help="penalización por µm de Ra en la puntuación ponderada")
    p.add_argument("--peso-densidad", type=float, default=0.0, help="penalización por %% de relleno en la puntuación ponderada")
    p.add_argument("--continuo", action="store_true", help="numéricas continuas dentro del rango de los datos")
    p.add_argument("--sin-barrido", action="store_true")
    args = p.parse_args()
    if args.impresora == "libre":
        args.impresora = None

    red = RedExportada(DIR_MODELOS / "sustituto_mlp.json")
    modelo: Sustituto = joblib.load(DIR_MODELOS / "sustituto_mlp.joblib")
    datos = pd.read_csv(DATOS)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # ---- 1. óptimo puntual
    sol = resolver(red, args)
    if sol is None:
        raise SystemExit("El MILP es infactible con esas restricciones.")
    sol.update(soporte(sol, datos))
    modo = ("continuo" if args.continuo else "discreto") + (f"_{args.impresora}" if args.impresora else "")
    imprimir(sol, f"ÓPTIMO MILP (modo {modo.split(chr(95))[0]}, HiGHS, {sol['segundos']*1000:.0f} ms, {sol['binarias_relu']} binarias ReLU)")

    # ---- 2. verificación por enumeración (solo tiene sentido en modo discreto)
    if not args.continuo:
        enum = enumerar(modelo, red, args)
        adm = enum[enum.admisible]
        mejor = adm.loc[adm.objetivo.idxmax()]
        dif = abs(mejor.objetivo - sol["objetivo"])
        print(f"\nVERIFICACIÓN: {len(enum)} combinaciones enumeradas, {len(adm)} admisibles.")
        print(f"  mejor por enumeración: objetivo {mejor.objetivo:.4f} | MILP: {sol['objetivo']:.4f} | diferencia {dif:.2e} "
              f"→ {'COINCIDEN' if dif < 1e-4 else 'NO COINCIDEN, revisar codificación'}")
        enum.sort_values("objetivo", ascending=False).to_csv(DIR_TABLAS / "milp_enumeracion.csv", index=False)
    else:
        # cordura: la predicción de scikit-learn en el punto óptimo debe reproducir la salida del MILP
        df_sol = pd.DataFrame([{k: sol[k] for k in red.numericas + red.categoricas}])
        pred = modelo.predict(df_sol)[0]
        dif = float(np.max(np.abs(pred - [sol[f"pred_{r}"] for r in red.respuestas])))
        print(f"\nCORDURA: predicción de scikit-learn en el óptimo continuo difiere del MILP en {dif:.2e} → {'OK' if dif < 1e-5 else 'REVISAR'}")

    pd.DataFrame([sol]).to_csv(DIR_TABLAS / f"milp_optimo_{modo}.csv", index=False)

    # ---- 3. frontera por barrido de restricciones (método epsilon)
    if not args.sin_barrido:
        filas = []
        for dmax in NIVELES["densidad_relleno_pct"]:
            for ra_max in np.arange(2.0, 8.01, 0.5):
                a2 = argparse.Namespace(**vars(args)); a2.densidad_max = dmax; a2.ra_max = float(ra_max)
                s2 = resolver(red, a2)
                if s2 is not None:
                    s2.update(restr_densidad_max=dmax, restr_ra_max=float(ra_max), **soporte(s2, datos))
                    filas.append(s2)
        front = pd.DataFrame(filas)
        # solo puntos no dominados en (−tensión, Ra, densidad)
        pts = front[["pred_tension_maxima_mpa", "pred_rugosidad_ra_um", "densidad_relleno_pct"]].to_numpy()
        nd = np.array([not np.any(np.all(np.c_[pts[:, 0] >= t, pts[:, 1] <= ra, pts[:, 2] <= d], axis=1)
                                  & np.any(np.c_[pts[:, 0] > t, pts[:, 1] < ra, pts[:, 2] < d], axis=1))
                       for t, ra, d in pts])
        front["no_dominado"] = nd
        front.to_csv(DIR_TABLAS / f"milp_frontera_{modo}.csv", index=False)
        nd_df = front[front.no_dominado].drop_duplicates(subset=red.numericas + red.categoricas)
        print(f"\nFRONTERA (método epsilon, {len(front)} resoluciones, {len(nd_df)} configuraciones no dominadas distintas):")
        cols = red.numericas + ["patron_relleno", "tipo_impresora", "pred_tension_maxima_mpa", "pred_rugosidad_ra_um", "pred_dureza_shore", "tiempo_relativo", "soporte_impresora_patron"]
        print(nd_df.sort_values("pred_tension_maxima_mpa", ascending=False)[cols].round(2).to_string(index=False))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for dmax, g in front.groupby("densidad_relleno_pct"):
            ax.scatter(g.pred_rugosidad_ra_um, g.pred_tension_maxima_mpa, label=f"relleno {dmax:.0f} %", s=28)
        ax.set_xlabel("Rugosidad Ra predicha (µm)"); ax.set_ylabel("Tensión máxima predicha (MPa)")
        ax.set_title(f"Frontera resistencia–rugosidad–material, {args.material}, modo {modo} (MILP sobre red ReLU)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(DIR_FIGURAS / f"milp_frontera_{modo}.png", dpi=160)
        print(f"\nFigura: {DIR_FIGURAS / f'milp_frontera_{modo}.png'}")


if __name__ == "__main__":
    main()
