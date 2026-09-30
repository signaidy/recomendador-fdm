"""Optimización exacta de parámetros de impresión sobre el sustituto neuronal (enfoque híbrido).

La red ReLU entrenada por entrenar_sustituto.py es lineal a trozos. Cada unidad oculta
h = max(z, 0) se codifica con una variable binaria a y cotas [L, U] de z obtenidas por
aritmética de intervalos:

    h >= 0,  h >= z,  h <= z - L·(1 - a),  h <= U·a

de modo que a = 1 fuerza h = z (unidad activa) y a = 0 fuerza h = 0. Con esto el problema
«maximizar la resistencia predicha sujeto a restricciones sobre las variables relevantes y sobre
las demás respuestas» es un programa lineal entero mixto (MILP) que HiGHS resuelve por simplex
con ramificación y acotamiento hasta el óptimo global de la red. No hay aproximación del modelo.

Tiempo y material (desde el 29 de septiembre de 2026). En modo discreto, cada una de las 270
combinaciones laminadas con OrcaSlicer 2.4.2 para la Creality K1C (estimar_tiempo_orca.py) tiene
una binaria de combinación z_c ligada a las binarias de nivel y de patrón; el tiempo estimado y
los gramos de filamento entran como sumas Σ t_c z_c y Σ m_c z_c, exactas y lineales. Los cuatro
patrones sin equivalente en OrcaSlicer (quarter cubic, cross, cross3d, octet) quedan fuera del
espacio admisible, porque no pueden laminarse ni imprimirse en el equipo disponible. La energía
se estima como potencia media documentada por tiempo (--potencia-w). En modo continuo no hay
tabla y se conserva el proxy `tiempo_relativo`.

Variables de decisión:
- Numéricas: altura de capa, densidad de relleno, velocidad. En modo «discreto» toman los
  niveles del diseño experimental; en modo «continuo» se mueven dentro del rango de los datos.
- Categóricas: patrón de relleno y tipo de impresora (one-hot con suma 1), fijables. La impresora
  se fija por defecto en «closed»: la Creality K1C disponible para la validación es de cámara
  cerrada, igual que la K1 con la que Aktepe y Ergün produjeron 453 de las 500 combinaciones.
- Material: fijado (por defecto PLA+, el filamento disponible para la validación).

Verificación: en modo discreto el espacio es enumerable, así que el script evalúa la red de
scikit-learn en todas las combinaciones y comprueba que el MILP devuelve el mismo óptimo.

Hipótesis: contra la configuración de referencia proyectada al envolvente de datos (numeral 4.3
del programa definitivo) se listan las soluciones no dominadas que mantienen resistencia y
rugosidad dentro de la degradación admisible y reducen tiempo en ≥ 20 % o material en ≥ 10 %
(numeral 4.4.5). Es la evaluación sobre predicción; la física se hace en la validación.

Salidas en resultados/tablas/ y resultados/figuras/: óptimo puntual, verificación, frontera por
método epsilon (barrido de cotas de rugosidad y de tiempo) y tabla de hipótesis.

Uso:
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py --ra-max 4 --tiempo-max 20 --sin-barrido
    .venv/Scripts/python.exe proyecto/src/optimizar_milp.py --peso-tiempo 0.5 --peso-material 2 --sin-barrido
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
TABLA_ORCA = ROOT / "proyecto" / "datos" / "procesados" / "orca_k1c_tiempo_material.csv"
VALIDACION = DIR_TABLAS / "sustituto_mlp_validacion.csv"

NIVELES = {
    "altura_capa_mm": [0.1, 0.2, 0.3],
    "densidad_relleno_pct": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    "velocidad_impresion_mms": [30.0, 50.0, 70.0],
}
# Rango observado en el conjunto de entrenamiento. Los niveles de densidad 10, 90 y 100 % no existen en los datos
# (Aktepe y Ergün usaron 20, 50 y 80 %): se admiten solo con --extrapolar y toda solución que los use se marca como
# extrapolación. Los niveles 30, 40, 60 y 70 % están dentro del rango y son interpolación del modelo.
RANGO_DATOS = {"altura_capa_mm": (0.1, 0.3), "densidad_relleno_pct": (20.0, 80.0), "velocidad_impresion_mms": (30.0, 70.0)}
CLAVE = ("altura_capa_mm", "densidad_relleno_pct", "patron_relleno", "velocidad_impresion_mms")


def tiempo_relativo(altura: float, velocidad: float, densidad: float) -> float:
    """Proxy de tiempo, normalizado a 1 en (0.2 mm, 50 mm/s, 50 %). Solo para el modo continuo."""
    return (0.2 / altura) * (50.0 / velocidad) * (0.5 + densidad / 100.0)


def cargar_orca() -> dict[tuple, tuple[float, float]] | None:
    """{(altura, densidad, patrón, velocidad): (tiempo_min, filamento_g)} desde el laminado real."""
    if not TABLA_ORCA.exists():
        return None
    t = pd.read_csv(TABLA_ORCA)
    if "tiempo_estimado_min" not in t or t.tiempo_estimado_min.isna().all():
        return None
    t = t.dropna(subset=["tiempo_estimado_min", "filamento_g"])
    return {(float(r.altura_capa_mm), float(r.densidad_relleno_pct), r.patron_relleno, float(r.velocidad_impresion_mms)):
            (float(r.tiempo_estimado_min), float(r.filamento_g)) for r in t.itertuples()}


PESOS_VACIOS = {"resistencia": 0.0, "calidad": 0.0, "material": 0.0, "tiempo": 0.0}
PERFILES = {
    "Resistencia máxima": {"resistencia": 1.0, "calidad": 0.0, "material": 0.0, "tiempo": 0.0},
    "Equilibrado": {"resistencia": 0.4, "calidad": 0.2, "material": 0.2, "tiempo": 0.2},
    "Económico": {"resistencia": 0.2, "calidad": 0.1, "material": 0.35, "tiempo": 0.35},
    "Acabado": {"resistencia": 0.2, "calidad": 0.6, "material": 0.1, "tiempo": 0.1},
}


def normalizar_pesos(pesos: dict) -> dict:
    tot = sum(max(0.0, v) for v in pesos.values())
    return {k: max(0.0, v) / tot for k, v in pesos.items()} if tot > 0 else dict(PESOS_VACIOS)


def rangos_dominio(enum: pd.DataFrame) -> dict:
    """Mínimo y máximo de cada objetivo sobre el dominio admisible: constantes de normalización.

    Se calculan una vez sobre el dominio completo (no sobre la consulta) para que el significado
    de los pesos no cambie entre ejecuciones, como exige el numeral 1.3.7 del marco teórico.
    """
    adm = enum[enum.admisible]
    r = {"resistencia": (float(adm.pred_tension_maxima_mpa.min()), float(adm.pred_tension_maxima_mpa.max())),
         "calidad": (float(adm.pred_rugosidad_ra_um.min()), float(adm.pred_rugosidad_ra_um.max()))}
    if "tiempo_min" in adm:
        r["material"] = (float(adm.filamento_g.min()), float(adm.filamento_g.max()))
        r["tiempo"] = (float(adm.tiempo_min.min()), float(adm.tiempo_min.max()))
    return r


def puntuar(df: pd.DataFrame, pesos: dict, rangos: dict) -> pd.Series:
    """S = w_r·R_n + w_q·Q_n − w_m·M_n − w_t·T_n con cada término normalizado a [0, 1].

    La energía se estima como potencia media por tiempo, de modo que su término normalizado es
    idéntico al de tiempo; el peso de energía se suma al de tiempo.
    """
    w = normalizar_pesos(pesos)

    def n(col, clave):
        lo, hi = rangos[clave]
        return (df[col] - lo) / (hi - lo) if hi > lo else 0.0

    S = w["resistencia"] * n("pred_tension_maxima_mpa", "resistencia") + w["calidad"] * (1 - n("pred_rugosidad_ra_um", "calidad"))
    if "material" in rangos and "filamento_g" in df:
        S = S - w["material"] * n("filamento_g", "material") - w["tiempo"] * n("tiempo_min", "tiempo")
    return S


def rodilla(pareto: pd.DataFrame, rangos: dict) -> pd.Series | None:
    """Solución de compromiso: punto de la frontera beneficio–costo más alejado de la cuerda entre extremos.

    Beneficio = media de resistencia y calidad normalizadas; costo = media de material y tiempo
    normalizados. Es la «rodilla» del marco teórico: a partir de ella una mejora marginal del
    beneficio exige un incremento desproporcionado del costo.
    """
    if pareto is None or len(pareto) < 3 or "tiempo_min" not in pareto:
        return None
    df = pareto.copy()

    def n(col, clave):
        lo, hi = rangos[clave]
        return (df[col] - lo) / (hi - lo) if hi > lo else 0.0

    df["beneficio"] = 0.5 * (n("pred_tension_maxima_mpa", "resistencia") + (1 - n("pred_rugosidad_ra_um", "calidad")))
    df["costo"] = 0.5 * (n("filamento_g", "material") + n("tiempo_min", "tiempo"))
    # frontera 2D beneficio-costo (no dominados en ese plano)
    keep = []
    for i, r in df.iterrows():
        dom = ((df.beneficio >= r.beneficio) & (df.costo <= r.costo) & ((df.beneficio > r.beneficio) | (df.costo < r.costo))).any()
        keep.append(not dom)
    f = df[keep].sort_values("costo")
    if len(f) < 3:
        return None
    a = f.iloc[0][["costo", "beneficio"]].to_numpy(float)
    b = f.iloc[-1][["costo", "beneficio"]].to_numpy(float)
    d = b - a
    norm = np.hypot(*d) or 1.0
    dist = [abs(d[0] * (a[1] - y) - (a[0] - x) * d[1]) / norm for x, y in f[["costo", "beneficio"]].to_numpy(float)]
    f = f.assign(distancia_cuerda=dist)
    return f.loc[f.distancia_cuerda.idxmax()]


def estabilidad(red, args, orca, sol: dict, pesos: dict, delta: float = 0.10) -> tuple[int, int]:
    """Cuántas de las perturbaciones ±delta de cada peso devuelven la misma configuración."""
    clave = ("altura_capa_mm", "densidad_relleno_pct", "patron_relleno", "velocidad_impresion_mms")
    base = tuple(sol[k] for k in clave)
    iguales = total = 0
    for k in pesos:
        if pesos[k] <= 0:
            continue
        for signo in (+1, -1):
            pp = dict(pesos)
            pp[k] = max(0.0, pp[k] * (1 + signo * delta))
            a2 = argparse.Namespace(**vars(args))
            a2.pesos = pp
            s2 = resolver(red, a2, orca)
            total += 1
            iguales += int(s2 is not None and tuple(s2[k2] for k2 in clave) == base)
    return iguales, total


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


def construir_milp(red: RedExportada, args, orca: dict | None) -> tuple[cp.Problem, dict]:
    v: dict = {}
    idx = {c: i for i, c in enumerate(red.numericas)}

    # ---- entradas numéricas
    if args.continuo:
        x = cp.Variable(red.n_num, name="x")
        cotas_x = np.array([[min(NIVELES[c]), max(NIVELES[c])] for c in red.numericas])
        if not args.extrapolar:
            for c, (lo, hi) in RANGO_DATOS.items():
                cotas_x[idx[c]] = [max(cotas_x[idx[c], 0], lo), min(cotas_x[idx[c], 1], hi)]
        restr = [x >= cotas_x[:, 0], x <= cotas_x[:, 1]]
        niveles_b = {}
    else:
        x_expr, restr, cotas_x, niveles_b = [], [], [], {}
        for c in red.numericas:
            niveles = np.array(NIVELES[c])
            b = cp.Variable(len(niveles), boolean=True, name=f"nivel_{c}")
            restr.append(cp.sum(b) == 1)
            x_expr.append(niveles @ b)
            cotas_x.append([niveles.min(), niveles.max()])
            niveles_b[c] = b
            v[f"nivel_{c}"] = b
        x = cp.hstack(x_expr)
        cotas_x = np.array(cotas_x)
        if not args.extrapolar:
            for c, (lo, hi) in RANGO_DATOS.items():
                for k, nivel in enumerate(NIVELES[c]):
                    if not (lo - 1e-9 <= nivel <= hi + 1e-9):
                        restr.append(niveles_b[c][k] == 0)

    if args.densidad_max is not None:
        restr.append(x[idx["densidad_relleno_pct"]] <= args.densidad_max)
    if args.altura_min is not None:
        restr.append(x[idx["altura_capa_mm"]] >= args.altura_min)
    if args.velocidad_min is not None:
        restr.append(x[idx["velocidad_impresion_mms"]] >= args.velocidad_min)

    # ---- categóricas one-hot
    onehot, cotas_c, cat_b = [], [], {}
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
        cat_b[c] = b
        v[f"cat_{c}"] = b

    # ---- tabla del laminador: binarias de combinación (solo modo discreto)
    tiempo = material = None
    if orca is not None and not args.continuo and not args.sin_tiempo:
        pats = red.categorias["patron_relleno"]
        combos = [k for k in orca if k[2] in pats]
        z = cp.Variable(len(combos), boolean=True, name="combo")
        restr.append(cp.sum(z) == 1)
        for j, (h, dd, pat, vel) in enumerate(combos):
            restr += [z[j] <= niveles_b["altura_capa_mm"][NIVELES["altura_capa_mm"].index(h)],
                      z[j] <= niveles_b["densidad_relleno_pct"][NIVELES["densidad_relleno_pct"].index(dd)],
                      z[j] <= niveles_b["velocidad_impresion_mms"][NIVELES["velocidad_impresion_mms"].index(vel)],
                      z[j] <= cat_b["patron_relleno"][pats.index(pat)]]
        t_c = np.array([orca[k][0] for k in combos])
        m_c = np.array([orca[k][1] for k in combos])
        tiempo = t_c @ z
        material = m_c @ z
        v.update(combo=z, combos=combos)
        if args.tiempo_max is not None:
            restr.append(tiempo <= args.tiempo_max)
        if args.material_max is not None:
            restr.append(material <= args.material_max)

    # ---- estandarización y propagación de cotas (aritmética de intervalos)
    s = cp.hstack([cp.multiply(x - red.x_media, 1.0 / red.x_escala)] + onehot)
    lo = np.concatenate([(cotas_x[:, 0] - red.x_media) / red.x_escala, np.array(cotas_c)[:, 0]])
    hi = np.concatenate([(cotas_x[:, 1] - red.x_media) / red.x_escala, np.array(cotas_c)[:, 1]])

    h = s
    n_binarias_relu = 0
    for capa, (w, b) in enumerate(zip(red.pesos[:-1], red.sesgos[:-1])):
        zz = h @ w + b
        wp, wn = np.maximum(w, 0), np.minimum(w, 0)
        z_lo = lo @ wp + hi @ wn + b
        z_hi = hi @ wp + lo @ wn + b
        a = cp.Variable(w.shape[1], boolean=True, name=f"relu_{capa}")
        hh = cp.Variable(w.shape[1], name=f"h_{capa}")
        for j in range(w.shape[1]):
            if z_hi[j] <= 0:
                restr.append(hh[j] == 0)
            elif z_lo[j] >= 0:
                restr.append(hh[j] == zz[j])
            else:
                n_binarias_relu += 1
                restr += [hh[j] >= 0, hh[j] >= zz[j],
                          hh[j] <= zz[j] - z_lo[j] * (1 - a[j]),
                          hh[j] <= z_hi[j] * a[j]]
        h = hh
        lo, hi = np.maximum(z_lo, 0), np.maximum(z_hi, 0)
        v[f"relu_{capa}"] = a

    # cp.multiply es elemento a elemento; con «*» cvxpy haría un producto matricial.
    y = cp.multiply(h @ red.pesos[-1] + red.sesgos[-1], red.y_escala) + red.y_media
    r = {nombre: y[i] for i, nombre in enumerate(red.respuestas)}
    v.update(x=x, y=y, n_binarias_relu=n_binarias_relu, tiempo=tiempo, material=material)

    if args.ra_max is not None:
        restr.append(r["rugosidad_ra_um"] <= args.ra_max)
    if args.dureza_min is not None:
        restr.append(r["dureza_shore"] >= args.dureza_min)

    pesos = normalizar_pesos(getattr(args, "pesos", None) or PESOS_VACIOS)
    rangos = getattr(args, "rangos", None)
    if sum(pesos.values()) > 0 and rangos:
        # Puntuación ponderada de respuestas normalizadas (numeral 1.3.7): lineal en las respuestas
        # porque los rangos son constantes del dominio, así que el MILP la maximiza de forma exacta.
        def n(expr, clave):
            lo, hi = rangos[clave]
            return (expr - lo) / (hi - lo) if hi > lo else 0.0
        objetivo = pesos["resistencia"] * n(r["tension_maxima_mpa"], "resistencia") + pesos["calidad"] * (1 - n(r["rugosidad_ra_um"], "calidad"))
        if tiempo is not None and "material" in rangos:
            objetivo = objetivo - pesos["material"] * n(material, "material") - pesos["tiempo"] * n(tiempo, "tiempo")
    else:
        objetivo = r["tension_maxima_mpa"] - args.peso_ra * r["rugosidad_ra_um"] - args.peso_densidad * x[idx["densidad_relleno_pct"]]
        if tiempo is not None:
            objetivo = objetivo - args.peso_tiempo * tiempo - args.peso_material * material
    return cp.Problem(cp.Maximize(objetivo), restr), v


def leer_solucion(red: RedExportada, v: dict, args, orca: dict | None) -> dict:
    sol = {c: float(np.round(v["x"].value[i], 4)) for i, c in enumerate(red.numericas)}
    for c in red.categoricas:
        sol[c] = red.categorias[c][int(np.argmax(v[f"cat_{c}"].value))]
    for i, nombre in enumerate(red.respuestas):
        sol[f"pred_{nombre}"] = float(v["y"].value[i])
    sol["tiempo_relativo"] = tiempo_relativo(sol["altura_capa_mm"], sol["velocidad_impresion_mms"], sol["densidad_relleno_pct"])
    if v.get("tiempo") is not None:
        sol["tiempo_min"] = float(v["tiempo"].value)
        sol["filamento_g"] = float(v["material"].value)
    elif orca is not None:
        k = (sol["altura_capa_mm"], sol["densidad_relleno_pct"], sol["patron_relleno"], sol["velocidad_impresion_mms"])
        sol["tiempo_min"], sol["filamento_g"] = orca.get(k, (float("nan"), float("nan")))
    if "tiempo_min" in sol and args.potencia_w:
        sol["energia_wh"] = args.potencia_w * sol["tiempo_min"] / 60.0
    return sol


def resolver(red: RedExportada, args, orca: dict | None) -> dict | None:
    prob, v = construir_milp(red, args, orca)
    t0 = time.perf_counter()
    prob.solve(solver=cp.HIGHS)
    dt = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    sol = leer_solucion(red, v, args, orca)
    sol.update(objetivo=float(prob.value), estado=prob.status, segundos=dt, binarias_relu=v["n_binarias_relu"])
    return sol


def enumerar(modelo: Sustituto, red: RedExportada, args, orca: dict | None) -> pd.DataFrame:
    """Evalúa la red de scikit-learn en todas las combinaciones discretas admisibles."""
    cats = {c: ([f] if f is not None else red.categorias[c])
            for c, f in zip(red.categoricas, [args.patron, args.impresora, args.material])}
    filas = [dict(zip(red.numericas + red.categoricas, combo))
             for combo in itertools.product(*[NIVELES[c] for c in red.numericas], *[cats[c] for c in red.categoricas])]
    df = pd.DataFrame(filas)
    if not args.extrapolar:
        for c, (lo, hi) in RANGO_DATOS.items():
            df = df[(df[c] >= lo - 1e-9) & (df[c] <= hi + 1e-9)]
        df = df.reset_index(drop=True)
    pred = modelo.predict(df)
    for i, nombre in enumerate(red.respuestas):
        df[f"pred_{nombre}"] = pred[:, i]
    df["tiempo_relativo"] = [tiempo_relativo(a, vel, d) for a, vel, d in zip(df.altura_capa_mm, df.velocidad_impresion_mms, df.densidad_relleno_pct)]
    con_tabla = orca is not None and not args.sin_tiempo
    if con_tabla:
        tm = [orca.get((a, d, p, vel), (np.nan, np.nan)) for a, d, p, vel in zip(df.altura_capa_mm, df.densidad_relleno_pct, df.patron_relleno, df.velocidad_impresion_mms)]
        df["tiempo_min"] = [t for t, _ in tm]
        df["filamento_g"] = [m for _, m in tm]
    pesos = normalizar_pesos(getattr(args, "pesos", None) or PESOS_VACIOS)
    rangos = getattr(args, "rangos", None)
    if sum(pesos.values()) > 0 and rangos:
        df["objetivo"] = puntuar(df, pesos, rangos)
    else:
        df["objetivo"] = df.pred_tension_maxima_mpa - args.peso_ra * df.pred_rugosidad_ra_um - args.peso_densidad * df.densidad_relleno_pct
        if con_tabla:
            df["objetivo"] -= args.peso_tiempo * df.tiempo_min + args.peso_material * df.filamento_g
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
    if con_tabla:
        adm &= df.tiempo_min.notna()
        if args.tiempo_max is not None:
            adm &= df.tiempo_min <= args.tiempo_max
        if args.material_max is not None:
            adm &= df.filamento_g <= args.material_max
    df["admisible"] = adm
    return df


def soporte(sol: dict, datos: pd.DataFrame) -> dict:
    """Cuántas filas de entrenamiento respaldan la combinación categórica del óptimo."""
    d = datos[datos.material == sol["material"]]
    mismo_imp = d[d.tipo_impresora == sol["tipo_impresora"]]
    mismo_pat = mismo_imp[mismo_imp.patron_relleno == sol["patron_relleno"]]
    exacto = mismo_pat[(mismo_pat.altura_capa_mm == sol["altura_capa_mm"])
                       & (mismo_pat.densidad_relleno_pct == sol["densidad_relleno_pct"])
                       & (mismo_pat.velocidad_impresion_mms == sol["velocidad_impresion_mms"])]
    fuera = [k for k, (lo, hi) in RANGO_DATOS.items() if not (lo - 1e-9 <= sol[k] <= hi + 1e-9)]
    return dict(soporte_material=len(d), soporte_impresora=len(mismo_imp),
                soporte_impresora_patron=len(mismo_pat), soporte_exacto=len(exacto),
                tension_observada_exacta=float(exacto.tension_maxima_mpa.mean()) if len(exacto) else float("nan"),
                fuera_rango_entrenamiento=";".join(fuera))


def imprimir(sol: dict, titulo: str) -> None:
    print(f"\n{titulo}")
    print(f"  capa {sol['altura_capa_mm']} mm | relleno {sol['densidad_relleno_pct']} % | velocidad {sol['velocidad_impresion_mms']} mm/s | "
          f"patrón {sol['patron_relleno']} | impresora {sol['tipo_impresora']} | material {sol['material']}")
    extra = f" | tiempo {sol['tiempo_min']:.1f} min | filamento {sol['filamento_g']:.2f} g" if "tiempo_min" in sol and np.isfinite(sol["tiempo_min"]) else f" | tiempo relativo {sol['tiempo_relativo']:.2f}"
    if "energia_wh" in sol:
        extra += f" | energía {sol['energia_wh']:.1f} Wh"
    print(f"  predicho: tensión {sol['pred_tension_maxima_mpa']:.2f} MPa | Ra {sol['pred_rugosidad_ra_um']:.2f} µm | "
          f"dureza {sol['pred_dureza_shore']:.1f} Shore D{extra}")
    if "soporte_impresora_patron" in sol:
        aviso = "" if sol["soporte_impresora_patron"] > 0 else "  ← EXTRAPOLACIÓN: combinación impresora–patrón sin datos"
        obs = f", observada {sol['tension_observada_exacta']:.2f} MPa" if sol["soporte_exacto"] else ""
        print(f"  soporte en datos: {sol['soporte_impresora']} filas con esa impresora, {sol['soporte_impresora_patron']} con ese patrón, "
              f"{sol['soporte_exacto']} idénticas{obs}{aviso}")
        if sol.get("fuera_rango_entrenamiento"):
            print(f"  ← EXTRAPOLACIÓN: {sol['fuera_rango_entrenamiento']} fuera del rango de entrenamiento; la predicción es una extensión lineal del modelo")


def no_dominados(front: pd.DataFrame, cols_min: list[str], cols_max: list[str]) -> np.ndarray:
    M = np.c_[-front[cols_max].to_numpy(), front[cols_min].to_numpy()]     # todo a minimizar
    nd = []
    for i in range(len(M)):
        dom = np.all(M <= M[i], axis=1) & np.any(M < M[i], axis=1)
        nd.append(not dom.any())
    return np.array(nd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", default="PLA+")
    p.add_argument("--impresora", default="closed",
                   help="closed (por defecto: la Creality K1C disponible es de cámara cerrada, como la K1 de Aktepe y Ergün) | open | libre")
    p.add_argument("--patron", default=None, help="fija el patrón de relleno; libre si se omite")
    p.add_argument("--ra-max", type=float, default=None, help="cota superior de Ra predicha (µm)")
    p.add_argument("--densidad-max", type=float, default=None, help="cota superior de densidad de relleno (%%)")
    p.add_argument("--altura-min", type=float, default=None, help="cota inferior de altura de capa (mm)")
    p.add_argument("--velocidad-min", type=float, default=None, help="cota inferior de velocidad (mm/s)")
    p.add_argument("--dureza-min", type=float, default=None)
    p.add_argument("--tiempo-max", type=float, default=None, help="cota superior del tiempo estimado por OrcaSlicer (min)")
    p.add_argument("--material-max", type=float, default=None, help="cota superior del filamento estimado (g)")
    p.add_argument("--potencia-w", type=float, default=None, help="potencia media documentada (W) para estimar energía = P·t")
    p.add_argument("--peso-ra", type=float, default=0.0, help="penalización por µm de Ra en la puntuación ponderada")
    p.add_argument("--peso-densidad", type=float, default=0.0, help="penalización por %% de relleno")
    p.add_argument("--peso-tiempo", type=float, default=0.0, help="penalización por minuto de impresión (MPa/min)")
    p.add_argument("--peso-material", type=float, default=0.0, help="penalización por gramo de filamento (MPa/g)")
    p.add_argument("--referencia", default="0.2,20,grid,70", help="configuración de referencia proyectada: capa,relleno,patrón,velocidad")
    p.add_argument("--w-resistencia", type=float, default=0.0, help="peso de la resistencia en la puntuación ponderada")
    p.add_argument("--w-calidad", type=float, default=0.0, help="peso de la calidad (rugosidad baja)")
    p.add_argument("--w-material", type=float, default=0.0, help="peso del filamento")
    p.add_argument("--w-tiempo", type=float, default=0.0, help="peso del tiempo (y de la energía, que es potencia por tiempo)")
    p.add_argument("--perfil", choices=list(PERFILES), default=None, help="conjunto de pesos de ejemplo; sustituye a --w-*")
    p.add_argument("--continuo", action="store_true", help="numéricas continuas dentro del rango de los datos")
    p.add_argument("--extrapolar", "--relleno-100", dest="extrapolar", action="store_true",
                   help="admite niveles fuera del rango de entrenamiento (densidad 10, 90 y 100 %%); toda solución que los use se marca como extrapolación")
    p.add_argument("--sin-tiempo", action="store_true", help="ignora la tabla del laminador (comportamiento anterior al 29-09-2026)")
    p.add_argument("--sin-barrido", action="store_true")
    args = p.parse_args()
    if args.impresora == "libre":
        args.impresora = None
    args.pesos = dict(PERFILES[args.perfil]) if args.perfil else {"resistencia": args.w_resistencia, "calidad": args.w_calidad, "material": args.w_material, "tiempo": args.w_tiempo}
    args.rangos = None

    red = RedExportada(DIR_MODELOS / "sustituto_mlp.json")
    modelo: Sustituto = joblib.load(DIR_MODELOS / "sustituto_mlp.joblib")
    datos = pd.read_csv(DATOS)
    orca = cargar_orca()
    if orca is None:
        print("AVISO: no hay tabla del laminador; tiempo y material se omiten (proxy tiempo_relativo).")
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    modo = ("continuo" if args.continuo else "discreto") + (f"_{args.impresora}" if args.impresora else "")
    if args.extrapolar:
        modo += "_extrapolar"
    if not args.continuo:
        # rangos de normalización sobre el dominio completo (sin las cotas de la consulta)
        a_dom = argparse.Namespace(**vars(args))
        for k in ("ra_max", "densidad_max", "altura_min", "velocidad_min", "dureza_min", "tiempo_max", "material_max"):
            setattr(a_dom, k, None)
        args.rangos = rangos_dominio(enumerar(modelo, red, a_dom, orca))
        if sum(normalizar_pesos(args.pesos).values()) > 0:
            w = normalizar_pesos(args.pesos)
            print("PUNTUACIÓN PONDERADA: pesos " + ", ".join(f"{k} {v:.2f}" for k, v in w.items()) + " | rangos " + ", ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in args.rangos.items()))

    # ---- 1. óptimo puntual
    sol = resolver(red, args, orca)
    if sol is None:
        raise SystemExit("El MILP es infactible con esas restricciones.")
    sol.update(soporte(sol, datos))
    imprimir(sol, f"ÓPTIMO MILP (modo {modo.split(chr(95))[0]}, HiGHS, {sol['segundos']*1000:.0f} ms, {sol['binarias_relu']} binarias ReLU)")
    if args.rangos and sum(normalizar_pesos(args.pesos).values()) > 0:
        ig, tot = estabilidad(red, args, orca, sol, normalizar_pesos(args.pesos))
        print(f"  puntuación S = {sol['objetivo']:.3f} | estabilidad: la misma configuración gana en {ig} de {tot} perturbaciones de ±10 % de los pesos")

    # ---- 2. verificación por enumeración (solo tiene sentido en modo discreto)
    if not args.continuo:
        enum = enumerar(modelo, red, args, orca)
        adm = enum[enum.admisible]
        mejor = adm.loc[adm.objetivo.idxmax()]
        dif = abs(mejor.objetivo - sol["objetivo"])
        print(f"\nVERIFICACIÓN: {len(enum)} combinaciones enumeradas, {len(adm)} admisibles.")
        print(f"  mejor por enumeración: objetivo {mejor.objetivo:.4f} | MILP: {sol['objetivo']:.4f} | diferencia {dif:.2e} "
              f"→ {'COINCIDEN' if dif < 1e-4 else 'NO COINCIDEN, revisar codificación'}")
        enum.sort_values("objetivo", ascending=False).to_csv(DIR_TABLAS / "milp_enumeracion.csv", index=False)
    else:
        df_sol = pd.DataFrame([{k: sol[k] for k in red.numericas + red.categoricas}])
        pred = modelo.predict(df_sol)[0]
        dif = float(np.max(np.abs(pred - [sol[f"pred_{r}"] for r in red.respuestas])))
        print(f"\nCORDURA: predicción de scikit-learn en el óptimo continuo difiere del MILP en {dif:.2e} → {'OK' if dif < 1e-5 else 'REVISAR'}")

    pd.DataFrame([sol]).to_csv(DIR_TABLAS / f"milp_optimo_{modo}.csv", index=False)

    # ---- 3. frontera por barrido de restricciones (método epsilon)
    if not args.sin_barrido:
        con_tabla = orca is not None and not args.continuo and not args.sin_tiempo
        filas = []
        ra_caps = np.arange(2.0, 8.01, 0.5)
        if con_tabla:
            t_vals = np.array([t for t, _ in orca.values()])
            t_caps = np.arange(np.ceil(t_vals.min()), np.ceil(t_vals.max()) + 1, 3.0)
            barrido = [(ra, tc) for ra in ra_caps for tc in t_caps]
        else:
            lo_d, hi_d = RANGO_DATOS["densidad_relleno_pct"]
            barrido = [(ra, dmax) for ra in ra_caps for dmax in NIVELES["densidad_relleno_pct"]
                       if args.extrapolar or lo_d <= dmax <= hi_d]
        for ra_max, otro in barrido:
            a2 = argparse.Namespace(**vars(args))
            a2.ra_max = float(ra_max)
            if con_tabla:
                a2.tiempo_max = float(otro)
            else:
                a2.densidad_max = float(otro)
            s2 = resolver(red, a2, orca)
            if s2 is not None:
                s2.update(restr_ra_max=float(ra_max), restr_otro=float(otro), **soporte(s2, datos))
                filas.append(s2)
        front = pd.DataFrame(filas)
        if con_tabla:
            front["no_dominado"] = no_dominados(front, ["pred_rugosidad_ra_um", "tiempo_min", "filamento_g"], ["pred_tension_maxima_mpa"])
        else:
            front["no_dominado"] = no_dominados(front, ["pred_rugosidad_ra_um", "densidad_relleno_pct"], ["pred_tension_maxima_mpa"])
        front.to_csv(DIR_TABLAS / f"milp_frontera_{modo}.csv", index=False)
        nd_df = front[front.no_dominado].drop_duplicates(subset=red.numericas + red.categoricas)
        print(f"\nFRONTERA (método epsilon, {len(front)} resoluciones, {len(nd_df)} configuraciones no dominadas distintas):")
        cols = red.numericas + ["patron_relleno", "pred_tension_maxima_mpa", "pred_rugosidad_ra_um", "pred_dureza_shore"]
        cols += ["tiempo_min", "filamento_g"] if con_tabla else ["tiempo_relativo"]
        cols += ["soporte_impresora_patron"]
        print(nd_df.sort_values("pred_tension_maxima_mpa", ascending=False)[cols].round(2).to_string(index=False))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for dmax, g in front.groupby("densidad_relleno_pct"):
            ax.scatter(g.pred_rugosidad_ra_um, g.pred_tension_maxima_mpa, label=f"relleno {dmax:.0f} %", s=28)
        ax.set_xlabel("Rugosidad Ra predicha (µm)"); ax.set_ylabel("Tensión máxima predicha (MPa)")
        ax.set_title(f"Frontera resistencia–rugosidad–relleno, {args.material}, modo {modo} (MILP sobre red ReLU)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(DIR_FIGURAS / f"milp_frontera_{modo}.png", dpi=160)
        print(f"\nFigura: {DIR_FIGURAS / f'milp_frontera_{modo}.png'}")

        if con_tabla:
            # Conjunto de Pareto EXACTO: en modo discreto las 270 combinaciones laminables están
            # enumeradas con su predicción, así que la no dominancia en cuatro objetivos se calcula
            # sobre todas ellas. El barrido épsilon anterior demuestra el MILP con cotas; este es el
            # resultado definitivo para el capítulo 4.
            adm_all = enum[enum.admisible].copy()
            adm_all["no_dominado"] = no_dominados(adm_all, ["pred_rugosidad_ra_um", "tiempo_min", "filamento_g"], ["pred_tension_maxima_mpa"])
            pareto = adm_all[adm_all.no_dominado].copy()
            pareto = pareto.merge(pd.DataFrame([dict(zip(["altura_capa_mm", "densidad_relleno_pct", "patron_relleno", "velocidad_impresion_mms"], k)) for k in orca]), how="left")
            pareto["soporte_impresora_patron"] = [soporte(r, datos)["soporte_impresora_patron"] for r in pareto.to_dict("records")]
            pareto.sort_values("pred_tension_maxima_mpa", ascending=False).to_csv(DIR_TABLAS / f"milp_pareto_exacto_{modo}.csv", index=False)
            print(f"\nPARETO EXACTO por enumeración (cuatro objetivos: tensión, Ra, tiempo, filamento): {len(pareto)} de {len(adm_all)} combinaciones laminables son no dominadas.")
            print(pareto.sort_values("pred_tension_maxima_mpa", ascending=False)[red.numericas + ["patron_relleno", "pred_tension_maxima_mpa", "pred_rugosidad_ra_um", "tiempo_min", "filamento_g", "soporte_impresora_patron"]].round(2).to_string(index=False))
            nd_df = pareto
            rod = rodilla(pareto, args.rangos)
            if rod is not None:
                print(f"\nSOLUCIÓN DE RODILLA (máxima distancia a la cuerda beneficio–costo): capa {rod.altura_capa_mm} mm, relleno {rod.densidad_relleno_pct:.0f} %, "
                      f"{rod.patron_relleno}, {rod.velocidad_impresion_mms:.0f} mm/s → {rod.pred_tension_maxima_mpa:.1f} MPa, Ra {rod.pred_rugosidad_ra_um:.1f} µm, "
                      f"{rod.tiempo_min:.1f} min, {rod.filamento_g:.2f} g (beneficio {rod.beneficio:.2f}, costo {rod.costo:.2f})")
                pd.DataFrame([rod]).to_csv(DIR_TABLAS / f"milp_rodilla_{modo}.csv", index=False)
                for nombre, w in PERFILES.items():
                    pareto[f"S_{nombre}"] = puntuar(pareto, w, args.rangos)
                pareto.sort_values("pred_tension_maxima_mpa", ascending=False).to_csv(DIR_TABLAS / f"milp_pareto_exacto_{modo}.csv", index=False)
                print("MEJOR DEL PARETO POR PERFIL DE PESOS:")
                for nombre in PERFILES:
                    b = pareto.loc[pareto[f"S_{nombre}"].idxmax()]
                    print(f"  {nombre:20}: capa {b.altura_capa_mm} mm, relleno {b.densidad_relleno_pct:.0f} %, {b.patron_relleno}, {b.velocidad_impresion_mms:.0f} mm/s → "
                          f"{b.pred_tension_maxima_mpa:.1f} MPa, Ra {b.pred_rugosidad_ra_um:.1f} µm, {b.tiempo_min:.1f} min, {b.filamento_g:.2f} g, S = {b[f'S_{nombre}']:.3f}")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sc = ax.scatter(nd_df.tiempo_min, nd_df.pred_tension_maxima_mpa, c=nd_df.pred_rugosidad_ra_um, s=18 + 12 * (nd_df.filamento_g - nd_df.filamento_g.min()), cmap="viridis")
            fig.colorbar(sc, ax=ax, label="Rugosidad Ra predicha (µm)")
            ax.set_xlabel("Tiempo estimado por OrcaSlicer (min)"); ax.set_ylabel("Tensión máxima predicha (MPa)")
            ax.set_title(f"Pareto exacto: tiempo, resistencia, rugosidad (color) y filamento (tamaño), {args.material}")
            ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(DIR_FIGURAS / f"milp_frontera_tiempo_{modo}.png", dpi=160)
            print(f"Figura: {DIR_FIGURAS / f'milp_frontera_tiempo_{modo}.png'}")

            # ---- 4. evaluación de la hipótesis sobre predicción (numerales 4.4.5 y 4.4.6 del programa)
            h, dd, pat, vel = args.referencia.split(",")
            ref = dict(altura_capa_mm=float(h), densidad_relleno_pct=float(dd), patron_relleno=pat, velocidad_impresion_mms=float(vel),
                       tipo_impresora=args.impresora or "closed", material=args.material)
            pr = modelo.predict(pd.DataFrame([ref]))[0]
            ref.update({f"pred_{n}": float(pr[i]) for i, n in enumerate(red.respuestas)})
            ref["tiempo_min"], ref["filamento_g"] = orca.get((ref["altura_capa_mm"], ref["densidad_relleno_pct"], pat, ref["velocidad_impresion_mms"]), (np.nan, np.nan))
            mae = pd.read_csv(VALIDACION) if VALIDACION.exists() else None
            mae_t = float(mae[(mae.ambito == "todas") & (mae.respuesta == "tension_maxima_mpa")].mae.iloc[0]) if mae is not None else 0.0
            mae_r = float(mae[(mae.ambito == "todas") & (mae.respuesta == "rugosidad_ra_um")].mae.iloc[0]) if mae is not None else 0.0
            tol_t = max(0.05 * ref["pred_tension_maxima_mpa"], mae_t)
            tol_r = max(0.05 * ref["pred_rugosidad_ra_um"], mae_r)
            cand = nd_df.copy()
            cand["mantiene_resistencia"] = cand.pred_tension_maxima_mpa >= ref["pred_tension_maxima_mpa"] - tol_t
            cand["mantiene_rugosidad"] = cand.pred_rugosidad_ra_um <= ref["pred_rugosidad_ra_um"] + tol_r
            cand["mejora_tiempo_pct"] = 100 * (1 - cand.tiempo_min / ref["tiempo_min"])
            cand["mejora_material_pct"] = 100 * (1 - cand.filamento_g / ref["filamento_g"])
            cand["cumple_hipotesis"] = cand.mantiene_resistencia & cand.mantiene_rugosidad & ((cand.mejora_tiempo_pct >= 20) | (cand.mejora_material_pct >= 10))
            cand.to_csv(DIR_TABLAS / f"milp_hipotesis_{modo}.csv", index=False)
            print(f"\nHIPÓTESIS sobre predicción. Referencia proyectada ({args.referencia}): tensión {ref['pred_tension_maxima_mpa']:.1f} MPa, "
                  f"Ra {ref['pred_rugosidad_ra_um']:.1f} µm, {ref['tiempo_min']:.1f} min, {ref['filamento_g']:.2f} g. "
                  f"Tolerancias: −{tol_t:.1f} MPa, +{tol_r:.1f} µm. Criterio: tiempo −20 % o material −10 %.")
            ok = cand[cand.cumple_hipotesis].sort_values("mejora_tiempo_pct", ascending=False)
            print(f"  configuraciones no dominadas que cumplen: {len(ok)} de {len(cand)}")
            if len(ok):
                print(ok[red.numericas + ["patron_relleno", "pred_tension_maxima_mpa", "pred_rugosidad_ra_um", "tiempo_min", "filamento_g", "mejora_tiempo_pct", "mejora_material_pct", "soporte_impresora_patron"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
