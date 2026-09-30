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
    "densidad_relleno_pct": [20.0, 50.0, 80.0],
    "velocidad_impresion_mms": [30.0, 50.0, 70.0],
}
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
    pred = modelo.predict(df)
    for i, nombre in enumerate(red.respuestas):
        df[f"pred_{nombre}"] = pred[:, i]
    df["tiempo_relativo"] = [tiempo_relativo(a, vel, d) for a, vel, d in zip(df.altura_capa_mm, df.velocidad_impresion_mms, df.densidad_relleno_pct)]
    con_tabla = orca is not None and not args.sin_tiempo
    if con_tabla:
        tm = [orca.get((a, d, p, vel), (np.nan, np.nan)) for a, d, p, vel in zip(df.altura_capa_mm, df.densidad_relleno_pct, df.patron_relleno, df.velocidad_impresion_mms)]
        df["tiempo_min"] = [t for t, _ in tm]
        df["filamento_g"] = [m for _, m in tm]
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
    return dict(soporte_material=len(d), soporte_impresora=len(mismo_imp),
                soporte_impresora_patron=len(mismo_pat), soporte_exacto=len(exacto),
                tension_observada_exacta=float(exacto.tension_maxima_mpa.mean()) if len(exacto) else float("nan"))


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
    p.add_argument("--continuo", action="store_true", help="numéricas continuas dentro del rango de los datos")
    p.add_argument("--sin-tiempo", action="store_true", help="ignora la tabla del laminador (comportamiento anterior al 29-09-2026)")
    p.add_argument("--sin-barrido", action="store_true")
    args = p.parse_args()
    if args.impresora == "libre":
        args.impresora = None

    red = RedExportada(DIR_MODELOS / "sustituto_mlp.json")
    modelo: Sustituto = joblib.load(DIR_MODELOS / "sustituto_mlp.joblib")
    datos = pd.read_csv(DATOS)
    orca = cargar_orca()
    if orca is None:
        print("AVISO: no hay tabla del laminador; tiempo y material se omiten (proxy tiempo_relativo).")
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    modo = ("continuo" if args.continuo else "discreto") + (f"_{args.impresora}" if args.impresora else "")

    # ---- 1. óptimo puntual
    sol = resolver(red, args, orca)
    if sol is None:
        raise SystemExit("El MILP es infactible con esas restricciones.")
    sol.update(soporte(sol, datos))
    imprimir(sol, f"ÓPTIMO MILP (modo {modo.split(chr(95))[0]}, HiGHS, {sol['segundos']*1000:.0f} ms, {sol['binarias_relu']} binarias ReLU)")

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
            barrido = [(ra, dmax) for ra in ra_caps for dmax in NIVELES["densidad_relleno_pct"]]
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
