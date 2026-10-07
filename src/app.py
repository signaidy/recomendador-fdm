"""Prototipo de interfaz: recomendación de parámetros FDM para PLA+ con optimización exacta y explicación.

    .venv/Scripts/python.exe -m streamlit run proyecto/src/app.py

La interfaz no reentrena nada: carga el sustituto exportado, la tabla del laminador y el conjunto
de Pareto exacto, resuelve el programa lineal entero mixto con las restricciones y pesos que el
usuario introduce, y explica la recomendación con valores de Shapley. Toda predicción se muestra
con su error validado y con su respaldo en datos; una combinación sin respaldo se marca como
extrapolación. El prototipo apoya decisiones experimentales y no certifica seguridad estructural.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizar_milp as om  # noqa: E402
from entrenar_sustituto import RESPUESTAS, UNIDADES, Sustituto  # noqa: E402
from explicar import NOMBRES, Explicador, explicar_local  # noqa: E402

ETIQUETA = {"tension_maxima_mpa": "Tensión máxima", "rugosidad_ra_um": "Rugosidad Ra", "dureza_shore": "Dureza Shore D"}
REFERENCIA = dict(altura_capa_mm=0.2, densidad_relleno_pct=20.0, velocidad_impresion_mms=70.0, patron_relleno="grid", tipo_impresora="closed", material="PLA+")


@st.cache_resource(show_spinner="Cargando modelo, tabla del laminador y frontera...")
def recursos():
    import joblib
    red = om.RedExportada(om.DIR_MODELOS / "sustituto_mlp.json")
    modelo: Sustituto = joblib.load(om.DIR_MODELOS / "sustituto_mlp.joblib")
    datos = pd.read_csv(om.DATOS)
    orca = om.cargar_orca()
    val = pd.read_csv(om.VALIDACION)
    mae = {r: float(val[(val.ambito == "todas") & (val.respuesta == r)].mae.iloc[0]) for r in RESPUESTAS}
    fondo = datos.dropna(subset=RESPUESTAS)
    fondo = fondo[(fondo.material == "PLA+") & (fondo.tipo_impresora == "closed")]
    expl = Explicador(modelo, fondo)
    pareto = {}
    for ext, sufijo in ((False, ""), (True, "_extrapolar")):
        ruta = om.DIR_TABLAS / f"milp_pareto_exacto_discreto_closed{sufijo}.csv"
        pareto[ext] = pd.read_csv(ruta) if ruta.exists() else None
    version = hashlib.sha256((om.DIR_MODELOS / "sustituto_mlp.json").read_bytes()).hexdigest()[:12]
    rangos = {}
    for ext in (False, True):
        rangos[ext] = om.rangos_dominio(om.enumerar(modelo, red, args_base(extrapolar=ext), orca))
    return red, modelo, datos, orca, mae, expl, pareto, version, rangos


def args_base(**kw) -> argparse.Namespace:
    a = dict(material="PLA+", impresora="closed", patron=None, ra_max=None, densidad_max=None, altura_min=None, velocidad_min=None,
             dureza_min=None, tiempo_max=None, material_max=None, potencia_w=None, peso_ra=0.0, peso_densidad=0.0, peso_tiempo=0.0,
             peso_material=0.0, continuo=False, sin_tiempo=False, extrapolar=False, pesos=None, rangos=None)
    a.update(kw)
    return argparse.Namespace(**a)


def predecir(modelo: Sustituto, orca: dict, cfg: dict, potencia: float | None) -> dict:
    pred = modelo.predict(pd.DataFrame([cfg]))[0]
    out = {f"pred_{r}": float(pred[i]) for i, r in enumerate(RESPUESTAS)}
    k = (cfg["altura_capa_mm"], cfg["densidad_relleno_pct"], cfg["patron_relleno"], cfg["velocidad_impresion_mms"])
    out["tiempo_min"], out["filamento_g"] = orca.get(k, (np.nan, np.nan)) if orca else (np.nan, np.nan)
    if potencia and np.isfinite(out["tiempo_min"]):
        out["energia_wh"] = potencia * out["tiempo_min"] / 60
    return out


def fila_resultado(nombre: str, cfg: dict, pred: dict, mae: dict) -> dict:
    fila = {"Configuración": nombre, "Capa (mm)": cfg["altura_capa_mm"], "Relleno (%)": cfg["densidad_relleno_pct"],
            "Velocidad (mm/s)": cfg["velocidad_impresion_mms"], "Patrón": cfg["patron_relleno"]}
    for r in RESPUESTAS:
        fila[f"{ETIQUETA[r]} ({UNIDADES[r]})"] = f"{pred[f'pred_{r}']:.1f} ± {mae[r]:.1f}"
    fila["Tiempo (min)"] = f"{pred['tiempo_min']:.1f}" if np.isfinite(pred["tiempo_min"]) else "sin laminado"
    fila["Filamento (g)"] = f"{pred['filamento_g']:.2f}" if np.isfinite(pred["filamento_g"]) else "sin laminado"
    if "energia_wh" in pred:
        fila["Energía (Wh)"] = f"{pred['energia_wh']:.1f}"
    return fila


def main() -> None:
    st.set_page_config(page_title="Recomendador FDM PLA+", layout="wide")
    red, modelo, datos, orca, mae, expl, pareto_todos, version, rangos_todos = recursos()
    patrones = red.categorias["patron_relleno"]
    con_laminado = sorted({k[2] for k in orca}) if orca else []

    st.title("Recomendador de parámetros de impresión FDM para PLA+")
    st.caption(f"Sustituto neuronal ReLU + optimización entera mixta exacta (HiGHS) · modelo {version} · "
               f"dominio: Creality K1/K1C (cámara cerrada), eSUN PLA+, probeta ASTM D638 tipo IV. "
               "Apoyo a decisiones experimentales; no certifica seguridad estructural.")

    # ------------------------------------------------------------------ barra lateral
    sb = st.sidebar
    sb.header("1. Restricciones")
    ra_max = sb.slider("Rugosidad Ra máxima (µm)", 1.0, 10.0, 10.0, 0.5)
    t_lim = int(np.ceil(max(t for t, _ in orca.values()))) if orca else 61
    t_min = int(np.floor(min(t for t, _ in orca.values()))) if orca else 12
    m_lim = float(np.ceil(max(m for _, m in orca.values()) * 10) / 10) if orca else 6.5
    m_min = float(np.floor(min(m for _, m in orca.values()) * 10) / 10) if orca else 3.0
    tiempo_max = sb.slider("Tiempo máximo (min)", t_min, t_lim, t_lim, 1)
    material_max = sb.slider("Filamento máximo (g)", m_min, m_lim, m_lim, 0.1)
    extrapolar = sb.checkbox("Permitir densidades fuera del rango de entrenamiento (10 %, 90 % y 100 %; el modelo extrapola)", value=True)
    lo_d, hi_d = om.RANGO_DATOS["densidad_relleno_pct"]
    niveles_d = [int(d) for d in om.NIVELES["densidad_relleno_pct"] if extrapolar or lo_d <= d <= hi_d]
    densidad_max = sb.select_slider("Densidad de relleno máxima (%)", niveles_d, niveles_d[-1])
    altura_min = sb.select_slider("Altura de capa mínima (mm)", [0.1, 0.2, 0.3], 0.1)
    patron = sb.selectbox("Patrón de relleno", ["libre"] + [p for p in patrones if p in con_laminado])
    sb.header("2. Prioridades")
    perfil = sb.selectbox("Perfil de ejemplo", list(om.PERFILES), index=1)
    base_w = om.PERFILES[perfil]
    sb.caption("Los pesos se normalizan para sumar 1. La energía se estima como potencia por tiempo, así que comparte el peso del tiempo.")
    pesos = {}
    for k, etiqueta in (("resistencia", "Resistencia"), ("calidad", "Calidad (rugosidad baja)"), ("material", "Filamento"), ("tiempo", "Tiempo y energía")):
        pesos[k] = sb.slider(etiqueta, 0, 10, int(round(10 * base_w[k])), 1, key=f"w_{k}_{perfil}")
    pesos_n = om.normalizar_pesos(pesos)
    normalizacion = sb.radio("Normalización de los objetivos", ["Rango del dominio", "Respecto de la referencia"], index=0,
                             help="Rango del dominio: cada objetivo se escala entre su mínimo y su máximo en todas las combinaciones laminables. "
                                  "Respecto de la referencia: cada objetivo se divide por su valor en la configuración de referencia, de modo que S mide cambios proporcionales.")
    sb.caption("Pesos efectivos: " + ", ".join(f"{k} {v:.2f}" for k, v in pesos_n.items()))
    sb.header("3. Energía")
    potencia = sb.number_input("Potencia media de la impresora (W), 0 = no estimar", 0.0, 500.0, 0.0, 10.0)
    potencia = potencia or None
    sb.header("4. Versión")
    sb.write(f"Modelo `{version}`; MAE validado: " + ", ".join(f"{ETIQUETA[r]} {mae[r]:.2f} {UNIDADES[r]}" for r in RESPUESTAS))

    a = args_base(ra_max=ra_max, tiempo_max=float(tiempo_max), material_max=float(material_max), densidad_max=float(densidad_max),
                  altura_min=float(altura_min), patron=None if patron == "libre" else patron,
                  potencia_w=potencia, extrapolar=extrapolar, pesos=pesos_n, rangos=rangos_todos[extrapolar])
    rangos = rangos_todos[extrapolar]
    if normalizacion == "Respecto de la referencia":
        rangos = om.rangos_referencia(predecir(modelo, orca, REFERENCIA, potencia))
        a.rangos = rangos
    pareto = pareto_todos[extrapolar]

    # ------------------------------------------------------------------ recomendación
    st.header("Recomendación")
    sol = om.resolver(red, a, orca)
    if sol is None:
        st.error("No existe ninguna combinación laminable que cumpla todas las restricciones. Relaje alguna cota.")
        st.stop()
    sol.update(om.soporte(sol, datos))
    ref = predecir(modelo, orca, REFERENCIA, potencia)
    S_sol = float(om.puntuar(pd.DataFrame([sol]), pesos_n, rangos).iloc[0])
    S_ref = float(om.puntuar(pd.DataFrame([ref]), pesos_n, rangos).iloc[0])
    filas = [dict(fila_resultado("Recomendada (mayor puntuación S)", sol, sol, mae), **{"S": f"{S_sol:.3f}"}),
             dict(fila_resultado("Referencia proyectada", REFERENCIA, ref, mae), **{"S": f"{S_ref:.3f}"})]
    rod = om.rodilla(pareto, rangos_todos[extrapolar]) if pareto is not None else None  # la rodilla no depende de la normalización elegida
    if rod is not None:
        cfg_rod = {k: rod[k] for k in ["altura_capa_mm", "densidad_relleno_pct", "velocidad_impresion_mms", "patron_relleno"]} | {"tipo_impresora": "closed", "material": "PLA+"}
        pr_rod = predecir(modelo, orca, cfg_rod, potencia)
        filas.insert(1, dict(fila_resultado("Solución de compromiso (rodilla)", cfg_rod, pr_rod, mae), **{"S": f"{float(om.puntuar(pd.DataFrame([pr_rod]), pesos_n, rangos).iloc[0]):.3f}"}))
    tabla = pd.DataFrame(filas)
    st.dataframe(tabla.set_index("Configuración"), width="stretch")
    st.write(f"**S** es la puntuación ponderada de respuestas normalizadas {'al rango del dominio' if normalizacion == 'Rango del dominio' else 'respecto de la referencia (lo = 0, hi = valor de la referencia)'} (resistencia y calidad suman, filamento y tiempo restan; "
             f"rangos: resistencia {rangos['resistencia'][0]:.1f}–{rangos['resistencia'][1]:.1f} MPa, Ra {rangos['calidad'][0]:.1f}–{rangos['calidad'][1]:.1f} µm, "
             f"filamento {rangos['material'][0]:.2f}–{rangos['material'][1]:.2f} g, tiempo {rangos['tiempo'][0]:.0f}–{rangos['tiempo'][1]:.0f} min). "
             "La recomendación maximiza S de forma exacta con las cotas dadas; la solución de compromiso es el punto de la frontera donde una mejora "
             "adicional de resistencia o calidad empieza a exigir un aumento desproporcionado de filamento y tiempo, y no depende de los pesos.")
    ig, tot = om.estabilidad(red, a, orca, sol, pesos_n)
    if tot:
        st.write(f"Estabilidad: la misma recomendación gana en {ig} de {tot} perturbaciones de ±10 % de los pesos" + (
            "; la recomendación es robusta." if ig == tot else "; con pesos cercanos cambia la recomendación, conviene revisar las alternativas."))
    if sol.get("fuera_rango_entrenamiento"):
        st.warning(f"EXTRAPOLACIÓN: {sol['fuera_rango_entrenamiento'].replace('densidad_relleno_pct', 'la densidad de relleno')} está fuera del rango de entrenamiento "
                   "(relleno de 20 a 80 %). La predicción es una extensión lineal del modelo, no una interpolación entre datos; el tiempo y el filamento sí son del laminador. Valide antes de usar.")
    if sol["soporte_impresora_patron"] == 0:
        st.error("EXTRAPOLACIÓN: esta combinación de impresora y patrón no tiene datos de entrenamiento. No use la recomendación sin validarla.")
    else:
        st.info(f"Respaldo en datos: {sol['soporte_impresora_patron']} combinaciones con esta impresora y patrón; "
                f"{sol['soporte_exacto']} idénticas" + (f" (observada {sol['tension_observada_exacta']:.1f} MPa)." if sol["soporte_exacto"] else "."))

    # ------------------------------------------------------------------ comparación con la referencia
    st.subheader("Cambio frente a la referencia proyectada")
    cols = st.columns(5)
    items = [("Tensión", sol["pred_tension_maxima_mpa"], ref["pred_tension_maxima_mpa"], "MPa", True),
             ("Rugosidad", sol["pred_rugosidad_ra_um"], ref["pred_rugosidad_ra_um"], "µm", False),
             ("Dureza", sol["pred_dureza_shore"], ref["pred_dureza_shore"], "Shore D", True),
             ("Tiempo", sol.get("tiempo_min", np.nan), ref["tiempo_min"], "min", False),
             ("Filamento", sol.get("filamento_g", np.nan), ref["filamento_g"], "g", False)]
    for c, (nom, v, r, u, mayor_mejor) in zip(cols, items):
        if np.isfinite(v) and np.isfinite(r):
            d = v - r
            c.metric(f"{nom} ({u})", f"{v:.1f}", f"{d:+.1f} ({100*d/r:+.0f} %)", delta_color="normal" if mayor_mejor else "inverse")
    tol_t = max(0.05 * ref["pred_tension_maxima_mpa"], mae["tension_maxima_mpa"])
    tol_r = max(0.05 * ref["pred_rugosidad_ra_um"], mae["rugosidad_ra_um"])
    dentro = sol["pred_tension_maxima_mpa"] >= ref["pred_tension_maxima_mpa"] - tol_t and sol["pred_rugosidad_ra_um"] <= ref["pred_rugosidad_ra_um"] + tol_r
    mejora_t = 100 * (1 - sol.get("tiempo_min", np.nan) / ref["tiempo_min"])
    mejora_m = 100 * (1 - sol.get("filamento_g", np.nan) / ref["filamento_g"])
    st.write(f"Regla del programa definitivo: mantiene resistencia y rugosidad dentro de la degradación admisible "
             f"(−{tol_t:.1f} MPa, +{tol_r:.1f} µm): **{'sí' if dentro else 'no'}**. Mejora de tiempo {mejora_t:+.0f} % (criterio ≥ 20 %), "
             f"de filamento {mejora_m:+.0f} % (criterio ≥ 10 %). Las diferencias menores que el MAE están dentro del error del modelo.")

    # ------------------------------------------------------------------ explicación
    st.header("Por qué")
    st.write("Contribución de cada variable a la predicción, respecto de la combinación típica del dominio (valores de Shapley, en unidades físicas). "
             "Describe el comportamiento del modelo, no causalidad física.")
    cfg = {k: sol[k] for k in ["altura_capa_mm", "densidad_relleno_pct", "velocidad_impresion_mms", "patron_relleno", "tipo_impresora", "material"]}
    loc = explicar_local(expl, cfg)
    contrib = loc.iloc[:4]
    c1, c2, c3 = st.columns(3)
    for c, r in zip((c1, c2, c3), RESPUESTAS):
        c.write(f"**{ETIQUETA[r]}**: base {loc[r]['Valor base (media del dominio)']:.1f} → {loc[r]['Predicción']:.1f} {UNIDADES[r]}")
        c.bar_chart(contrib[[r]].rename(columns={r: UNIDADES[r]}), horizontal=True)
    principal = contrib["tension_maxima_mpa"].abs().idxmax()
    signo = "sube" if contrib["tension_maxima_mpa"][principal] > 0 else "baja"
    st.write(f"En palabras: la resistencia predicha {signo} sobre todo por **{principal.lower()}** "
             f"({contrib['tension_maxima_mpa'][principal]:+.1f} MPa); la rugosidad se mueve principalmente por "
             f"**{contrib['rugosidad_ra_um'].abs().idxmax().lower()}** ({contrib['rugosidad_ra_um'][contrib['rugosidad_ra_um'].abs().idxmax()]:+.1f} µm).")

    # ------------------------------------------------------------------ alternativas
    st.header("Alternativas no dominadas que cumplen las restricciones")
    if pareto is not None:
        alt = pareto[(pareto.pred_rugosidad_ra_um <= ra_max) & (pareto.tiempo_min <= tiempo_max) & (pareto.filamento_g <= material_max)
                     & (pareto.densidad_relleno_pct <= densidad_max) & (pareto.altura_capa_mm >= altura_min)]
        if patron != "libre":
            alt = alt[alt.patron_relleno == patron]
        st.write(f"{len(alt)} de {len(pareto)} configuraciones del conjunto de Pareto exacto (resistencia, rugosidad, tiempo, filamento) cumplen las cotas.")
        alt = alt.assign(S=om.puntuar(alt, pesos_n, rangos))
        vista = alt[["S", "altura_capa_mm", "densidad_relleno_pct", "velocidad_impresion_mms", "patron_relleno", "pred_tension_maxima_mpa",
                     "pred_rugosidad_ra_um", "pred_dureza_shore", "tiempo_min", "filamento_g", "soporte_impresora_patron"]].rename(columns={
            "altura_capa_mm": "Capa (mm)", "densidad_relleno_pct": "Relleno (%)", "velocidad_impresion_mms": "Velocidad (mm/s)", "patron_relleno": "Patrón",
            "pred_tension_maxima_mpa": "Tensión (MPa)", "pred_rugosidad_ra_um": "Ra (µm)", "pred_dureza_shore": "Dureza", "tiempo_min": "Tiempo (min)",
            "filamento_g": "Filamento (g)", "soporte_impresora_patron": "Respaldo"})
        st.dataframe(vista.sort_values("S", ascending=False).round(3), width="stretch", hide_index=True)
        st.scatter_chart(alt.rename(columns={"tiempo_min": "Tiempo (min)", "pred_tension_maxima_mpa": "Tensión (MPa)", "pred_rugosidad_ra_um": "Ra (µm)"}),
                         x="Tiempo (min)", y="Tensión (MPa)", color="Ra (µm)")

    # ------------------------------------------------------------------ comparar dos configuraciones
    st.header("Comparar dos configuraciones")
    ca, cb = st.columns(2)
    cfgs = []
    for c, nombre, ini in ((ca, "A", REFERENCIA), (cb, "B", cfg)):
        c.subheader(nombre)
        h = c.select_slider(f"Capa {nombre} (mm)", [0.1, 0.2, 0.3], ini["altura_capa_mm"])
        d = c.select_slider(f"Relleno {nombre} (%)", om.NIVELES["densidad_relleno_pct"], ini["densidad_relleno_pct"])
        v = c.select_slider(f"Velocidad {nombre} (mm/s)", [30.0, 50.0, 70.0], ini["velocidad_impresion_mms"])
        p = c.selectbox(f"Patrón {nombre}", patrones, index=patrones.index(ini["patron_relleno"]))
        cfgs.append(dict(altura_capa_mm=h, densidad_relleno_pct=d, velocidad_impresion_mms=v, patron_relleno=p, tipo_impresora="closed", material="PLA+"))
    comp = pd.DataFrame([fila_resultado(n, c_, predecir(modelo, orca, c_, potencia), mae) for n, c_ in zip(("A", "B"), cfgs)])
    st.dataframe(comp.set_index("Configuración"), width="stretch")
    for n, c_ in zip(("A", "B"), cfgs):
        if not (20 <= c_["densidad_relleno_pct"] <= 80):
            st.warning(f"{n}: relleno de {c_['densidad_relleno_pct']:.0f} % está fuera del rango de entrenamiento (20 a 80 %); la predicción es una extrapolación lineal.")
        if c_["patron_relleno"] not in con_laminado:
            st.warning(f"{n}: el patrón {c_['patron_relleno']} no tiene equivalente en OrcaSlicer; no hay tiempo ni filamento y no puede imprimirse tal cual.")

    # ------------------------------------------------------------------ dominio
    st.header("Dominio de validez")
    st.write("El modelo se entrenó con 483 combinaciones de Aktepe y Ergün (Mendeley Data, CC BY 4.0), diez filamentos y dos impresoras; "
             "las recomendaciones se restringen a PLA+ en cámara cerrada, que es el equipo disponible. Rangos de entrada: altura de capa 0.1 a 0.3 mm, "
             "relleno 20 a 80 %, velocidad 30 a 70 mm/s; temperaturas fijas de 215 a 220 °C de boquilla y 70 °C de cama. La densidad se explora en pasos de 10 %: los niveles 30, 40, 60 y 70 % son "
             "interpolación entre los datos, y 10, 90 y 100 % se admiten como extrapolación marcada; en todos ellos el tiempo y el filamento provienen del laminador, "
             "pero las tres propiedades son la respuesta de la red fuera de los puntos medidos.")
    st.write(f"Patrones con tiempo y filamento estimados por OrcaSlicer: {', '.join(con_laminado)}. "
             f"Sin equivalente en el laminador: {', '.join(p for p in patrones if p not in con_laminado)}.")
    st.image(str(om.DIR_FIGURAS / "shap_global.png"), caption="Importancia global de las variables en el sustituto (media de |SHAP|), PLA+ en cámara cerrada.")


if __name__ == "__main__" or True:
    main()
