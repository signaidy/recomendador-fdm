"""Calcula la dispersión de las variables respuesta y la línea base sin destreza.

El propósito es fijar umbrales de error predictivo que signifiquen algo. Un
umbral de MAE solo tiene sentido comparado contra el MAE que obtiene un
predictor constante —la media o la mediana de la muestra—, porque ese es el
error que se consigue sin modelar nada. Un umbral más laxo que esa línea base
se satisface prediciendo siempre el mismo número.

También estima el piso de ruido de medición cuando la fuente publica la
desviación estándar de lecturas repetidas sobre la misma probeta. Ninguna
predicción puede ser sistemáticamente mejor que la repetibilidad del
instrumento.
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATIL = ROOT / "proyecto" / "datos" / "procesados" / "patil_l16_pla.csv"
OKUDAN = ROOT / "proyecto" / "datos" / "crudos" / "okudan_3dprinter_manufacturing.csv"


def resumen(nombre: str, valores: list[float], unidad: str) -> dict:
    n = len(valores)
    media = st.mean(valores)
    mediana = st.median(valores)
    mae_media = sum(abs(v - media) for v in valores) / n
    mae_mediana = sum(abs(v - mediana) for v in valores) / n
    # El predictor constante óptimo bajo MAE es la mediana, no la media.
    linea_base = min(mae_media, mae_mediana)
    return {
        "variable": nombre,
        "unidad": unidad,
        "n": n,
        "min": min(valores),
        "max": max(valores),
        "rango": max(valores) - min(valores),
        "media": media,
        "sd": st.stdev(valores) if n > 1 else 0.0,
        "mae_base": linea_base,
        "mae_base_pct_rango": 100 * linea_base / (max(valores) - min(valores)),
    }


def imprimir(filas: list[dict]) -> None:
    cab = f"{'variable':38} {'unid':6} {'n':>3} {'min':>9} {'max':>9} {'rango':>9} {'sd':>8} {'MAEbase':>9} {'%rango':>7}"
    print(cab)
    print("-" * len(cab))
    for f in filas:
        print(
            f"{f['variable']:38} {f['unidad']:6} {f['n']:3d} "
            f"{f['min']:9.3f} {f['max']:9.3f} {f['rango']:9.3f} "
            f"{f['sd']:8.3f} {f['mae_base']:9.4f} {f['mae_base_pct_rango']:6.1f}%"
        )


def main() -> None:
    print("=" * 104)
    print("PATIL ET AL. — Mendeley 10.17632/mf76r4zxg4.2 — PLA, FFF, Taguchi L16, CC BY 4.0")
    print("=" * 104)
    filas = list(csv.DictReader(PATIL.open(encoding="utf-8")))
    col = lambda c: [float(r[c]) for r in filas]

    resultados = [
        resumen("rugosidad Ra cara superior", col("rugosidad_ra_um_cara_superior"), "um"),
        resumen("error lado del cubo", col("error_lado_cubo_mm"), "mm"),
        resumen("error diagonal de cara", col("error_diagonal_mm"), "mm"),
        resumen("dureza Shore D cara superior", col("dureza_shore_d_cara_superior"), "HD"),
        resumen("tiempo de construccion", col("tiempo_construccion_min"), "min"),
    ]
    imprimir(resultados)

    print()
    print("PISO DE RUIDO DE MEDICION (sd de lecturas repetidas sobre la misma probeta)")
    ra_sd = col("rugosidad_ra_sd_cara_superior")
    ra_media = col("rugosidad_ra_um_cara_superior")
    dur_sd = col("dureza_shore_d_sd_cara_superior")
    dur_media = col("dureza_shore_d_cara_superior")
    print(f"  Ra: sd intra-probeta media = {st.mean(ra_sd):.3f} um "
          f"| sd entre probetas = {st.stdev(ra_media):.3f} um "
          f"| razon ruido/senal = {st.mean(ra_sd)/st.stdev(ra_media):.2f}")
    print(f"  Dureza: sd intra-probeta media = {st.mean(dur_sd):.3f} HD "
          f"| sd entre probetas = {st.stdev(dur_media):.3f} HD "
          f"| razon ruido/senal = {st.mean(dur_sd)/st.stdev(dur_media):.2f}")

    print()
    print("=" * 104)
    print("OKUDAN — Kaggle/apmonitor — subconjunto PLA")
    print("=" * 104)
    crudo = list(csv.DictReader(OKUDAN.open(encoding="utf-8")))
    pla = [r for r in crudo if r["material"] == "pla"]
    # Se excluye la fila fisicamente imposible antes de cualquier estadistica.
    limpio = [
        r for r in pla
        if float(r["roughness"]) >= 0
        and float(r["bed_temperature"]) <= 120
        and float(r["print_speed"]) <= 200
    ]
    print(f"  PLA en bruto: {len(pla)} filas | tras excluir filas imposibles: {len(limpio)}")

    resultados = [
        resumen("resistencia a traccion (PLA, bruto)",
                [float(r["tension_strength"]) for r in pla], "MPa"),
        resumen("resistencia a traccion (PLA, limpio)",
                [float(r["tension_strength"]) for r in limpio], "MPa"),
        resumen("rugosidad declarada (PLA, limpio)",
                [float(r["roughness"]) for r in limpio], "um"),
        resumen("elongacion (PLA, limpio)",
                [float(r["elongation"]) for r in limpio], "%"),
    ]
    imprimir(resultados)

    print()
    print("LECTURA: el umbral de MAE debe quedar por DEBAJO de MAEbase para que el modelo")
    print("demuestre destreza. Se recomienda expresarlo como reduccion relativa a esa linea")
    print("base y no como valor absoluto, porque el valor absoluto depende de la muestra.")


if __name__ == "__main__":
    main()
