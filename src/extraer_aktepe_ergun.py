"""Extrae el conjunto Aktepe y Ergün (Mendeley DOI 10.17632/zd6td6svd6.2) a CSV ordenado.

Fuente cruda: proyecto/datos/crudos/aktepe_ergun_fdm_500_process_mechanical.xlsx
Licencia: CC BY 4.0. 500 combinaciones de parámetros, cada fila es la media de tres
probetas ISO 527-2 tipo 4 según el artículo asociado (Micromachines 17(7), 859, 2026).

Solo biblioteca estándar, igual que extraer_patil_l16.py, para que la extracción sea
reproducible en cualquier instalación de Python.

Transformaciones aplicadas, todas registradas aquí y ninguna en el archivo crudo:
1. Se recortan espacios y se unifica a minúsculas el tipo de impresora («closed » → «closed»).
2. Se normaliza la ortografía del patrón de relleno, que en el crudo mezcla mayúsculas,
   minúsculas y espacios («Zigzag», «zig zag»; «Triangles», «triangles»; «tri hexagon»).
3. La rugosidad igual a 0 se trata como valor centinela y se deja vacía, con la bandera
   `rugosidad_centinela` = 1. Una Ra de 0 µm no es físicamente plausible en FDM.
4. Se añade `tension_maxima_mpa` = tensión en kPa / 1000, para comparar con las demás
   fuentes, que reportan MPa. La columna original en kPa se conserva.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ROOT = Path(__file__).resolve().parents[2]
CRUDO = ROOT / "proyecto" / "datos" / "crudos" / "aktepe_ergun_fdm_500_process_mechanical.xlsx"
SALIDA = ROOT / "proyecto" / "datos" / "procesados" / "aktepe_ergun_fdm_500.csv"

# Ortografías observadas en el crudo → forma canónica.
PATRONES = {
    "grid": "grid",
    "triangles": "triangles",
    "zigzag": "zigzag",
    "zig zag": "zigzag",
    "lines": "lines",
    "cubic": "cubic",
    "quarter cubic": "quarter_cubic",
    "cross": "cross",
    "gyroid": "gyroid",
    "cubic subdivision": "cubic_subdivision",
    "tri hexagon": "tri_hexagon",
    "tri-hexagon": "tri_hexagon",
    "lightning": "lightning",
    "cross3d": "cross3d",
    "concentric": "concentric",
    "octet": "octet",
}

# Encabezado del crudo → nombre de columna de salida.
COLUMNAS = {
    "Sample No": "muestra",
    "Printer Type": "tipo_impresora",
    "Material": "material",
    "Layer Thickness (mm)": "altura_capa_mm",
    "Infill Density (%)": "densidad_relleno_pct",
    "Infill Pattern": "patron_relleno",
    "Speed (mm/s)": "velocidad_impresion_mms",
    "Roughness AVG": "rugosidad_ra_um",
    "Hardness AVG": "dureza_shore",
    "Peak Load (N)": "carga_maxima_n",
    "Peak Stress (kPa)": "tension_maxima_kpa",
    "Strain at Break (mm/mm)": "deformacion_rotura",
    "Modulus (MPa)": "modulo_mpa",
    "Width (mm)": "ancho_probeta_mm",
    "Thickness (mm)": "espesor_probeta_mm",
}


def col_letra(ref: str) -> str:
    return "".join(c for c in ref if c.isalpha())


def leer_hoja(zf: zipfile.ZipFile, compartidas: list[str]) -> list[dict[str, str]]:
    root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    filas: list[dict[str, str]] = []
    for fila in root.iter(NS + "row"):
        celdas: dict[str, str] = {}
        for celda in fila.iter(NS + "c"):
            valor = celda.find(NS + "v")
            if valor is None or valor.text is None:
                continue
            texto = compartidas[int(valor.text)] if celda.get("t") == "s" else valor.text
            celdas[col_letra(celda.get("r"))] = texto
        filas.append(celdas)
    return filas


def limpiar_encabezado(texto: str) -> str:
    # El encabezado de velocidad contiene espacios de ancho cero (U+200B) en el crudo.
    return " ".join(texto.replace("​", "").split())


def numero(valor: str | None) -> float | None:
    if valor is None or valor == "":
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def main() -> None:
    zf = zipfile.ZipFile(CRUDO)
    compartidas = [
        "".join(t.text or "" for t in si.iter(NS + "t"))
        for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(NS + "si")
    ]
    filas = leer_hoja(zf, compartidas)

    encabezado = {letra: limpiar_encabezado(v) for letra, v in filas[0].items()}
    faltan = set(COLUMNAS) - set(encabezado.values())
    if faltan:
        raise SystemExit(f"Encabezados no encontrados en el crudo: {faltan}")

    registros = []
    patrones_desconocidos: set[str] = set()
    centinelas = 0
    for fila in filas[1:]:
        crudo = {encabezado[letra]: v for letra, v in fila.items() if letra in encabezado}
        if not crudo.get("Sample No"):
            continue
        r: dict[str, object] = {}
        for origen, destino in COLUMNAS.items():
            v = crudo.get(origen)
            if destino in ("tipo_impresora", "material"):
                r[destino] = (v or "").strip()
                if destino == "tipo_impresora":
                    r[destino] = r[destino].lower()
            elif destino == "patron_relleno":
                clave = " ".join((v or "").strip().lower().split())
                if clave not in PATRONES:
                    patrones_desconocidos.add(clave)
                r[destino] = PATRONES.get(clave, clave)
            else:
                r[destino] = numero(v)
        r["muestra"] = int(r["muestra"]) if r["muestra"] is not None else None

        # Centinela de rugosidad.
        r["rugosidad_centinela"] = 0
        if r["rugosidad_ra_um"] is not None and r["rugosidad_ra_um"] <= 0:
            r["rugosidad_ra_um"] = None
            r["rugosidad_centinela"] = 1
            centinelas += 1

        r["tension_maxima_mpa"] = (
            r["tension_maxima_kpa"] / 1000.0 if r["tension_maxima_kpa"] is not None else None
        )
        registros.append(r)

    if patrones_desconocidos:
        raise SystemExit(f"Patrones de relleno sin forma canónica: {patrones_desconocidos}")

    orden = list(COLUMNAS.values()) + ["rugosidad_centinela", "tension_maxima_mpa"]
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=orden)
        escritor.writeheader()
        for r in registros:
            escritor.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in orden})

    print(f"Escrito: {SALIDA}")
    print(f"Registros: {len(registros)}")
    materiales: dict[str, int] = {}
    for r in registros:
        materiales[r["material"]] = materiales.get(r["material"], 0) + 1
    print("Por material:", dict(sorted(materiales.items(), key=lambda kv: -kv[1])))
    print(f"Rugosidad centinela (0 → vacío): {centinelas}")
    faltantes = {k: sum(1 for r in registros if r.get(k) is None) for k in orden}
    print("Valores faltantes:", {k: v for k, v in faltantes.items() if v} or "ninguno")


if __name__ == "__main__":
    main()
