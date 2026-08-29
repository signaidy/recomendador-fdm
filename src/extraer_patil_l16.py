"""Extrae el conjunto Patil et al. (Mendeley DOI 10.17632/mf76r4zxg4.2) a CSV ordenado.

Fuente cruda: proyecto/datos/crudos/patil_pla_fff_L16_process_parameters.xlsx
Licencia: CC BY 4.0. Diseño Taguchi L16, 16 probetas de PLA en FFF.

Se extraen únicamente las columnas de media ya calculadas por los autores, junto
con los factores del diseño. No se recalculan las medias a partir de las
lecturas individuales: se conserva el dato tal como lo publicaron los autores y
se registra la procedencia.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ROOT = Path(__file__).resolve().parents[2]
CRUDO = ROOT / "proyecto" / "datos" / "crudos" / "patil_pla_fff_L16_process_parameters.xlsx"
SALIDA = ROOT / "proyecto" / "datos" / "procesados" / "patil_l16_pla.csv"


def col_letra(ref: str) -> str:
    return "".join(c for c in ref if c.isalpha())


def leer_hoja(zf: zipfile.ZipFile, indice: int, compartidas: list[str]) -> dict[int, dict[str, str]]:
    """Devuelve {numero_de_fila: {letra_de_columna: valor}}."""
    root = ET.fromstring(zf.read(f"xl/worksheets/sheet{indice}.xml"))
    filas: dict[int, dict[str, str]] = {}
    for fila in root.iter(NS + "row"):
        numero = int(fila.get("r"))
        celdas: dict[str, str] = {}
        for celda in fila.iter(NS + "c"):
            valor = celda.find(NS + "v")
            if valor is None or valor.text is None:
                continue
            texto = (
                compartidas[int(valor.text)]
                if celda.get("t") == "s"
                else valor.text
            )
            celdas[col_letra(celda.get("r"))] = texto
        filas[numero] = celdas
    return filas


def numero(valor: str | None) -> float | None:
    if valor is None or valor == "":
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def main() -> None:
    zf = zipfile.ZipFile(CRUDO)

    compartidas: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(NS + "si"):
            compartidas.append("".join(t.text or "" for t in si.iter(NS + "t")))

    parametros = leer_hoja(zf, 1, compartidas)   # incluye tiempo de construcción
    rugosidad = leer_hoja(zf, 2, compartidas)
    dureza = leer_hoja(zf, 3, compartidas)
    dimensional = leer_hoja(zf, 4, compartidas)

    registros = []
    for i in range(1, 17):
        # Hoja 1: el arreglo L16 arranca en la fila 27; hojas 2-4 en la fila 5.
        p = parametros.get(26 + i, {})
        r = rugosidad.get(4 + i, {})
        h = dureza.get(4 + i, {})
        d = dimensional.get(4 + i, {})

        registros.append({
            "experimento": i,
            "altura_capa_mm": numero(p.get("B")),
            "angulo_raster_grados": numero(p.get("C")),
            "velocidad_impresion_mms": numero(p.get("D")),
            "densidad_relleno_pct": numero(p.get("E")),
            "tiempo_construccion_min": numero(p.get("F")),
            # Hoja 2: columna O = media de 9 lecturas de Ra en la cara superior.
            "rugosidad_ra_um_cara_superior": numero(r.get("O")),
            "rugosidad_ra_sd_cara_superior": numero(r.get("P")),
            # Hoja 3: columna K = media de 5 lecturas de dureza en cara superior.
            "dureza_shore_d_cara_superior": numero(h.get("K")),
            "dureza_shore_d_sd_cara_superior": numero(h.get("L")),
            # Hoja 4: J = media medida del lado del cubo, K = error medio.
            "lado_cubo_medido_mm": numero(d.get("J")),
            "error_lado_cubo_mm": numero(d.get("K")),
            "diagonal_medida_mm": numero(d.get("N")),
            "error_diagonal_mm": numero(d.get("O")),
        })

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(registros[0].keys()))
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"Escrito: {SALIDA}")
    print(f"Registros: {len(registros)}")
    faltantes = {
        k: sum(1 for x in registros if x[k] is None) for k in registros[0]
    }
    faltantes = {k: v for k, v in faltantes.items() if v}
    print(f"Valores faltantes: {faltantes or 'ninguno'}")


if __name__ == "__main__":
    main()
