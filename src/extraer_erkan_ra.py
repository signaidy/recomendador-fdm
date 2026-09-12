"""Extrae el conjunto Erkan, Sürer y Yaman (Zenodo DOI 10.5281/zenodo.18827518) a CSV ordenado.

Fuente cruda: proyecto/datos/crudos/erkan_fff_pla_surface_roughness.csv
Licencia: CC BY 4.0. Diseño Box-Behnken de siete factores a tres niveles, PLA 1.75 mm,
Bambu Lab A1 Combo, boquilla 0.4 mm, rugosímetro de palpador. 87 probetas, cada una medida
en 18 inclinaciones de superficie (0° a 170°), 1,566 mediciones de Ra.

Solo biblioteca estándar. No se altera ningún valor: se renombran columnas al español, se
conserva el identificador de probeta como variable de agrupación y se añade la bandera
`cara_principal` para las inclinaciones 0° y 90° (cara superior y pared vertical), que son
las comparables con las caras medidas por Patil et al.

Advertencia de uso: las 18 filas de una misma probeta comparten los siete parámetros de
proceso. En validación cruzada deben agruparse por `probeta`; de lo contrario se produce
fuga de información y la destreza aparente se infla.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRUDO = ROOT / "proyecto" / "datos" / "crudos" / "erkan_fff_pla_surface_roughness.csv"
SALIDA = ROOT / "proyecto" / "datos" / "procesados" / "erkan_ra_pla_box_behnken.csv"

COLUMNAS = {
    "ID": "probeta",
    "Layer Height (mm)": "altura_capa_mm",
    "Extrusion Temp (°C)": "temperatura_extrusion_c",
    "Outer Wall Speed (mm/s)": "velocidad_pared_exterior_mms",
    "Infill Density (%)": "densidad_relleno_pct",
    "Wall Thickness (mm)": "espesor_pared_mm",
    "Bed Temp (°C)": "temperatura_cama_c",
    "Fan Speed (%)": "ventilador_pct",
    "Surface Angle (°)": "inclinacion_superficie_grados",
    "Ra": "rugosidad_ra_um",
}


def main() -> None:
    with CRUDO.open(encoding="utf-8-sig", newline="") as fh:
        lector = csv.DictReader(fh)
        faltan = set(COLUMNAS) - set(lector.fieldnames or [])
        if faltan:
            raise SystemExit(f"Encabezados no encontrados en el crudo: {faltan}")
        registros = []
        for fila in lector:
            r = {destino: fila[origen].strip() for origen, destino in COLUMNAS.items()}
            for k in list(r):
                if k != "probeta":
                    r[k] = float(r[k])
            r["cara_principal"] = int(r["inclinacion_superficie_grados"] in (0.0, 90.0))
            registros.append(r)

    orden = list(COLUMNAS.values()) + ["cara_principal"]
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=orden)
        escritor.writeheader()
        escritor.writerows(registros)

    probetas = {r["probeta"] for r in registros}
    print(f"Escrito: {SALIDA}")
    print(f"Mediciones: {len(registros)} | Probetas: {len(probetas)}")
    print(f"Filas de caras principales (0° y 90°): {sum(r['cara_principal'] for r in registros)}")
    ra = [r["rugosidad_ra_um"] for r in registros]
    print(f"Ra: min {min(ra):.3f} | max {max(ra):.3f} µm")


if __name__ == "__main__":
    main()
