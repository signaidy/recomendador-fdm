"""Estima tiempo de impresión y filamento con OrcaSlicer para cada combinación de parámetros.

Sustituye el proxy `tiempo_relativo` de optimizar_milp.py por la estimación del laminador
real sobre la geometría real, en la impresora real (Creality K1C, cámara cerrada, boquilla
0.4 mm). El material consumido (g y mm) sale del mismo G-code.

Flujo:
1. Genera (o recibe con --stl) la probeta de tracción ASTM D638 tipo IV como STL. Aktepe y
   Ergün la declaran «ISO 527-2 tipo 4, equivalente a ASTM D638»; el tipo IV de D638 es la
   probeta plana de 115 mm de largo, 6 mm de ancho en la zona calibrada y 3.2 mm de espesor
   (STL estanco de 4,812 mm³, es decir 5.97 g de PLA+ a 1.24 g/cm³ con relleno pleno).
2. Escribe un perfil de proceso de OrcaSlicer por combinación (altura de capa, densidad y
   patrón de relleno, velocidades) que hereda del perfil base indicado con --proceso-base.
3. Invoca la línea de comandos de OrcaSlicer, que rebana y exporta G-code.
4. Extrae del encabezado del G-code «estimated printing time» y «filament used [g] / [mm]».
5. Escribe datos/procesados/orca_k1c_tiempo_material.csv.

Probado con OrcaSlicer 2.4.2 el 29 de septiembre de 2026. Hallazgos: la opción de filamento es
--load-filaments; la CLI acepta perfiles de proceso y filamento aplanados (sin herencia) pero
rechaza copias aplanadas de máquina, así que la máquina se toma del JSON de sistema instalado.

Ejecución en paralelo: cada combinación se lamina en un proceso independiente de OrcaSlicer y se
exporta a un 3MF propio, del que se leen tiempo y filamento; por defecto corren hasta seis a la
vez. Una probeta de este tamaño se lamina en pocos segundos, así que las 270 combinaciones toman
del orden de minutos. Los 3MF existentes se reutilizan (se puede interrumpir y reanudar); con
--rehacer se descartan. --limite 3 sirve para probar la cadena antes de lanzar todo.

Uso típico (PowerShell, desde la raíz del repositorio):
    .venv\\Scripts\\python.exe proyecto\\src\\estimar_tiempo_orca.py --solo-preparar
    .venv\\Scripts\\python.exe proyecto\\src\\estimar_tiempo_orca.py --limite 3        (prueba de la cadena)
    .venv\\Scripts\\python.exe proyecto\\src\\estimar_tiempo_orca.py ^
        --filamento "$env:APPDATA\\OrcaSlicer\\user\\<id>\\filament\\eSUN PLA+ @K1C.json" --paralelo 6

Por defecto se usan las copias resueltas (herencia aplanada) de los preajustes de sistema de
OrcaSlicer 2.4.2 guardadas en proyecto/datos/crudos/orca_v2.4.2/, que sí funcionan con
--load-settings. Los perfiles de usuario exportados desde OrcaSlicer (Archivo → Exportar →
Exportar preajustes) también son JSON planos y pueden sustituirlos; en particular, el filamento
debe reemplazarse por el preajuste propio de eSUN PLA+, porque 2.4.2 solo lo trae para la K2 Plus.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import os
import struct
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SALIDA = ROOT / "proyecto" / "datos" / "procesados" / "orca_k1c_tiempo_material.csv"
TRABAJO = ROOT / "proyecto" / "resultados" / "orca"
# Probeta de validación: ISO 527-2 tipo 1A (170 x 20 x 4 mm, zona calibrada 10 mm), Printables 653017, JoshEmmel, CC BY-NC 4.0.
PROBETA_ISO527_1A = ROOT / "proyecto" / "datos" / "crudos" / "probetas" / "iso527_1a_testspecimen.stl"
# Copias resueltas (herencia aplanada) de los preajustes de sistema de OrcaSlicer 2.4.2 para la K1C.
ORCA_SISTEMA = ROOT / "proyecto" / "datos" / "crudos" / "orca_v2.4.2"
# Perfiles de sistema de la instalación local de OrcaSlicer (la CLI resuelve su herencia).
ORCA_INSTALADO = Path("C:/Program Files/OrcaSlicer/resources/profiles/Creality")

NIVELES = {
    "altura_capa_mm": [0.1, 0.2, 0.3],
    "densidad_relleno_pct": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "velocidad_impresion_mms": [30, 50, 70],
}

# Patrón de Aktepe y Ergün (nomenclatura de Cura) → patrón de OrcaSlicer. None = sin equivalente.
PATRONES_ORCA = {
    "grid": "grid",
    "triangles": "triangles",
    "zigzag": "zig-zag",
    "lines": "line",
    "cubic": "cubic",
    "gyroid": "gyroid",
    "tri_hexagon": "tri-hexagon",
    "concentric": "concentric",
    "lightning": "lightning",
    "quarter_cubic": None,
    "cubic_subdivision": "adaptivecubic",   # aproximación: ambos son cúbicos adaptativos
    "cross": None,
    "cross3d": None,
    "octet": None,
}


# ----------------------------------------------------------------------------- STL de la probeta
def contorno_d638_iv(n_arco: int = 24) -> list[tuple[float, float]]:
    """Semicontorno superior de la probeta ASTM D638 tipo IV, en mm, centrado en el origen.

    Dimensiones nominales: longitud total 115, ancho de agarre 19, ancho de la zona estrecha 6,
    longitud de la zona estrecha 33, radio de transición 14, radio exterior 25, espesor 3.2.
    Las transiciones se aproximan con arcos tangentes.
    """
    L, W, w, ln, r, R = 115.0, 19.0, 6.0, 33.0, 14.0, 25.0
    puntos: list[tuple[float, float]] = []
    # arco interior de radio r: desde el fin de la zona estrecha (x = ln/2, y = w/2) hacia afuera
    cx, cy = ln / 2, w / 2 + r
    # ángulo del punto de tangencia con el arco exterior (círculo de radio R centrado en (cx2, W/2 - R))
    cx2, cy2 = None, W / 2 - R
    # los dos círculos son tangentes exteriormente: distancia entre centros = r + R
    d = r + R
    dy = cy - cy2
    dx = math.sqrt(max(d * d - dy * dy, 0.0))
    cx2 = cx + dx
    ang_t = math.atan2(cy2 - cy, cx2 - cx)          # dirección del punto de tangencia visto desde el centro interior
    for k in range(n_arco + 1):
        a = -math.pi / 2 + (ang_t + math.pi / 2) * k / n_arco
        puntos.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    ang_s = math.atan2(cy - cy2, cx - cx2)          # mismo punto visto desde el centro exterior
    for k in range(1, n_arco + 1):
        a = ang_s + (math.pi / 2 - ang_s) * k / n_arco
        puntos.append((cx2 + R * math.cos(a), cy2 + R * math.sin(a)))
    puntos.append((L / 2, W / 2))
    return puntos


def triangular_simetrica(superior: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Triangula la probeta como banda entre el contorno superior y su reflejo en y = 0.

    La probeta es simétrica respecto de su eje y cada vertical corta una vez arriba y una abajo,
    así que la banda de cuadriláteros (T_i, T_{i+1}, B_{i+1}, B_i) cubre el polígono sin ambigüedad.
    Es más robusto que el corte de orejas, que se atasca con los puntos colineales de los tramos
    rectos.
    """
    n = len(superior)
    inferior = [(x, -y) for x, y in superior]
    vertices = superior + inferior
    tri = []
    for i in range(n - 1):
        t0, t1, b0, b1 = i, i + 1, n + i, n + i + 1
        tri += [(t0, b0, b1), (t0, b1, t1)]
    return vertices, tri


def escribir_stl(superior: list[tuple[float, float]], espesor: float, ruta: Path) -> float:
    """Escribe el STL binario y devuelve el volumen en mm³ (positivo si las normales apuntan afuera)."""
    verts, tri = triangular_simetrica(superior)
    n = len(superior)
    # contorno cerrado en sentido antihorario (visto desde +z): inferior de izquierda a derecha,
    # superior de derecha a izquierda, para que las paredes queden orientadas hacia afuera.
    lazo = list(range(n, 2 * n)) + list(range(n - 1, -1, -1))
    z0, z1 = 0.0, espesor
    caras = []
    for i, j, k in tri:
        caras.append(((*verts[i], z1), (*verts[j], z1), (*verts[k], z1)))
        caras.append(((*verts[i], z0), (*verts[k], z0), (*verts[j], z0)))
    for a, b in zip(lazo, lazo[1:] + lazo[:1]):
        pa, pb = verts[a], verts[b]
        caras.append(((*pa, z0), (*pb, z0), (*pb, z1)))
        caras.append(((*pa, z0), (*pb, z1), (*pa, z1)))

    def vol(c):
        (a, b, d) = c
        return (a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) + a[2] * (b[0] * d[1] - b[1] * d[0])) / 6

    volumen = sum(vol(c) for c in caras)
    if volumen < 0:                       # normales hacia adentro: invertir todas las caras
        caras = [(c[0], c[2], c[1]) for c in caras]
        volumen = -volumen
    with ruta.open("wb") as fh:
        fh.write(b"ASTM D638 Type IV tensile specimen, generated for OrcaSlicer".ljust(80, b" "))
        fh.write(struct.pack("<I", len(caras)))
        for v1, v2, v3 in caras:
            ux, uy, uz = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
            vx, vy, vz = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            fh.write(struct.pack("<3f", nx / norm, ny / norm, nz / norm))
            for v in (v1, v2, v3):
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))
    return volumen


def contorno_superior_d638_iv() -> list[tuple[float, float]]:
    """Contorno superior completo (y > 0), de izquierda a derecha, sin puntos duplicados."""
    mitad = contorno_d638_iv()
    izq = [(-x, y) for x, y in reversed(mitad)]
    puntos = izq + mitad[1:]
    return puntos


# ----------------------------------------------------------------------------- perfiles de proceso
def como_usuario(preset: dict, nombre: str, tipo: str) -> dict:
    """Convierte una copia resuelta de preajuste de sistema en un preajuste de usuario plano.

    La CLI carga preajustes planos; las claves de herencia y de origen de sistema se retiran.
    """
    p = {k: v for k, v in preset.items() if k not in ("inherits", "from", "setting_id", "instantiation")}
    p.update({"name": nombre, "type": tipo, "from": "User", "version": preset.get("version", "2.4.2.0")})
    return p


def perfil_proceso(base: dict, altura: float, densidad: int, patron_orca: str, velocidad: int, nombre: str) -> dict:
    p = como_usuario(base, nombre, "process")
    p.update({
        "layer_height": str(altura),
        "initial_layer_print_height": str(altura),
        "sparse_infill_density": f"{densidad}%",
        "sparse_infill_pattern": patron_orca,
        # Aktepe y Ergün varían una sola «velocidad de impresión»; se aplica a las trayectorias
        # principales y se deja el resto del perfil base intacto.
        "outer_wall_speed": str(velocidad),
        "inner_wall_speed": str(velocidad),
        "sparse_infill_speed": str(velocidad),
        "internal_solid_infill_speed": str(velocidad),
        "top_surface_speed": str(velocidad),
        "gap_infill_speed": str(velocidad),
    })
    return p


# ----------------------------------------------------------------------------- lectura del G-code
RE_TIEMPO = re.compile(r";\s*(?:estimated printing time|total estimated time)[^=]*=\s*(.+)", re.I)
RE_FIL_G = re.compile(r";\s*(?:total )?filament used \[g\]\s*=\s*([\d.]+)", re.I)
RE_FIL_MM = re.compile(r";\s*(?:total )?filament used \[mm\]\s*=\s*([\d.]+)", re.I)


def segundos(texto: str) -> float:
    total = 0.0
    for val, unidad in re.findall(r"(\d+)\s*([dhms])", texto):
        total += int(val) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unidad]
    return total


def leer_3mf(ruta: Path) -> dict:
    """Lee tiempo y filamento del 3MF que exporta la CLI (Metadata/plate_1.gcode y slice_info.config)."""
    out: dict = {}
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        gcodes = [n for n in nombres if n.startswith("Metadata/plate_") and n.endswith(".gcode")]
        if gcodes:
            cab = z.read(gcodes[0]).decode("utf-8", errors="ignore")[:20000]
            t = RE_TIEMPO.search(cab); g = RE_FIL_G.search(cab); mm = RE_FIL_MM.search(cab)
            out.update(tiempo_estimado_s=segundos(t.group(1)) if t else None,
                       filamento_g=float(g.group(1)) if g else None,
                       filamento_mm=float(mm.group(1)) if mm else None)
        if "Metadata/slice_info.config" in nombres:
            info = z.read("Metadata/slice_info.config").decode("utf-8", errors="ignore")
            pred = re.search(r'key="prediction"\s+value="(\d+)"', info)
            peso = re.search(r'key="weight"\s+value="([\d.]+)"', info)
            if pred and out.get("tiempo_estimado_s") is None:
                out["tiempo_estimado_s"] = float(pred.group(1))
            if peso and out.get("filamento_g") is None:
                out["filamento_g"] = float(peso.group(1))
    if out.get("tiempo_estimado_s") is not None:
        out["tiempo_estimado_min"] = round(out["tiempo_estimado_s"] / 60, 2)
    return out


def laminar_uno(orca: str, maquina: Path, perfil: Path, filamento: Path, stl: Path, salida_3mf: Path, reintentos: int = 1) -> dict:
    """Invoca la CLI de OrcaSlicer para una combinación y devuelve tiempo y filamento."""
    if salida_3mf.exists() and salida_3mf.stat().st_size > 0:
        try:
            r = leer_3mf(salida_3mf)
            if r.get("tiempo_estimado_s") is not None:
                r["reutilizado"] = 1
                return r
        except zipfile.BadZipFile:
            salida_3mf.unlink()
    cmd = [orca, "--debug", "0", "--load-settings", f"{maquina};{perfil}", "--load-filaments", str(filamento),
           "--uptodate", "--slice", "0", "--export-3mf", str(salida_3mf), str(stl)]
    t0 = time.perf_counter()
    for intento in range(reintentos + 1):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, errors="ignore", timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": f"no se pudo ejecutar OrcaSlicer: {e}", "segundos_laminado": round(time.perf_counter() - t0, 1)}
        if res.returncode == 0 and salida_3mf.exists():
            try:
                r = leer_3mf(salida_3mf)
                r["segundos_laminado"] = round(time.perf_counter() - t0, 1)
                if r.get("tiempo_estimado_s") is None:
                    r["error"] = "3MF sin tiempo estimado: revisar exportación"
                return r
            except zipfile.BadZipFile:
                pass
    return {"error": (res.stderr or res.stdout)[-400:].replace("\n", " ") or f"código {res.returncode}",
            "segundos_laminado": round(time.perf_counter() - t0, 1)}


def leer_gcode(ruta: Path) -> dict:
    cab = ruta.read_text(encoding="utf-8", errors="ignore")
    t = RE_TIEMPO.search(cab)
    g = RE_FIL_G.search(cab)
    mm = RE_FIL_MM.search(cab)
    return {
        "tiempo_estimado_s": segundos(t.group(1)) if t else None,
        "tiempo_estimado_min": round(segundos(t.group(1)) / 60, 2) if t else None,
        "filamento_g": float(g.group(1)) if g else None,
        "filamento_mm": float(mm.group(1)) if mm else None,
    }


# ----------------------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orca", default=r"C:\Program Files\OrcaSlicer\orca-slicer.exe")
    ap.add_argument("--maquina", default=str(ORCA_INSTALADO / "machine" / "Creality K1C 0.4 nozzle.json"),
                    help="JSON de perfil de máquina. Debe ser el de sistema instalado: la CLI rechaza copias aplanadas de máquina")
    ap.add_argument("--proceso-base", default=str(ORCA_SISTEMA / "process_0.20mm_Standard_Creality_K1C_0.4_nozzle.json"),
                    help="JSON de perfil de proceso del que heredan las combinaciones; por defecto «0.20mm Standard @Creality K1C 0.4 nozzle» resuelto")
    ap.add_argument("--filamento", default=str(ORCA_SISTEMA / "filament_eSUN_PLA+_K1C_tesis.json"),
                    help="JSON de perfil de filamento plano; por defecto el preajuste eSUN PLA+ derivado de «Generic PLA @K1C» con los valores de la hoja técnica")
    ap.add_argument("--stl", default=str(PROBETA_ISO527_1A),
                    help="probeta a laminar (por defecto la ISO 527-2 tipo 1A de Printables 653017); --stl generar produce la ASTM D638 tipo IV propia")
    ap.add_argument("--espesor", type=float, default=3.2)
    ap.add_argument("--patrones", default="grid,triangles,zigzag,lines,cubic,gyroid,tri_hexagon,concentric,lightning,cubic_subdivision",
                    help="patrones de Aktepe a laminar (nomenclatura del CSV procesado)")
    ap.add_argument("--solo-preparar", action="store_true", help="genera STL y perfiles sin invocar OrcaSlicer")
    ap.add_argument("--paralelo", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)),
                    help="procesos de OrcaSlicer simultáneos (por defecto hasta 6, según núcleos)")
    ap.add_argument("--limite", type=int, default=None, help="laminar solo las primeras N combinaciones (prueba)")
    ap.add_argument("--rehacer", action="store_true", help="ignora los 3MF ya existentes en lugar de reutilizarlos")
    args = ap.parse_args()

    TRABAJO.mkdir(parents=True, exist_ok=True)
    (TRABAJO / "perfiles").mkdir(exist_ok=True)
    (TRABAJO / "gcode").mkdir(exist_ok=True)

    stl = TRABAJO / "probeta_astm_d638_tipo_iv.stl" if args.stl == "generar" else Path(args.stl)
    if args.stl == "generar":
        volumen = escribir_stl(contorno_superior_d638_iv(), args.espesor, stl)
        print(f"STL generado: {stl} | volumen {volumen:.0f} mm³ | masa a 1.24 g/cm³ y 100 % de relleno: {volumen * 1.24 / 1000:.2f} g")

    base = json.loads(Path(args.proceso_base).read_text(encoding="utf-8")) if args.proceso_base else {}
    patrones = [p.strip() for p in args.patrones.split(",") if p.strip()]
    sin_eq = [p for p in patrones if PATRONES_ORCA.get(p) is None]
    if sin_eq:
        print(f"Sin equivalente en OrcaSlicer, se omiten: {sin_eq}")
    patrones = [p for p in patrones if PATRONES_ORCA.get(p)]

    combos = list(itertools.product(NIVELES["altura_capa_mm"], NIVELES["densidad_relleno_pct"], patrones, NIVELES["velocidad_impresion_mms"]))
    print(f"Combinaciones a laminar: {len(combos)}")

    if args.limite:
        combos = combos[: args.limite]

    # Máquina y filamento como preajustes de usuario planos, junto a los perfiles de proceso.
    maquina = Path(args.maquina)
    filamento = TRABAJO / "perfiles" / "_filamento.json"
    filamento.write_text(json.dumps(como_usuario(json.loads(Path(args.filamento).read_text(encoding="utf-8")), "Filamento (tesis)", "filament"), indent=1), encoding="utf-8")

    filas = []
    tareas = {}
    for altura, densidad, patron, velocidad in combos:
        nombre = f"h{altura}_d{densidad}_{patron}_v{velocidad}"
        perfil = TRABAJO / "perfiles" / f"{nombre}.json"
        perfil.write_text(json.dumps(perfil_proceso(base, altura, densidad, PATRONES_ORCA[patron], velocidad, nombre), indent=1), encoding="utf-8")
        fila = dict(impresora="Creality K1C", tipo_impresora="closed", altura_capa_mm=altura, densidad_relleno_pct=densidad,
                    patron_relleno=patron, patron_orca=PATRONES_ORCA[patron], velocidad_impresion_mms=velocidad, perfil=perfil.name)
        filas.append(fila)
        if not args.solo_preparar:
            salida = TRABAJO / "gcode" / f"{nombre}.3mf"
            if args.rehacer and salida.exists():
                salida.unlink()
            tareas[len(filas) - 1] = (perfil, salida)

    if tareas:
        if not Path(args.orca).exists():
            raise SystemExit(f"No se encuentra OrcaSlicer en {args.orca}; indique la ruta con --orca")
        print(f"Laminando {len(tareas)} combinaciones con {args.paralelo} procesos en paralelo...")
        t0 = time.perf_counter()
        hechas = errores = 0
        with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
            futuros = {ex.submit(laminar_uno, args.orca, maquina, perfil, filamento, stl, salida): idx
                       for idx, (perfil, salida) in tareas.items()}
            for fut in as_completed(futuros):
                idx = futuros[fut]
                filas[idx].update(fut.result())
                hechas += 1
                errores += int("error" in filas[idx])
                if hechas % 10 == 0 or hechas == len(tareas):
                    print(f"  {hechas}/{len(tareas)} listas, {errores} con error, {time.perf_counter() - t0:.0f} s transcurridos")
        if errores:
            primero = next(f["error"] for f in filas if "error" in f)
            print(f"Primer error registrado: {primero[:300]}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    campos = sorted({k for f in filas for k in f}, key=lambda k: (k not in filas[0], k))
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"Escrito: {SALIDA} ({len(filas)} filas{'; solo preparación, sin tiempos' if args.solo_preparar else ''})")


if __name__ == "__main__":
    main()
