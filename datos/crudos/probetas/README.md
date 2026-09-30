# Probetas de validación

| Archivo | Origen | Licencia | Geometría |
|---|---|---|---|
| `iso527_1a_testspecimen.stl` | EMMEL, Josh. *3D Printed Mechanical Properties Test Specimen*. Printables, modelo 653017, publicado el 20 de noviembre de 2023. https://www.printables.com/model/653017-3d-printed-mechanical-properties-test-specimen | CC BY-NC 4.0 (atribución, no comercial) | Probeta de tracción ISO 527-2 tipo 1A. Medida sobre el STL: 170 × 20 × 4 mm, zona calibrada de 10 mm de ancho; STL binario estanco de 108 triángulos, 9,609 mm³, 11.92 g de PLA+ a 1.24 g/cm³ con relleno pleno |

El archivo se conserva tal como se descargó (30 de septiembre de 2026); no se modifica. Es la probeta
que se lamina para la tabla de tiempo y filamento (`estimar_tiempo_orca.py`, opción `--stl` por
defecto) y la que se imprime en la Creality K1C para la validación física.

Antes de adoptarla, el laminado se hizo sobre una probeta ASTM D638 tipo IV generada por el propio
script (115 × 19 × 3.2 mm, 4,812 mm³); su tabla se conserva en
`proyecto/datos/procesados/orca_k1c_tiempo_material_d638iv.csv` y el STL se regenera con
`--stl generar`. La tipo 1A tiene el doble de volumen, por lo que tiempo y filamento por pieza
aproximadamente se duplican; la resistencia (MPa) y la rugosidad no dependen del tamaño de la
probeta, aunque el experimento de origen (Aktepe y Ergün) ensayó la ISO 527-2 tipo 4, más pequeña.
