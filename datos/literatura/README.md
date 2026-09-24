# TABLAS EXTRAÍDAS DE LITERATURA

Matriz de extracción prevista en el método aprobado. Cada archivo es **una campaña separada**
y se agrupa como tal en la validación; no se fusionan campañas (regla 3 de uso de
`referencias/fuentes_y_datasets.md`). Los valores se transcriben tal como aparecen en el
artículo; las medias son las publicadas por los autores, no recalculadas.

| Archivo | Fuente | Material | Equipo y protocolo | Factores | Respuestas | Réplicas |
|---|---|---|---|---|---|---|
| `kam2022_esun_plaplus_L9.csv` | KAM, Menderes; İPEKÇİ, Ahmet y ŞENGÜL, Ömer. *Scientia Iranica B*. **29**(1), pp. 79-89, 2022. DOI 10.24200/sci.2021.57012.5020 | **eSUN PLA+**, 1.75 mm | Rigid3D Zero2, boquilla 0.4 mm, Simplify3D; extrusor 205 °C, mesa 60 °C, capa 0.2 mm, ancho 0.35 mm, 60 mm/s; ISO 527 tipo IV (tracción, UTEST 10 t) e ISO 180 tipo I (Izod) | Taguchi L9: patrón (rectilinear, triangular, full honeycomb), ocupación 10/30/50 %, orientación en mesa 0/60/−45° | Tensión máxima (MPa), elongación a rotura, impacto Izod (kJ/m²) | 3 por corrida, publicadas individualmente |

Notas sobre `kam2022`:

- Los autores reportan la elongación como fracción (0.018 a 0.030), no en por ciento; se
  conserva tal cual. Es anómalamente baja para PLA+ y probablemente sea desplazamiento
  relativo del travesaño y no deformación en la zona calibrada. No comparar con la elongación
  de otras campañas.
- En la corrida 9 la media de tracción publicada (30.81 MPa) no coincide con la media de los
  tres ensayos transcritos (30.98 MPa). Se transcriben ambas tal como aparecen en el artículo;
  al modelar, recalcular la media a partir de los ensayos individuales.
- Tres factores variados, pero solo **ocupación** es un parámetro de proceso continuo; patrón y
  orientación son categóricos. Cumple el criterio c) del anteproyecto (dos o más parámetros).
- Es la única fuente localizada con la **misma marca y grado de filamento** disponible para la
  validación física (eSUN PLA+). Sirve para comprobar si el envolvente de resistencia del
  filamento propio coincide con el de la campaña de Aktepe y Ergün, cuya marca no está
  declarada.
- La hoja técnica de eSUN reporta 45.6 MPa de tensión máxima según la tabla 1 del artículo
  (2022), 53.3 MPa en la TDS v4.0 y 70.4 MPa en la página de producto de 2026, todas a 100 %
  de relleno y en distintas normas. El propio fabricante ha cambiado la formulación o el
  protocolo: registrar **lote y fecha de compra** del filamento usado en la validación.
