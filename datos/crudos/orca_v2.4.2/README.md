# Preajustes de sistema de OrcaSlicer 2.4.2 para la Creality K1C (copias resueltas)

Descargados el 23 de septiembre de 2026 desde el repositorio público de OrcaSlicer
(`SoftFever/OrcaSlicer`, etiqueta `v2.4.2`, ruta `resources/profiles/Creality/`). Cada archivo es
la **herencia aplanada** del preajuste: el hijo sobreescribe a sus padres (`fdm_process_creality_common`,
`fdm_process_common`; `fdm_filament_pla`, `fdm_filament_common`), de modo que el JSON es plano y
puede pasarse a `orca-slicer --load-settings`. No se modificó ningún valor.

| Archivo | Preajuste de origen | SHA-256 |
|---|---|---|
| `process_0.20mm_Standard_Creality_K1C_0.4_nozzle.json` | 0.20mm Standard @Creality K1C 0.4 nozzle | `bbf355fb3086ae383dc98274cd7d2a7c56932e1db19ec64e93b8d958e1f5b158` |
| `machine_Creality_K1C_0.4_nozzle.json` | Creality K1C 0.4 nozzle | `4b3061a36067d4e83658ae3d4600a4ebccc18d6a63aa60929a45b74afc5e771d` |
| `filament_Generic_PLA_K1C.json` | Generic PLA @K1C | `eb9e0959dfba55ed1b05ba7db111625fb3e61492934cc76fe4bb544e1a6c0c11` |
| `filament_eSUN_PLA+_K2Plus_referencia.json` | eSUN PLA+ @K2 Plus (solo referencia: 2.4.2 no trae eSUN PLA+ para la K1C) | `003982298b658d62fbd2151b625c84e1d98729fde3f2371895f6b58dc700a601` |

El preajuste de proceso es la **configuración de referencia** del numeral 4.3 del programa
definitivo. Sus valores principales: capa 0.2 mm, relleno 15 % grid, 2 perímetros, 5 capas
superiores y 3 inferiores, pared exterior 200 mm/s, interior 250, relleno 270, primera capa 60,
aceleración 10,000 mm/s². El filamento genérico imprime a 220 °C de boquilla y 50 °C de cama.
Los perfiles son parte de OrcaSlicer (licencia AGPL-3.0); se conservan como registro de la
configuración usada, no como redistribución del programa.
