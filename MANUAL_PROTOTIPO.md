# MANUAL DE USO DEL PROTOTIPO

Recomendador de parámetros de impresión FDM para PLA+ con optimización exacta y explicación.
Anexo del Trabajo de Graduación *Aprendizaje automático explicable para optimizar resistencia y
eficiencia material FDM*. Versión del 29 de septiembre de 2026.

## 1. Qué hace y qué no hace

El prototipo recomienda una combinación de altura de capa, densidad y patrón de relleno y
velocidad de impresión que maximiza la resistencia a la tracción predicha, sujeta a las cotas y
prioridades que el usuario introduce. La recomendación es la configuración de mayor puntuación ponderada S = w_r·R_n + w_q·Q_n − w_m·M_n − w_t·T_n, con las respuestas normalizadas al dominio y los pesos sumando 1; junto a ella se muestra la solución de compromiso, que no depende de los pesos. La normalización por rango del dominio favorece a las configuraciones lentas cuando el rango de tiempo es ancho (un +45 % de tiempo sobre la referencia es solo un 8 % del rango); la normalización respecto de la referencia corrige ese efecto. La predicción
la hace un sustituto neuronal entrenado con 483 combinaciones del conjunto de Aktepe y Ergün; el
tiempo y el filamento provienen del laminado real con OrcaSlicer 2.4.2 de la probeta ISO 527-2 tipo 1A
en la Creality K1C; la optimización es un programa lineal entero mixto resuelto al óptimo
global con HiGHS; y la explicación usa valores de Shapley.

No certifica seguridad estructural, no extrapola fuera del dominio de entrenamiento y no sustituye
la validación física. Toda predicción se muestra con su error validado (MAE) y con su respaldo en
datos.

## 2. Instalación y ejecución con Docker (cualquier sistema)

Es la vía recomendada para quien evalúa el prototipo: no requiere Python, OrcaSlicer ni Windows,
solo Docker (Docker Desktop en Windows o macOS, o Docker Engine en Linux). Desde la raíz del
repositorio:

```bash
docker compose up --build
```

La primera construcción descarga las dependencias y tarda unos minutos; después la interfaz
queda en `http://localhost:8501`. Para detenerla, `Ctrl+C` y `docker compose down`. La imagen
incluye el sustituto entrenado, la tabla del laminador y el conjunto de Pareto, así que es
autocontenida y reproducible: la etiqueta de la imagen lleva la fecha de la versión.

## 3. Instalación y ejecución sin Docker (Windows)

Alternativa para desarrollo. Requiere Python 3.12 o 3.13. Desde la raíz del repositorio:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r proyecto\requirements-app.txt
.venv\Scripts\python.exe -m streamlit run proyecto\src\app.py
```

Se abre en el navegador en `http://localhost:8501`. La primera carga tarda unos 15 segundos
(carga del modelo y del explicador); cada cambio posterior, entre 1 y 3 segundos. No hace falta
OrcaSlicer para usar la interfaz: la tabla de tiempo y filamento ya está calculada en
`proyecto/datos/procesados/orca_k1c_tiempo_material.csv`; sí hace falta para regenerarla con
otra geometría (`proyecto/src/estimar_tiempo_orca.py`).

## 4. Pantalla

La barra lateral contiene las entradas; la página principal, los resultados.

| Zona | Contenido |
|---|---|
| Barra lateral, 1. Restricciones | Cotas de rugosidad máxima, tiempo máximo, filamento máximo, densidad máxima y altura de capa mínima; patrón de relleno libre o fijado |
| Barra lateral, 2. Prioridades | Perfil de ejemplo (Resistencia máxima, Equilibrado, Económico, Acabado) y cuatro pesos de 0 a 10 para resistencia, calidad, filamento y tiempo y energía, que se normalizan para sumar 1 |
| Barra lateral, 2. Normalización | «Rango del dominio» escala cada objetivo entre su mínimo y su máximo en todas las combinaciones laminables; «Respecto de la referencia» divide cada objetivo por su valor en la configuración de referencia, de modo que S mide cambios proporcionales y un +45 % de tiempo pesa tanto como un −45 % de filamento |
| Barra lateral, 3. Energía | Potencia media de la impresora en W; si se indica, la energía se estima como potencia por tiempo |
| Barra lateral, 4. Versión | Identificador del modelo y su MAE validado por respuesta |
| Recomendación | La configuración de mayor puntuación S bajo las cotas, la solución de compromiso (rodilla de la frontera) y la referencia proyectada, con sus predicciones ± MAE, tiempo, filamento, energía y S; aviso de respaldo en datos o de extrapolación, y análisis de estabilidad de la recomendación ante ±10 % en los pesos |
| Cambio frente a la referencia | Cinco indicadores con la diferencia absoluta y relativa, y la regla del programa definitivo (degradación admisible y mejoras mínimas) |
| Por qué | Contribución de cada variable a cada respuesta respecto de la combinación típica del dominio, con una frase que nombra el factor dominante |
| Alternativas no dominadas | Las configuraciones del conjunto de Pareto exacto que cumplen las cotas, ordenadas por S, en tabla y en gráfica tiempo–resistencia |
| Comparar dos configuraciones | Selección manual de dos combinaciones y sus predicciones lado a lado |
| Dominio de validez | Rangos de entrada, temperaturas fijas, patrones con y sin laminado, e importancia global de las variables |

## 5. Tareas típicas

1. **Obtener una recomendación sin restricciones.** Deje los controles por defecto. La
   recomendación es la combinación de mayor resistencia predicha del dominio.
2. **Imponer una cota.** Mueva «Rugosidad Ra máxima» a 4 µm y «Tiempo máximo» a 30 min. La
   recomendación cambia y los indicadores muestran cuánto se cede en resistencia.
3. **Cambiar prioridades.** Elija el perfil «Económico» o suba el peso de «Tiempo y energía».
   La puntuación S reordena las alternativas y la recomendación cambia; la línea de estabilidad
   dice si pesos parecidos darían otra recomendación.
4. **Interpretar la explicación.** En «Por qué», la barra más larga es la variable que más aleja
   la predicción del valor base; el texto la nombra. Las contribuciones describen el modelo, no
   causalidad física.
5. **Comparar dos candidatas.** En «Comparar dos configuraciones», seleccione A y B y lea las
   predicciones con su ± MAE. Si las bandas se solapan, la diferencia está dentro del error.
6. **Caso sin solución.** Con cotas incompatibles (por ejemplo Ra ≤ 2 µm y tiempo ≤ 20 min) la
   interfaz informa que ninguna combinación laminable las cumple y pide relajar alguna.

## 6. Advertencias que muestra

- **Extrapolación**: la combinación de impresora y patrón no tiene datos de entrenamiento.
- **Sin laminado**: el patrón no tiene equivalente en OrcaSlicer, así que no hay tiempo ni
  filamento y no puede imprimirse tal cual.
- **Dentro del error**: una diferencia menor que el MAE no debe interpretarse como mejora.
- **Densidad en pasos de 10 %**: los datos de entrenamiento solo tienen 20, 50 y 80 %. Los niveles 30, 40, 60 y 70 % son interpolación del modelo; 10, 90 y 100 % están fuera del rango y se admiten con la casilla «Permitir densidades fuera del rango de entrenamiento». En todos los casos el tiempo y el filamento son del laminador, pero las propiedades en los niveles no medidos son una predicción del modelo y deben validarse.

## 7. Limitaciones

Dominio: PLA+ en impresora de cámara cerrada; altura de capa 0.1 a 0.3 mm, relleno 20 a 80 %,
velocidad 30 a 70 mm/s; temperaturas fijas de 215 a 220 °C y 70 °C. El filamento de validación
es eSUN PLA+, cuya marca no coincide necesariamente con la del conjunto de entrenamiento. La
energía es una estimación por potencia media. Las predicciones de resistencia y rugosidad se
validaron contra datos de prueba retenidos, no contra ensayos propios.

## 8. Archivos

| Archivo | Función |
|---|---|
| `Dockerfile`, `compose.yaml` | Imagen autocontenida del prototipo |
| `proyecto/requirements-app.txt` | Dependencias mínimas de la interfaz, con versiones fijadas |
| `proyecto/src/app.py` | Interfaz |
| `proyecto/src/optimizar_milp.py` | Optimizador exacto que la interfaz invoca |
| `proyecto/src/explicar.py` | Explicaciones locales y globales |
| `proyecto/resultados/modelos/sustituto_mlp.{joblib,json}` | Sustituto entrenado y sus pesos |
| `proyecto/resultados/tablas/milp_pareto_exacto_discreto_closed.csv` | Conjunto de Pareto exacto |
| `proyecto/datos/procesados/orca_k1c_tiempo_material.csv` | Tiempo y filamento por combinación |

## Anexo. Evaluación heurística del prototipo (7 de octubre de 2026)

Recorrido del investigador en el navegador, con la imagen de contenedor construida desde la carpeta de
entrega (`docker compose up --build`), sobre las tareas del numeral 5.

| Tarea | Completada | Tiempo aproximado | Errores de navegación | Dudas |
|---|---|---|---|---|
| Carga inicial de la aplicación | Sí | Más de 2 min | Ninguno | Ninguna |
| Cambiar el perfil de ejemplo | Sí | Alrededor de 1 min | Ninguno | Ninguna |
| Cambiar una restricción | Sí | 30 s | Ninguno | Ninguna |
| Comparar dos configuraciones | Sí | 33 s | Ninguno | Ninguna |

Incidencias: solo de tiempo de respuesta. La carga inicial enumera las 630 y 900 combinaciones del
dominio para fijar las constantes de normalización; el cambio de prioridades tarda alrededor de un
minuto porque, además de resolver el programa lineal entero mixto (uno a tres segundos), repite la
optimización ocho veces para el análisis de estabilidad. Acciones recomendadas: precalcular la
enumeración en la imagen y hacer opcional el análisis de estabilidad.
