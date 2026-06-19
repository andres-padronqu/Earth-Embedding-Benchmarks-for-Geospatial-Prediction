# Semana 3 — Papers aplicados + acceso a embeddings

**Reunión:** viernes (equipo + 1:1 de 25 min)

---

## De qué va esta semana

Semanas 1–2 fueron fundamentos y maquinaria. Esta semana pasamos a lo aplicado: leer los papers con atención, ver qué publicaron los autores (docs, código, datos), y configurar de verdad cómo se obtienen embeddings.

Ya no alcanza con “leí el abstract”. Tienen que entender cómo replicar o reutilizar lo que hicieron Gong y Yue, y cómo sacar vectores desde código.

---

## Lecturas (`papers/`)

| # | Archivo | De qué va |
|---|---------|-----------|
| 1 | `01_gong_urban_indicators_2026.pdf` | Benchmark urbano USA — AlphaEarth vs Prithvi vs Clay |
| 2 | `02_yue_economic_activity_2026.pdf` | Actividad económica (GDP) con embeddings + luces nocturnas |

Repito Gong a propósito: la semana pasada lo vieron en contexto de maquinaria; ahora léanlo pensando en replicación.

Para cada paper: lean el PDF completo y busquen **todo** lo que publicaron los autores — repos, datos, appendices, código. La idea es que al terminar la semana sepan lo suficiente para **replicar** el paper, no solo resumirlo.

---

## Modelos — investigar y documentar

La lista está en `modelos_para_investigar.md`. Para **cada** modelo tienen que:

1. Investigar cómo se configura y cómo se obtienen vectores (que lo busquen ustedes).
2. Hacerlo funcionar en la práctica — prioricen AlphaEarth, pero **revisen todos** los de la lista e investigar en cuál es factible hacer un pipeline. 
3. Escribir un **mini tutorial** que le puedan pasar a cualquier otra persona: qué es el modelo, cómo configurarlo, cómo sacar vectores. Detallado, en **inglés**, con calidad de algo publicable en formato de divulgación.

Formato libre: markdown, PDF, Jupyter, video, lo que quieran. Lo que importa es que otra persona pueda seguirlo sin preguntarles nada.

En el repo: una carpeta por modelo (`alphaearth/`, `clay/`, etc.) con el tutorial y lo que necesiten para reproducirlo.

Compartan conmigo acceso a lo que configuren. En Google (**caribaz@gmail.com**) — proyecto, assets, o archivos en Drive, lo que aplique.

---

## Reporte de la estancia

Redactar las **primeras secciones técnicas** del reporte final, que es el entregable final para la maestría. Ya podrían tener introducción y secciones iniciales. Es el borrador de la parte metodológica/conceptual — lo que un lector necesita entender antes de ver lo que ustedes construyeron después.

Tomen como base lo de semanas 1–2 (papers, book en `../semana_02/book/`). No es copiar el book: es **su síntesis** y agregar detalles adicionales que hayan investigado para alguien que no estuvo en la estancia. Revisar cuál es la dimnesión que les piden del reporte y piensen cuantas páginas deberían cubrir de introducción y fundamentos.

Temas que deberían quedar cubiertos:

- **Qué son los earth embeddings** — definición, para qué sirven, las cuatro funciones de Klemmer (compresión, transferencia, alineación, representación).
- **Imágenes satelitales como insumo** — qué es realmente una imagen satelital (bandas, tensores), por qué el dato crudo es demasiado grande para usarse directo.
- **Cómo se producen los embeddings** — ViT, patchification, masked autoencoders; la idea de entrenar sin etiquetas.
- **Panorama de modelos** — explícitos vs implícitos; AlphaEarth, Clay, Prithvi, SatCLIP (y lo que hayan investigado esta semana).
- **Pipeline aplicado** — de coordenadas a embedding, merge con datos tabulares, predicción de un outcome (como en Gong o Yue).

Si les falta contexto, vuelvan al book o a los papers de semana 1. El reporte tiene que demostrar que entienden la maquinaria, no que memorizaron definiciones.

---

## Repo

Sigan en GitHub con **`kennyldc`** como colaborador.

---

## Entregable

`entregables/guia_semana3.md`

---

## Contacto

Carlos López de la Cerda — carlos.l@wustl.edu
