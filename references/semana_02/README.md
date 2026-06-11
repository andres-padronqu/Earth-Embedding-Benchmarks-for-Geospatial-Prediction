# Semana 2 — La maquinaria + primeros pasos en Earth Engine

**40 horas** · 5 días × 8 h/día
**Reunión:** viernes (equipo + 1:1 de 25 min)

---

## De qué va esta semana

La semana 1 fue el *qué* y el *para qué*. Esta semana toca el *cómo*: la maquinaria que hay detrás de un embedding (Vision Transformers, masked autoencoders, los modelos) y, en paralelo, que ya se ensucien las manos con **Google Earth Engine**, que es de donde vamos a sacar los embeddings de AlphaEarth.

Dos frentes:

1. **Entender la maquinaria** — con las notas técnicas del book (carpeta `book/`) y el paper de aplicación de Gong.
2. **Empezar a sacar embeddings** — crear su cuenta/proyecto de Earth Engine y un repo donde empiecen a escribir el código de descarga.

No espero que terminen el pipeline esta semana. Tendrán más tiempo. La meta es que **entiendan la maquinaria** y que **ya estén dentro de Earth Engine explorando cómo se extraen los vectores**.

A partir de aquí ya le vamos bajando a las lecturas para pasar más a lo *hands-on*. Pero esta base teórica la necesitaban: es lo que les va a evitar cometer errores en el reporte final y en sus proyectos. De aquí en adelante leemos menos y construimos más.

---

## Las notas técnicas (`book/`)

Es un recurso que escribí como apoyo: entre glosario técnico y resumen de toda la maquinaria. Está en Markdown, capítulo por capítulo. Léanlo como notas, no como un libro de cabecera.

Prioridad para esta semana:

- `04-vision-transformers.md` — **el más importante.** Cómo un ViT convierte una imagen en un vector.
- `05-masked-autoencoders.md` — cómo se entrena ese modelo sin etiquetas.
- `02-satellite-imagery.md` — qué es realmente una imagen satelital (bandas, tensores, NDVI).
- `03-what-are-embeddings.md` — qué es el vector y cómo se compara/usa.
- `06-foundation-models.md` — el mapa de modelos (Clay, Prithvi, SatCLIP, AlphaEarth, MOSAIKS).
- `07-complete-pipeline.md` — cómo se conecta todo de imagen a predicción.

El capítulo `01` es contexto; opcional esta semana.

Mismas reglas que la semana pasada: **sí lean, sin atajos.** Si algo no cierra, googléenlo o busquen otra fuente hasta entenderlo.

### Cómo leer estas notas

Están en Markdown, con fórmulas en LaTeX (`$...$`) y bloques de código. En la *preview* básica de algunos editores las fórmulas no se renderean y se ven como texto crudo. Para verlas bien:

- **Lo más fácil:** súbanlas a su repo de GitHub y léanlas ahí — GitHub renderea las fórmulas y los bloques de cita sin que hagan nada.
- **En VS Code / Cursor:** usen un visor de Markdown con soporte de matemáticas (hay extensiones de "Markdown + Math"); la preview por defecto no siempre las muestra.
- Si ven algo tipo `$$ ... $$` como texto plano, no está roto: es una fórmula que su visor no está renderizando.

---

## El paper de aplicación: Gong et al. (`papers/`)

`01_gong_urban_indicators_2026.pdf`

Es uno de los pocos papers que muestra una aplicación posible de los earth embeddings: un benchmark de **indicadores urbanos** que usa **AlphaEarth embeddings** sobre áreas metropolitanas de EE.UU. Sirve como referencia de hacia dónde puede ir esto.

Al leerlo, fíjense en:

- Qué indicadores predicen y a qué unidad geográfica (vecindarios / census tracts).
- Qué modelos comparan y cuál gana.
- En qué outcomes los embeddings funcionan bien y en cuáles no (entorno construido vs. comportamiento).
- Cómo evalúan (métricas, validación).

---

## Earth Engine — manos a la obra

Esta semana quiero que ya entren a Earth Engine y empiecen a explorar **cómo se sacan los embeddings**.

1. **Creen su cuenta y proyecto de Earth Engine** (es gratis para uso académico/investigación). Lo que necesiten: cuenta, proyecto en la nube, API, etc.
   - Registro: https://earthengine.google.com/
2. **Lean la documentación del dataset** de embeddings que vamos a usar:
   - https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?hl=es-419
   - Es la colección `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`: 64 bandas (`A00`–`A63`), un vector de 64 dimensiones por píxel de 10 m, por año.
3. **Jueguen con el Code Editor** (JavaScript) usando el ejemplo del catálogo: carguen la colección, filtren por fecha y por un punto, visualicen. Que vean cómo se ve un embedding en el mapa.
4. **Empiecen a ver cómo se extrae para análisis** (Python, `earthengine-api` / `geemap`): dado un punto o polígono, cómo obtener el vector de 64 dims. No tiene que quedar perfecto — es exploración.

---

## El repositorio

Si no lo han hecho, **creen un repo de GitHub** para el proyecto. Ahí va a vivir todo el código.

- Agréguenme como colaborador: mi usuario es **`kennyldc`**.
- Con el repo creado y yo agregado es más que suficiente por ahora. Si además dejan un primer script o notebook de cómo se conecta a Earth Engine, mejor — pero no es obligatorio esta semana.

---

## Ritmo sugerido

| Día | Horas | Qué hacer |
|-----|------:|-----------|
| **Lunes** | 8 | Kickoff (async). Notas técnicas: ViT (`04`) y MAE (`05`) |
| **Martes** | 8 | Resto de la maquinaria (`02`, `03`, `06`, `07`) |
| **Miércoles** | 8 | Paper de Gong + relacionarlo con la Parte 2 de mi paper |
| **Jueves** | 8 | Earth Engine: cuenta/proyecto, Code Editor, leer catálogo, explorar extracción |
| **Viernes** | 8 | Repo creado (+ agregarme); reunión + 1:1 |

---

## Entregable

`entregables/guia_semana2.md` — resumen de la maquinaria (con énfasis en ViT y patchification) + estado de Earth Engine y el repo.

---

## Contacto

Carlos López de la Cerda — carlos.l@wustl.edu
