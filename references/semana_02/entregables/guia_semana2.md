# Entregable — Semana 2

**Nombre:**
**Fecha:**

Dos partes: un resumen de la maquinaria y el estado de Earth Engine + el repo.

---

## Parte A — Resumen de la maquinaria

Con sus palabras (no copiar del book). El centro de esta semana es **cómo una imagen se convierte en un vector**.

### Vision Transformers (lo principal)

- Expliquen la **lógica de un Vision Transformer**: por qué tratar una imagen como una secuencia de "tokens", igual que un Transformer trata una oración como secuencia de palabras.
- **Patchification**: cómo se parte la imagen en parches, por qué cada parche se vuelve un token, y qué pasa con su tamaño (más parches = secuencia más larga).
- Qué hace la **proyección lineal** de cada parche y por qué.
- Para qué sirve el **positional encoding** (un Transformer no tiene noción de orden por sí solo).
- Qué es el **CLS token** y por qué de ahí se lee el embedding final.
- En una frase, qué hace **self-attention** (cada token mira a los demás y decide cuánto tomar de cada uno).

### El resto de la maquinaria

- **Masked autoencoders**: por qué esconder el 75% de la imagen y reconstruirla enseña algo útil sin etiquetas; por qué al final se tira el decoder.
- **Imagen satelital**: qué es un tensor `(C, H, W)`, qué son las bandas, y por qué NDVI se queda corto frente a un embedding.
- **Embedding como vector**: por qué las dimensiones no tienen significado individual y cómo se compara con cosine similarity.
- **Modelos**: diferencia explícito vs. implícito; dónde cae AlphaEarth y por qué es el más fácil de usar (GEE).

---

## Parte B — Earth Engine + repo

- ¿Pudieron crear cuenta y proyecto de Earth Engine? Anoten cualquier traba.
- Peguen el snippet o capturen el resultado de cargar `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` y visualizar un punto.
- ¿Cómo se ve un embedding de 64 dims en la práctica? (qué obtuvieron para un punto)
- Link del repo de GitHub (con `kennyldc` ya agregado).
- Qué tienen del script de extracción (aunque sea borrador) y qué les falta.

---

## Conceptos para la entrevista del viernes

Algunos temas que vale la pena tener claros. Como la semana pasada, **no necesariamente será esto palabra por palabra**, pero por aquí empieza.

- La lógica de un ViT, de principio a fin (imagen → parches → tokens → atención → CLS → embedding)
- Qué es patchification y qué cambia si agrandas o achicas el parche
- Por qué hace falta el positional encoding
- Qué problema resuelve el masked autoencoder y por qué no necesita etiquetas
- Por qué se descarta el decoder después de entrenar
- Diferencia entre modelos explícitos e implícitos
- Qué es la colección de AlphaEarth en GEE (64 bandas, 10 m, anual) y cómo se consulta
- Qué encontró Gong: en qué outcomes los embeddings predicen bien y en cuáles no
