# RAG local sobre Fichas de Datos de Seguridad (FDS) de pinturas

Parcial final de Procesamiento de Lenguaje Natural (PCIA5011, 2026-2), Universidad Sergio Arboleda.

Integrantes: Thomas Ariza, Julian Cristancho y Juan Marquez.

Sistema de recuperación aumentada por generación (RAG) que responde preguntas sobre las 101 FDS
suministradas (1134 páginas, 2018 a 2025), citando para cada afirmación el fabricante, el producto,
la versión, el documento, la página y la sección de donde sale. Todo corre en local con modelos
abiertos; solo la evaluación con RAGAS usa la API gratuita de Gemini (ver
[Evaluador](#evaluador-de-ragas-gemini)).

> Estado: ver la sección [Resultados](#resultados). Las cifras de este documento salen de los
> archivos generados por el propio pipeline (`data/metadata`, `data/verificacion`, `eval/resultados`,
> `docs/recursos.json`).

## Contenido

1. [Arquitectura](#arquitectura)
2. [Instalación](#instalación)
3. [Ejecución](#ejecución)
4. [Extracción PDF → Markdown](#extracción-pdf--markdown)
5. [Procedencia, versión y secciones](#procedencia-versión-y-secciones)
6. [Chunking, embeddings y recuperación](#chunking-embeddings-y-recuperación)
7. [Generación, citas y evidencia insuficiente](#generación-citas-y-evidencia-insuficiente)
8. [Evaluación con RAGAS](#evaluación-con-ragas)
9. [Resultados](#resultados)
10. [Hardware, consumo y costo](#hardware-consumo-y-costo)
11. [Limitaciones y mitigaciones](#limitaciones-y-mitigaciones)
12. [Estructura del repositorio](#estructura-del-repositorio)

## Arquitectura

```
PDF (101 FDS)
  │  rag_fds.extract      Docling 2.150 (CPU): diseño, tablas (TableFormer), imágenes + clasificación
  │                       + reemparejado etiqueta/valor + bloque de encabezados y pies de página
  │                       + OCR RapidOCR solo sobre las imágenes + reparación de tablas con filas fusionadas
  ▼
Markdown por ficha (data/md/FDSxxx/FDSxxx.md + images/ + tablas.json)
  │  rag_fds.inventory    procedencia, fabricante, producto, versión y fechas con evidencia;
  │                       16 secciones: presente / ausente en el original / fallo de extracción
  │  rag_fds.verify       una tabla por ficha contra la capa de texto del PDF + recortes para revisión
  ▼
Fragmentos por sección (rag_fds.index): ≤ 1500 caracteres, solapamiento 200, cabecera con
fabricante, producto, versión, documento y sección
  │  embeddings bge-m3 (Ollama, GPU) → Chroma (coseno)
  ▼
Consulta (rag_fds.rag / rag_fds.ask)
  detección de producto, versión y fabricante → filtro de metadatos
  → recuperación híbrida (coseno + BM25, fusión RRF) → top-k
  → llama3.2 3B (Ollama, GPU, temperatura 0) con citas [n] → revisión de citas y fuentes
  ▼
Evaluación (rag_fds.evaluate): RAGAS 0.4 — Faithfulness, Context Precision, Context Recall
  con Gemini como evaluador; top-k 3 vs. 6
```

Decisiones principales:

| Componente | Elección | Por qué |
|---|---|---|
| Extracción | Docling (open source) | Conserva tablas, listas, títulos e imágenes y da la posición (página y caja) de cada elemento, que es la base de la trazabilidad. |
| OCR | RapidOCR (PP-OCRv6 multilingüe, ONNX) solo en imágenes | Las 101 FDS tienen capa de texto; el OCR de página costaba 65 de 76 s por ficha sin aportar texto. |
| Generador | `llama3.2` (3B) en Ollama | Indicado por el profesor; cabe en la GPU de 6 GB con contexto de 8192 tokens. |
| Embeddings | `bge-m3` en Ollama | Multilingüe, buen desempeño en español, 8192 tokens de entrada. |
| Base vectorial | Chroma (persistente, coseno) | Local, sin servidor, filtros por metadatos. |
| Chunking | Por sección de la FDS | Una pregunta sobre una FDS casi siempre apunta a una sección (p. ej. transporte = 14). |
| Dependencias | uv (`pyproject.toml`, `uv.lock`, `.python-version`) y `requirements.txt` | Versiones fijadas y reproducibles. |

## Instalación

Requisitos: Windows, Linux o macOS; Python 3.12; [uv](https://docs.astral.sh/uv/);
[Ollama](https://ollama.com) 0.40 o superior. Con GPU NVIDIA el generador responde en 1 a 3 s;
sin GPU funciona, pero más lento.

```bash
uv sync
```

Quien no use uv puede instalar con pip (torch de CPU):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Modelos locales:

```bash
ollama pull llama3.2
```

```bash
ollama pull bge-m3
```

Corpus: copie la carpeta `FDS EVALUADAS` (con las subcarpetas `FDS 2018` a `FDS 2025`) dentro de
`data/fds_pdf/`. Los PDF originales no se suben al repositorio.

Clave de Gemini (solo para la evaluación): copie `.env.example` como `.env` y pegue la clave en
`GOOGLE_API_KEY=`. El archivo `.env` está en `.gitignore`.

## Ejecución

Cada paso se puede repetir; los resultados quedan en `data/` y `eval/`.

```bash
uv run python -m rag_fds.extract
```

En Windows, el lector de PDF de Docling (`docling_parse`) a veces se cae con una violación de
acceso (código 139 o 0xC0000005), no siempre en la misma ficha. Basta con repetir el comando:
retoma las fichas que faltan. La extracción completa toma unos 65 minutos en CPU.

```bash
uv run python -m rag_fds.inventory
```

```bash
uv run python -m rag_fds.verify
```

```bash
uv run python -m rag_fds.index
```

Consultas (demo):

```bash
uv run python -m rag_fds.ask "¿Cuál es el punto de inflamación del Esmalte Uretano Part A de Sika?"
```

```bash
uv run python -m rag_fds.ask --contextos
```

Por defecto se recuperan 3 fragmentos; `--k 6` usa la otra configuración evaluada.

`extract` ya repara las tablas con filas fusionadas; `uv run python -m rag_fds.tablas_fix` lo
repite sobre todo el corpus y rehace `data/metadata/tablas_reconstruidas.csv`.

Evaluación (las preguntas de desarrollo primero; luego las 12 de evaluación):

```bash
uv run python -m rag_fds.evaluate desarrollo
```

```bash
uv run python -m rag_fds.evaluate generar
```

```bash
uv run python -m rag_fds.evaluate prueba
```

```bash
uv run python -m rag_fds.evaluate puntuar
```

```bash
uv run python -m rag_fds.evaluate diagnostico
```

```bash
uv run python -m rag_fds.evaluate resumen
```

El evaluador por defecto es `gemini-3.5-flash-lite`. Otro modelo se escoge con la variable de
entorno `RAGAS_MODELO`, y cada modelo guarda sus resultados en su propia carpeta
(`eval/resultados/evaluador_<modelo>/`). Si se agota la cuota diaria del plan gratuito, `puntuar`
se detiene y deja lo que falta "sin evaluar"; al repetirlo retoma solo eso.

Registro de hardware y consumo:

```bash
uv run python -m rag_fds.recursos
```

Interfaz web para la demostración (solo en el propio equipo, http://127.0.0.1:8000):

```bash
uv run python -m rag_fds.web
```

## Extracción PDF → Markdown

Módulo `rag_fds/extract.py` (más `kv_fix.py`, `tablas_fix.py` y `pdftexto.py`). Por cada ficha
produce `data/md/FDSxxx/FDSxxx.md`, la carpeta `images/` y `tablas.json`.

1. **Docling en CPU**, sin OCR de página: las 101 FDS tienen capa de texto (solo 2 de 1134
   páginas tienen menos de 200 caracteres: una en blanco y una de cierre). En la prueba con
   FDS 29 el OCR de página consumía 65 de 76 s. TableFormer en modo `ACCURATE`.
2. **Marcas de página**: cada página se exporta por separado y empieza con `<!-- pagina: N -->`.
   Así cada línea del Markdown, y luego cada fragmento, conserva su página.
3. **Reemparejado etiqueta-valor** (`kv_fix.py`): en las plantillas de dos columnas Docling deja
   todas las etiquetas y luego todos los valores. Pasa con los ":" en la columna de valores
   (AkzoNobel/International, Axalta, PPG, Sika 2023+) y con los ":" pegados a la etiqueta (Pintuco,
   Corona, Invesa, Pintuland, Bler, Sika 2019-2020). Se vuelven a unir por la posición vertical de
   cada bloque en la página y se escriben como `- **Etiqueta:** valor`. El texto no se modifica.
   La numeración de subsección ("14.3"), que Docling guarda como marcador de lista y no como
   texto, se pasa al comienzo de la etiqueta; antes se perdía al unir los bloques (104 números en
   25 fichas).
4. **Encabezados y pies de página**: Docling los separa del cuerpo ("furniture"), pero ahí suelen
   estar la versión, las fechas y la nota legal. Se agregan una sola vez al final del Markdown,
   con las páginas donde aparecen.
5. **Imágenes**: se guardan como PNG (144 dpi) en `images/`, una vez por contenido, y cada
   aparición se enlaza con ruta relativa y un texto alternativo con tipo, página y sección. El
   tipo lo da el clasificador de figuras de Docling (logo, ícono o pictograma, gráfico, etc.).
   `data/metadata/imagenes.csv` registra, por aparición, documento, página, sección, caja,
   clase, texto de la capa del PDF dentro de la imagen, texto OCR y tabla asociada. Si la imagen
   está dentro de una celda (pictogramas de primeros auxilios), solo se reescribe el enlace y la
   fila de la tabla queda intacta.
6. **Asociación imagen-tabla solo con evidencia**: una imagen se asocia a una tabla únicamente
   si al menos el 50 % de su área cae dentro del área de la tabla en la misma página. En los
   demás casos no se asocia.
7. **Texto dentro de las imágenes**: Docling guarda como "hijos" de una imagen las palabras de la
   capa de texto del PDF que caen en su área, pero no las exporta. Pasa con los rótulos de los
   pictogramas de EPP ("Protección obligatoria de las vías respiratorias") y con tablas que el
   modelo de diseño toma por imagen (protección respiratoria de FDS 55). Ese texto se agrega
   debajo del enlace como "Texto del PDF en el área de la imagen (capa de texto, no OCR)".
8. **OCR de imágenes** con RapidOCR (PP-OCRv6 multilingüe, ONNX en CPU): se aplica a cada imagen
   única de 40 px o más; se descartan textos con confianza menor a 0,7 y los caracteres chinos
   que el modelo "lee" a veces sobre los pictogramas. El resultado queda en `imagenes.csv`; en el
   Markdown se usa solo si la imagen no tiene texto en la capa del PDF, marcado como "puede
   contener errores".
9. **Tablas con filas fusionadas** (`tablas_fix.py`): en tablas sin líneas entre filas Docling
   junta todas las filas en una. Se reconstruyen con las palabras del PDF y su posición, usando
   la columna con menos renglones como guía de filas; solo se acepta si las palabras son
   exactamente las mismas que las de la tabla original y ninguna fila queda incompleta. Las que
   no pasan esa validación se dejan como estaban y se registran.
10. **Títulos de sección dentro de un párrafo**: en FDS 100 y 101 la marca de agua "COPIA
    CONTROLADA" hace que Docling junte cada página en un solo párrafo. Los títulos "SECCIÓN N:
    TÍTULO" que quedan en medio del texto se pasan a su propio renglón; solo se agregan saltos de
    línea.

### Límites con pictogramas y elementos no textuales

- El OCR no interpreta pictogramas: un rombo NFPA o un pictograma SGA no tienen texto, o tienen
  solo números o signos sueltos (por ejemplo "!"). El significado de un pictograma no se infiere
  de la imagen: se toma de las indicaciones H y de la clasificación escritas en la sección 2.
- En FDS 1 el OCR de la tabla de riesgos omitió el "1" de salud; el dato sí está en el texto.
- El clasificador de figuras se equivoca a veces (un logo clasificado como "gráfico"); la clase
  se guarda con su confianza y no se usa para decidir contenido.

## Procedencia, versión y secciones

`rag_fds/inventory.py` genera `data/metadata/inventario.csv`, con una fila por ficha:

- **Procedencia**: carpeta de origen (año), nombre original y normalizado del archivo (decodifica
  escapes como `#U00ed` a í y normaliza a NFC), SHA-256, páginas y metadatos del PDF.
- **Fabricante**: reglas explícitas sobre el texto de las dos primeras páginas (razón social,
  correo o dominio web), guardando el texto y la página que lo respaldan. Ejemplos: CORLANC S.A.S.
  y corona.com.co → Corona; Compañía Global de Pinturas S.A.S. con correo @akzonobel.com →
  AkzoNobel; Química Cosmos S.A. con pintuland.com.co → Pintuland. También se guarda la razón
  social tal como aparece (`proveedor_segun_ficha`).
- **Producto, versión y fechas** (emisión, revisión, actualización), cada dato con su evidencia.
  Si la ficha no indica versión, se deja vacía y se marca en `version_estado`; nunca se completa
  por inferencia.
- **Correcciones manuales** en `data/metadata/correcciones_inventario.csv`, cada una con página,
  evidencia y motivo.

**Secciones (16 del SGA)**: `rag_fds/sections.py` reconoce tres formas de encabezado: "SECCIÓN N:
TÍTULO" (SGA), "N. TÍTULO" en mayúsculas (fichas antiguas de Tecnidecor con orden ANSI, donde la
2 es Composición y la 3 es Riesgos; ahí la sección canónica se decide por el título) y solo el
título, sin número, aceptado únicamente si es la sección siguiente a la actual. Para cada ficha y
sección, `data/metadata/secciones.csv` y `secciones_matriz.csv` dan el estado:

- **presente**: el encabezado está en el Markdown;
- **fallo de extracción**: el encabezado está en la capa de texto del PDF pero no en el Markdown;
- **ausente en el original**: no está ni en el PDF ni en el Markdown;
- **revisar**: hay una línea con el número de la sección pero un título no reconocido.

**Verificación contra el PDF** (`rag_fds/verify.py`): por cada ficha se elige una tabla con datos
(prioridad: composición, propiedades, exposición, transporte, toxicología) y se compara palabra
por palabra con la capa de texto del PDF dentro del área de la tabla. Además se guarda un recorte
de la página para revisión visual (`data/verificacion/recortes/`, reporte en
`verificacion_tablas.md`). Los errores encontrados, automáticos y de revisión manual contra el PDF
renderizado, quedan en `data/verificacion/errores_extraccion.csv` y `revision_manual.csv`.

Dos controles más cubren todo el corpus:

- **Cobertura por página** (`data/metadata/cobertura_paginas.csv`): proporción de las palabras de
  la capa de texto del PDF que están en el Markdown de esa página (cobertura) y al revés
  (precisión), más los números con punto ("14.3") que están en el PDF y faltan en el Markdown.
  La cobertura sola no detecta la pérdida de un número de subsección.
- **Evidencias de la evaluación** (`data/verificacion/evidencias_evaluacion.csv`): cada texto de
  evidencia de `eval/preguntas.json` se busca en el Markdown, en su página y en su sección.
  Solo se lee el Markdown; las referencias no pasan al índice.

Resultado de la última extracción: 101 de 101 tablas verificadas sin diferencias, 1616 de 1616
secciones presentes, cobertura por página mínima 0,915 y media 0,992, 2 números con punto
faltantes y las 14 evidencias de evaluación en su página y sección.

## Chunking, embeddings y recuperación

| Parámetro | Valor |
|---|---|
| Unidad | sección de la FDS (1-16), más 0 = encabezado de la ficha y 17 = encabezados y pies de página |
| Tamaño máximo | 1500 caracteres por fragmento (sin contar la cabecera) |
| Solapamiento | 200 caracteres (últimos bloques completos del fragmento anterior de la misma sección) |
| Bloques indivisibles | párrafos y tablas; una tabla larga se parte por filas repitiendo el encabezado |
| Cabecera de cada fragmento | fabricante, producto, versión, fecha, documento y sección |
| Metadatos | doc_id, fabricante, producto, versión, fecha, archivo, carpeta, sección, páginas, imágenes |
| Embeddings | `bge-m3` (Ollama), guardados en Chroma |
| Recuperación | híbrida: coseno entre embeddings + BM25 (k1 = 1,2, b = 0,75), combinados con fusión por rangos recíprocos (RRF, k = 60) |
| top-k | 3 por defecto, porque respondió mejor que 6 en la evaluación (ver [Resultados](#resultados)) |

Un fragmento nunca mezcla dos secciones ni dos fichas. La cabecera ayuda a distinguir productos
y versiones tanto al recuperar como al generar.

**Por qué recuperación híbrida**: con solo embeddings, las preguntas de desarrollo D02 (densidad
del SIGMADUR) y D05 (materiales incompatibles del DIMETCOTE) recuperaban 6 fragmentos de la ficha
correcta pero ninguno de la sección con el dato. Dentro de una ficha todos los fragmentos
comparten la cabecera (fabricante, producto, versión), y una palabra clave de la pregunta
("densidad", "pH") pesa poco frente a ella. BM25 cuenta coincidencias de palabras (sin tildes y sin
palabras vacías) y sí la tiene en cuenta. RRF suma 1/(60 + puesto) de cada orden, así que no hay
que calibrar las dos escalas. La similitud coseno se calcula de forma exacta sobre los embeddings
guardados en Chroma: con 3583 fragmentos toma milisegundos.

**Distinción de fabricante, producto y versión**: antes de buscar, `rag.py` compara la pregunta
con los nombres de producto del inventario (según la ficha y según el archivo, sin las marcas,
ponderando las palabras poco frecuentes) y con los números "FDS N". Una palabra del nombre que en
todo el corpus solo aparece en las fichas de ese producto ("Vinicryl", "13883") basta para
reconocerlo. Si encuentra un producto, filtra por sus fichas (todas las versiones juntas; si la
pregunta menciona un año, se queda con la versión de ese año). Si encuentra dos o más productos
distintos, cada uno se busca por separado y los k fragmentos se reparten entre ellos, intercalados.
Si solo encuentra un fabricante, filtra por fabricante. Las palabras que ya sirvieron para escoger
las fichas (producto, marca, año) no se usan en BM25. Este procedimiento es igual en las dos
configuraciones evaluadas.

## Generación, citas y evidencia insuficiente

- Generador `llama3.2` (3B) con temperatura 0, semilla 42 y contexto de 8192 tokens.
- El prompt (`PROMPT_SISTEMA` en `rag.py`) exige usar solo los fragmentos, citar `[n]` después de
  cada afirmación, no mezclar productos ni versiones, copiar cifras y códigos tal como aparecen y
  responder exactamente "No hay evidencia suficiente en las FDS disponibles para responder esta
  pregunta." cuando falte la información. La regla de citar se repite al final del mensaje porque
  un modelo de 3B sigue mejor la última instrucción.
- **Revisión de citas**: se avisa de las citas a fragmentos inexistentes. Cada oración se compara
  con los fragmentos por coincidencia de palabras y cifras. Si no tiene cita, recibe la del
  fragmento que contiene el 60 % o más de sus palabras. Si tiene cita, pero otro fragmento cumple
  ese 60 % y supera en 0,2 o más al citado, se agrega también la de ese fragmento. El modelo de 3B
  suele citar el fragmento cuya cabecera tiene la versión (sección 0 o 1) y no el que tiene el
  dato. Las citas del modelo nunca se quitan, cada cita agregada queda en `advertencias` y la
  respuesta original del modelo se guarda aparte (`response_modelo`).
- Si el modelo repite el formato del ejemplo ("Pregunta: … Respuesta: …"), se quita ese eco.
- **Fuentes**: cada cita se traduce a fabricante, producto, documento y archivo, versión, fecha,
  páginas y sección.
- **Evidencia insuficiente**: además de la instrucción al modelo, si la mayor similitud coseno
  entre los candidatos es menor que 0,40, se responde que no hay evidencia sin llamar al
  generador. Una respuesta se marca como "insuficiente" si usa esa frase y no cita ningún
  fragmento válido. En las preguntas de desarrollo el umbral no se activó: las dos sin respuesta
  tuvieron 0,56 y 0,60, y fue el generador el que respondió que no había evidencia. El umbral queda
  como protección para preguntas ajenas a las FDS.

## Evaluación con RAGAS

**Preguntas** (`eval/preguntas.json`): 12 preguntas escritas a mano.

| Tipo | Preguntas | Secciones |
|---|---|---|
| Directas | P01 a P06 | 9, 3, 14, 4, 7, 8 |
| Integradoras | P07 (dos versiones del mismo producto), P08 (dos productos), P09 (dos secciones de una ficha), P10 (dos fabricantes) | 2, 9, 2 y 14, 14 |
| Sin respuesta | P11 (dato que la ficha no trae), P12 (producto que no está en el corpus) | — |

Cubren 7 secciones (2, 3, 4, 7, 8, 9 y 14). Cada referencia se verificó contra la página
renderizada del PDF y guarda la evidencia textual, el documento, la página y la sección. Las
referencias solo las lee el evaluador: no están en el índice ni en el prompt del generador. Para
ajustar la recuperación, el prompt y el umbral se usaron otras preguntas
(`eval/preguntas_desarrollo.json`, D01 a D09, de otros productos).

**Dos corridas de generación** (se declara porque toca la independencia de la evaluación):

- *Corrida 1* (`eval/resultados/corrida1/`): al revisarla, P10 respondía "no hay evidencia" porque
  la detección aplicaba el año de la pregunta ("versión de 2025", que se refería a un producto) a
  todas las fichas detectadas y descartaba el otro producto. Era un error de lógica, no un
  parámetro.
- Se escribió la pregunta de desarrollo D09 (otros dos productos, con el año solo para uno), se
  comprobó que reproducía el error, y se corrigió para que el año solo escoja entre versiones del
  mismo producto. Las otras 8 preguntas de desarrollo dieron lo mismo antes y después.
- *Corrida 2* (`eval/resultados/`): la que se puntúa. No se cambió nada más: P08, que también
  falla en las dos corridas (el generador tiene los dos datos y responde que no hay evidencia para
  comparar), no se usó para ajustar el prompt.

**Configuraciones**: top-k 3 y top-k 6. Corpus, fragmentos, embeddings, filtros, generador,
prompt y evaluador son los mismos en las dos.

**Métricas** (sobre las 10 preguntas con respuesta):

- *Faithfulness* (fidelidad): proporción de afirmaciones de la respuesta que se pueden deducir de
  los fragmentos recuperados.
- *Context Precision* con referencia: si los fragmentos que sirven para responder quedan en los
  primeros puestos.
- *Context Recall* (cobertura): proporción de las afirmaciones de la referencia que aparecen en
  los fragmentos recuperados.

P11 y P12 se revisan a mano: la respuesta correcta es decir que no hay evidencia, sin inventar.

**Registro**: las respuestas de cada configuración están en `eval/resultados/respuestas_k3.json`
y `respuestas_k6.json`. Lo del evaluador está en `eval/resultados/evaluador_gemini-3.5-flash-lite/`:
`resultados.json` y `resultados.csv` guardan, por pregunta y configuración, `user_input`,
`retrieved_contexts` en orden, `response`, `reference`, las fuentes citadas y la puntuación de cada
métrica; `puntuaciones.json` guarda además los intentos, los segundos y los errores de cada
cálculo. Si el evaluador falla después de los reintentos, la métrica queda como fallo (valor nulo
y mensaje de error), no como cero, y los promedios indican sobre cuántos casos válidos se
calcularon. Si un fallo se repite después con éxito, el fallo anterior se conserva en
`corridas_fallidas_previas`.

**Prueba previa**: `evaluate prueba` corre el evaluador sobre 2 casos (P01 y P07 con top-k 3)
antes de la evaluación completa y guarda en `prueba_evaluador.json` los puntajes y en
`evaluador_config.json` el modelo, la versión que reporta la API, los parámetros y las versiones
de las librerías.

**Diagnóstico de Context Precision**: RAGAS solo guarda el promedio de esta métrica. `evaluate
diagnostico` repite la misma consulta (mismo prompt, modelo y temperatura) fragmento por
fragmento y guarda en `diagnostico_context_precision.json` el veredicto (útil o no) y la razón que
da el evaluador. No cambia los puntajes.

### Evaluador de RAGAS (Gemini)

El enunciado pide un evaluador local y sin APIs pagas. **El profesor autorizó usar la API de
Gemini como evaluador**; se usa el plan gratuito, que no cobra. El generador y los embeddings
siguen siendo locales, y el evaluador es un modelo distinto del generador, así que el modelo no
califica sus propias respuestas.

| Parámetro | Valor |
|---|---|
| Modelo | `gemini-3.5-flash-lite` (la API reporta la versión `gemini-3.5-flash-lite`); se puede cambiar con la variable `RAGAS_MODELO` |
| Temperatura | 0 |
| Concurrencia | una llamada a la vez, con 6 s de pausa entre llamadas |
| Reintentos | hasta 6, con espera exponencial de 10 s a 120 s; límite de 900 s por métrica y caso. La cuota diaria agotada no se reintenta: detiene la corrida |
| Cliente | `google-genai` con `ragas.llms.llm_factory(provider="google")` |
| Métricas | `ragas.metrics.collections`: `Faithfulness`, `ContextPrecision` (con referencia), `ContextRecall` |

**Llamadas por caso**: Faithfulness hace 2 (extrae las afirmaciones de la respuesta y luego las
contrasta con los fragmentos), Context Precision hace una por fragmento (3 con k3, 6 con k6) y
Context Recall hace 1. La evaluación completa son unas 150 llamadas, sin contar reintentos.

**Por qué `gemini-3.5-flash-lite`** (en el orden en que se probó):

1. `gemini-2.5-flash`, el modelo previsto, aparece en la lista de modelos de la cuenta, pero la
   API lo rechaza ("no longer available to new users").
2. `gemini-3.8-flash` respondía con error 503 por alta demanda o tardaba 280 s por llamada.
3. `gemini-3.5-flash` respondió en 22 a 51 s por llamada y pasó la prueba de 2 casos. Pero el plan
   gratuito le da **20 solicitudes por día**, y la puntuación completa se cortó después de 2
   puntajes. A ese ritmo, las 150 llamadas tomaban unos 8 días. Lo que alcanzó a hacer quedó en
   `eval/resultados/evaluador_gemini-3.5-flash/`, con los fallos por cuota registrados como fallos,
   y no se usa en los resultados.
4. `gemini-3.5-flash-lite` tiene su propia cuota, que no se agotó en toda la evaluación (más de
   200 llamadas en el mismo día), y responde en pocos segundos por métrica. Es una versión más
   pequeña del mismo modelo, así que se comparó con `gemini-3.5-flash` en la prueba de 2 casos:

| Caso (k3) | Métrica | `gemini-3.5-flash` | `gemini-3.5-flash-lite` |
|---|---|---|---|
| P01 | Faithfulness | 1,00 | 1,00 |
| P01 | Context Precision | 1,00 | 1,00 |
| P01 | Context Recall | 1,00 | 1,00 |
| P07 | Faithfulness | 0,86 | 0,80 |
| P07 | Context Precision | 0,50 | 0,00 |
| P07 | Context Recall | 1,00 | 1,00 |

Coinciden en 5 de 6. En el Context Precision de P07 los dos se apartan de la revisión manual: los
fragmentos [1] y [2] traen exactamente las indicaciones de cada versión que pide la referencia,
así que el valor esperado es cercano a 1. Flash Lite se aparta más. Por eso esa métrica se
interpreta junto con el diagnóstico por fragmento (ver [Resultados](#resultados)). Todos los
puntajes de los resultados son de un solo evaluador, `gemini-3.5-flash-lite`, para no mezclar
criterios.

Se fija `ragas==0.4.4` porque con 0.4.3 la importación falla
(`ModuleNotFoundError: langchain_community.chat_models.vertexai`) con la versión de
`langchain-community` que resuelve uv (0.4.2).

## Resultados

Corrida 2 de generación, evaluada con `gemini-3.5-flash-lite` y temperatura 0. Los 60 puntajes
(10 preguntas × 3 métricas × 2 configuraciones) son válidos. Un cálculo, el Context Precision de
P08 con k6, falló los 6 intentos por error 503 (servicio saturado): quedó registrado como fallo
y salió bien al repetir `puntuar`, que conserva el fallo anterior. La puntuación tomó 28 minutos,
con 20 reintentos aparte de ese caso. Archivos: `eval/resultados/evaluador_gemini-3.5-flash-lite/` (`resumen.md`,
`resultados.csv`, `promedios.json`, `diagnostico_context_precision.json`).

| Métrica (promedio de 10 preguntas) | top-k 3 | top-k 6 |
|---|---|---|
| Faithfulness | 0,88 | 0,75 |
| Context Precision | 0,53 | 0,53 |
| Context Recall | 0,98 | 1,00 |

Por tipo de pregunta:

| Métrica | Directas, k3 | Directas, k6 | Integradoras, k3 | Integradoras, k6 |
|---|---|---|---|---|
| Faithfulness | 1,00 | 1,00 | 0,70 | 0,38 |
| Context Precision | 0,89 | 0,89 | 0,00 | 0,00 |
| Context Recall | 0,96 | 1,00 | 1,00 | 1,00 |

Por pregunta (F = Faithfulness, CP = Context Precision, CR = Context Recall):

| Pregunta | k3: F · CP · CR | k6: F · CP · CR | Revisión manual |
|---|---|---|---|
| P01 directa | 1,00 · 1,00 · 1,00 | 1,00 · 1,00 · 1,00 | correcta |
| P02 directa | 1,00 · 0,33 · 1,00 | 1,00 · 0,33 · 1,00 | correcta |
| P03 directa | 1,00 · 1,00 · 0,75 | 1,00 · 1,00 · 1,00 | correcta |
| P04 directa | 1,00 · 1,00 · 1,00 | 1,00 · 1,00 · 1,00 | k3 incompleta; k6 correcta |
| P05 directa | 1,00 · 1,00 · 1,00 | 1,00 · 1,00 · 1,00 | correcta |
| P06 directa | 1,00 · 1,00 · 1,00 | 1,00 · 1,00 · 1,00 | incompleta |
| P07 dos versiones | 0,80 · 0,00 · 1,00 | 0,67 · 0,00 · 1,00 | incorrecta |
| P08 dos productos | 0,00 · 0,00 · 1,00 | 0,00 · 0,00 · 1,00 | incorrecta (falso "sin evidencia") |
| P09 dos secciones | 1,00 · 0,00 · 1,00 | 0,33 · 0,00 · 1,00 | k3 incompleta; k6 incorrecta |
| P10 dos fabricantes | 1,00 · 0,00 · 1,00 | 0,50 · 0,00 · 1,00 | k3 correcta; k6 incorrecta |
| P11 sin respuesta | no aplica | no aplica | correcta ("no hay evidencia") |
| P12 sin respuesta | no aplica | no aplica | correcta ("no hay evidencia") |

P07 a P12 están revisadas en detalle, con los fragmentos y el PDF, en `eval/revision_manual.md`.
P07 a P10 son las de menor puntaje en las dos configuraciones. P01 a P06 se revisaron comparando
la respuesta con la referencia.

**top-k 3 frente a top-k 6**

- *Recuperación*: con k3 ya llega casi toda la evidencia. Context Recall solo cambia en P03
  (0,75 → 1,00), y Context Precision no cambia, porque k6 agrega fragmentos detrás de los que ya
  estaban.
- *Generación*: Faithfulness baja de 0,88 a 0,75, y toda la caída está en las integradoras (0,70
  → 0,38); en las directas es 1,00 con los dos valores. Con más fragmentos, el modelo de 3B mezcla
  datos de otro producto, versión o sección. La revisión manual lo confirma: con k6, P10 pasa de
  correcta a incorrecta y P09 de incompleta a incorrecta; solo P04 mejora (de incompleta a
  correcta).
- Con este generador, **k3 da mejores respuestas**; k6 solo agrega una cobertura que ya era casi
  completa. Por eso 3 es el valor por defecto de la demo.

**Lo que miden y lo que no miden las métricas**

- *Faithfulness no mide si la respuesta está completa*: P04 con k3, P06 y P09 con k3 sacan 1,00
  y les falta parte de la referencia. Sí detecta las afirmaciones falsas: P07 baja por atribuirle
  la H317 a la versión de 2021, y P08 saca 0 porque dice que no hay evidencia teniendo los datos.
- *Context Precision de las integradoras (0,00) no refleja la recuperación*. El diagnóstico por
  fragmento muestra el motivo: el evaluador marca como "no útil" todo fragmento que trae solo una
  de las dos partes de la referencia. Por ejemplo, el fragmento de la versión de 2019 en P07 "no
  contiene información sobre la ficha de 2021". En esos mismos casos Context Recall es 1,00, y la
  revisión manual muestra que los fragmentos necesarios están en los puestos 1 y 2 (P07, P09 y
  P10) o 2 y 3 (P08) con k3. Si se cuenta como útil un fragmento que trae una parte de la
  referencia, el Context Precision con k3 sería 1,00 en P07, P09 y P10 y 0,58 en P08. Ese cálculo
  es manual y se informa aparte: no reemplaza los puntajes de RAGAS.
- En las directas, Context Precision (0,89) sí señala un problema real de orden: en P02 el
  fragmento de la sección 3, que tiene las sustancias y los números CAS, quedó en el puesto 3,
  detrás de la sección 16 y del encabezado.

**Estabilidad del evaluador**: P01 y P07, puntuados en la prueba y otra vez en la corrida
completa, dieron los mismos 6 valores. Pero no es del todo determinista: en el diagnóstico, el
primer fragmento de P09 con k6 salió "útil", lo que daría un Context Precision de 1,00 en vez del
0,00 registrado. La razón que dio el evaluador ("no menciona la indicación de peligro H410")
contradice su propio veredicto.

**Preguntas sin respuesta**: P11 y P12 se respondieron bien en las dos configuraciones, sin
inventar datos, aunque el umbral de similitud no se activó (0,74 y 0,57). La protección vino del
prompt.

## Hardware, consumo y costo

Medido con `rag_fds.recursos` (`docs/recursos.json`).

| Concepto | Valor |
|---|---|
| Equipo | Portátil con Intel Core i5-13420H (8 núcleos, 12 hilos), 15,6 GB de RAM, NVIDIA GeForce RTX 3050 Laptop de 6 GB y Windows 11 |
| Extracción (CPU) | 101 fichas y 1134 páginas en 64 min (3,4 s por página): Docling 2804 s; imágenes y OCR 1034 s; RAM máxima del proceso 1,8 GB |
| Índice | 3583 fragmentos; embeddings en 165 s |
| Modelos en la GPU (Ollama 0.40.16) | llama3.2 3B: 2,9 GB de VRAM; bge-m3: 0,66 GB |
| Consulta | recuperación 0,02 s (mediana); generación 1,2 s con k3 y 1,6 s con k6 (mediana), 5,3 s como máximo |
| Disco | PDF 40,6 MB; Markdown e imágenes 16,4 MB; JSON de Docling 4,3 MB; Chroma 68,6 MB; entorno de Python 1,5 GB |
| Evaluación con RAGAS | 60 puntajes en 28 min con `gemini-3.5-flash-lite` (mediana por cálculo: Faithfulness 3,6 s, Context Precision 8,3 s, Context Recall 1,8 s) |
| Costo | 0: todo es open source y corre en el equipo; la evaluación usa el plan gratuito de Gemini |

**Desempeño frente a costo**:
- El generador de 3B y los embeddings caben juntos en los 6 GB de la GPU (3,6 GB) y responden en 1 a 2 s. El precio es la calidad de las comparaciones y de las citas (ver limitaciones).
- La extracción es lo más costoso (64 min en CPU), pero se hace una sola vez por ficha.
- No se usa OCR de página: costaba 65 de 76 s por ficha y no aportaba texto, porque todas las fichas tienen capa de texto.
- Pasar de k3 a k6 suma unos 0,3 s de generación (mediana).

## Limitaciones y mitigaciones

### Extracción

Cada caso está en `data/verificacion/revision_manual.csv` con su página y la evidencia del PDF.

| Limitación | Alcance | Mitigación |
|---|---|---|
| Tablas sin líneas entre filas: Docling junta las filas en una | 46 tablas; 26 reconstruidas | Las otras 20 no pasan la validación y quedan como estaban: el texto está, pero sin la correspondencia entre columnas de cada fila. Registradas en `tablas_reconstruidas.csv`. |
| TableFormer llega a su límite de pasos y Docling descarta la estructura | 3 tablas de valores límite de la sección 8 (FDS 10 p4, FDS 54 p4, FDS 94 p5) | El texto queda en una sola celda, en orden de lectura. Donde falta un valor (TLV-STEL vacío), la sustancia siguiente queda al lado de la etiqueta y la asociación es ambigua. |
| Celdas desplazadas o repetidas | FDS 23 p12 (UN1263 de la columna TDG en la columna DOT); celdas combinadas repetidas (FDS 37, FDS 71) | En los fragmentos del índice las celdas idénticas seguidas se unen en una. El Markdown queda como lo exporta Docling. |
| Etiqueta y valor que Docling ya juntó en un bloque | FDS 83 p2, FDS 47 p2, FDS 23 p1 y p2; sección 14 de la plantilla de Pintuco ("3 Etiquetas:", "Riesgos ambientales: 14.5 Sí") | El reemparejado no separa un bloque ya unido. Los valores están en el Markdown. El producto de FDS 23 se corrigió a mano en el inventario. |
| Subtítulo tomado como fila de tabla | FDS 50 p7 (10.5), FDS 58 p7 (9.2) | El número se pierde; el título y los valores están. Lo detecta el control de números con punto. |
| Pictogramas sin texto | Todas las fichas | No se interpreta la imagen: el significado se toma de las indicaciones H y de la clasificación de la sección 2. |
| Fichas sin número de versión | 17 fichas | Se marca "no indicada en el original"; no se completa por inferencia. Fabricante, producto y fecha siguen distinguiendo las fichas. |
| Caída nativa intermitente de `docling_parse` en Windows | 1 de las 2 últimas corridas completas | Repetir el comando retoma desde la ficha pendiente. |
| Producto mal leído por etiquetas fusionadas ("SDS code") | FDS 79, 96 y 98 (corregido) | "SDS code" se trata como etiqueta y el producto se toma del renglón que sigue al título de la ficha. |

### Recuperación y generación

| Limitación | Ejemplo | Mitigación |
|---|---|---|
| El modelo de 3B cita el fragmento con la versión y no el que tiene el dato | D01, D05; P05 | Revisión de citas por coincidencia de palabras: agrega la cita correcta sin quitar la del modelo, y lo deja en `advertencias`. |
| En comparaciones el modelo de 3B puede mezclar productos o decir que no hay evidencia teniendo los datos | D09 (atribuye "polvo ABC" al producto equivocado); P08 | Ninguna automática. La revisión de citas lo deja visible: la cita agregada apunta al fragmento del otro producto. |
| Un nombre de producto contenido en otro ("PINTURA LAVABLE" dentro de "PINTURA LAVABLE ANTIBACTERIAL") detecta los dos | P06 | Los k fragmentos se reparten entre los dos; la respuesta de P06 usó la ficha correcta. No se corrigió para no ajustar con preguntas de evaluación. |
| Más contexto no siempre ayuda al modelo de 3B | P10 con k6 atribuye la clase 3 a un producto de clase 9; con k3 responde bien. En las integradoras, Faithfulness baja de 0,70 a 0,38 con k6 | Se informa por configuración en los resultados. Con este generador conviene k3, y es el valor por defecto. |
| Preguntas que juntan dos secciones: el modelo responde solo una o las mezcla | P09 | Ninguna automática; queda visible en la revisión manual. |

### Evaluación

| Limitación | Ejemplo | Mitigación |
|---|---|---|
| Context Precision con `gemini-3.5-flash-lite` exige que cada fragmento alcance solo para toda la referencia | P07 a P10 dan 0,00 aunque su Context Recall es 1,00 | Diagnóstico por fragmento con la razón del evaluador, y cálculo manual aparte para las integradoras con k3. Los puntajes de RAGAS no se modifican. |
| Faithfulness no mide si la respuesta está completa | P04 con k3, P06 y P09 con k3: 1,00 e incompletas | Revisión manual de las respuestas, con la clasificación correcta, incompleta o incorrecta. |
| El evaluador no es del todo determinista con temperatura 0 | En el diagnóstico, el primer fragmento de P09 con k6 cambió de veredicto | La prueba y la corrida completa dieron los mismos 6 valores en P01 y P07; el cambio se informa. |
| Evaluador más pequeño que el probado primero | `gemini-3.5-flash` tenía 20 solicitudes por día en el plan gratuito | Comparación de los dos en la prueba de 2 casos (5 de 6 iguales). Un solo evaluador para todos los puntajes. |
| Los errores 503 (servicio saturado) del plan gratuito | 20 reintentos y 1 fallo en la corrida | Reintentos con espera exponencial; el fallo se repitió y quedó registrado. |

## Estructura del repositorio

```
rag_fds/              código, un módulo por paso
  config.py           rutas y parámetros
  extract.py          PDF → Markdown (Docling + texto de imágenes + OCR)
  kv_fix.py           reemparejado etiqueta-valor
  tablas_fix.py       reconstrucción de tablas con filas fusionadas
  pdftexto.py         palabras y renglones de la capa de texto del PDF (pypdfium2)
  sections.py         detector de las 16 secciones
  inventory.py        procedencia, versión, secciones y cobertura por página
  verify.py           verificación de tablas contra el PDF y registro de errores
  index.py            fragmentos, embeddings y Chroma
  rag.py              detección de producto, recuperación, generación y citas
  ask.py, web.py      demostración por consola y por navegador
  evaluate.py         evaluación con RAGAS
  recursos.py         hardware y consumo
data/
  fds_pdf/            PDF originales (no se versionan)
  md/                 Markdown, imágenes y tablas por ficha
  docling_json/       documento Docling completo de cada ficha (json.gz)
  metadata/           inventario, secciones, cobertura, imágenes, tablas reconstruidas, log
  verificacion/       verificación de tablas, recortes, errores y revisión manual
  index/              fragmentos.jsonl y resumen del índice
  chroma/             base vectorial (no se versiona; se regenera con rag_fds.index)
eval/
  preguntas.json, preguntas_desarrollo.json
  revision_manual.md  revisión manual de P07 a P12
  resultados/         respuestas de la corrida 2 (respuestas_k3.json, respuestas_k6.json),
                      desarrollo.json y corrida1/
    evaluador_gemini-3.5-flash-lite/  prueba, puntuaciones, diagnóstico y resumen (los resultados)
    evaluador_gemini-3.5-flash/       intento cortado por la cuota diaria (no se usa)
docs/                 informe y recursos.json
```
