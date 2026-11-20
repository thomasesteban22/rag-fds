# RAG local sobre Fichas de Datos de Seguridad de pinturas

Procesamiento de Lenguaje Natural (PCIA5011), 2026-2, Universidad Sergio Arboleda.
Integrantes: Thomas Ariza, Julian Cristancho y Juan Marquez. Instalación, ejecución y detalle:
`README.md` del repositorio.

## 1. Corpus y objetivo

101 FDS de 11 fabricantes (Pintuco 21, Corona 17, Sika 14, AkzoNobel/International 12, Tecnidecor
10, Pintuland 9, Invesa 6, PPG 6, Axalta 3, Bler 2, Graphenstone 1), de 2018 a 2025, con 1134
páginas. El sistema responde citando fabricante, producto, versión, documento, página y sección, y
dice cuando la evidencia es insuficiente. Todo corre en local salvo el evaluador de RAGAS.

## 2. Extracción PDF → Markdown

- **Herramienta:** Docling 2.150 en CPU, con TableFormer en modo preciso y sin OCR de página:
  todas las fichas tienen capa de texto, y ese OCR costaba 65 de 76 s por ficha sin aportar texto.
  Cada página empieza con `<!-- pagina: N -->`, así que cada fragmento conserva su página.
- **Imágenes:** 2438 apariciones (1278 distintas) con ruta relativa, documento, página y sección.
  Su texto sale de la capa del PDF (278) o de RapidOCR, marcado "puede contener errores". El OCR no
  interpreta pictogramas: su significado se toma de las indicaciones H de la sección 2.
- **Correcciones propias:** reemparejado etiqueta-valor por posición vertical; numeración de
  subsección; 26 de 46 tablas con filas fusionadas reconstruidas y validadas palabra por palabra.
- **Verificación:** una tabla por ficha contra el PDF, 101 de 101 sin diferencias; 1616 de 1616
  secciones, separando "ausente en el original" de "fallo de extracción"; cobertura de palabras por
  página mínima 0,915 y media 0,992; 27 casos revisados a mano con su evidencia.
- **Errores que quedan:** 20 tablas con filas fusionadas sin reconstruir, 3 cuya estructura
  descartó TableFormer y bloques etiqueta-valor que Docling ya había unido.
- **Procedencia:** versión y fechas con la evidencia y la página; las 17 fichas sin versión quedan
  como "no indicada en el original".

## 3. RAG con modelos locales

- **Chunking por sección** (hasta 1500 caracteres, 200 de solapamiento, sin partir tablas), con una
  cabecera de fabricante, producto, versión, fecha, documento y sección: 3583 fragmentos.
  Embeddings bge-m3 (Ollama) en Chroma; top-k 3 por defecto.
- **Distinción de producto, fabricante y versión:** se detectan en la pregunta comparando con el
  inventario y se filtra por esas fichas; con dos productos, cada uno se busca por separado.
- **Recuperación híbrida:** coseno y BM25 combinados por rangos (RRF). Solo con embeddings, la
  cabecera común pesaba más que la palabra clave ("densidad", "pH").
- **Generador:** llama3.2 3B con temperatura 0. Cita [n] tras cada afirmación y responde "No hay
  evidencia suficiente…" si falta información. Una revisión de citas por coincidencia de palabras
  agrega la cita del fragmento que respalda cada oración, sin quitar la del modelo.

## 4. Evaluación con RAGAS

**Preguntas:** 12 escritas a mano: 6 directas, 4 integradoras (dos versiones, dos productos, dos
secciones, dos fabricantes) y 2 sin respuesta, sobre 7 secciones. Las referencias se verificaron
contra el PDF y no están en el índice ni en el prompt. Los ajustes se hicieron con 9 preguntas de
desarrollo distintas. La primera corrida mostró un error de lógica (el año de la pregunta
descartaba el otro producto); se reprodujo con una pregunta de desarrollo, se corrigió y se puntúa
la corrida 2. Las dos se conservan.

**Evaluador:** Gemini por API gratuita, autorizado por el profesor, con temperatura 0, una llamada
a la vez y reintentos; un fallo se guarda como fallo, no como cero. `gemini-3.5-flash` pasó la
prueba de 2 casos, pero el plan gratuito le da 20 solicitudes diarias y la evaluación necesita unas
150. Se usó `gemini-3.5-flash-lite`, que tiene su propia cuota y coincidió en 5 de los 6 puntajes de
esa prueba. Los 60 puntajes son válidos; un fallo por error 503 se repitió con éxito y quedó
registrado.

| Métrica | k3 | k6 | Directas (k3 / k6) | Integradoras (k3 / k6) |
|---|---|---|---|---|
| Faithfulness | 0,88 | 0,75 | 1,00 / 1,00 | 0,70 / 0,38 |
| Context Precision | 0,53 | 0,53 | 0,89 / 0,89 | 0,00 / 0,00 |
| Context Recall | 0,98 | 1,00 | 0,96 / 1,00 | 1,00 / 1,00 |

**Análisis:**
- **k3 frente a k6:** con k3 ya llega casi toda la evidencia (Context Recall 0,98). k6 baja la
  fidelidad de las integradoras de 0,70 a 0,38, porque el modelo de 3B mezcla productos, versiones o
  secciones. La revisión manual coincide: con k6, P10 pasa de correcta a incorrecta y P09 de
  incompleta a incorrecta; solo P04 mejora. Con este generador conviene k3, el valor por defecto.
- **Faithfulness** detecta las afirmaciones falsas (P07, P08), pero no las respuestas incompletas:
  P04 con k3, P06 y P09 con k3 sacan 1,00.
- **Context Precision de 0 en las integradoras es un sesgo del evaluador.** El diagnóstico por
  fragmento muestra que descarta todo fragmento que trae solo una de las dos partes de la
  referencia, aunque en esos casos Context Recall es 1,00 y los fragmentos necesarios están en los
  puestos 1 y 2 (2 y 3 en P08). Contando como útil el fragmento que trae una parte, con k3 sería
  1,00 en P07, P09 y P10 y 0,58 en P08 (cálculo manual, aparte de RAGAS). En las directas sí señala
  un problema real: en P02 la sección 3 quedó en el tercer puesto.
- **Estabilidad:** P01 y P07, puntuados dos veces, dieron los mismos valores; un veredicto de P09
  (k6) cambió.

**Revisión manual** (`eval/revision_manual.md`): P11 y P12, sin respuesta, se respondieron bien en
las dos configuraciones; en P12 no usó los puntos de inflamación de otras pinturas recuperados. En
P07 a P10, las de menor puntaje, la recuperación trajo la evidencia y falló
el generador: P07 le atribuye a la versión de 2021 una H317 que ya no tiene; P08 responde "sin
evidencia" teniendo los dos datos; P09 responde una sola sección (k3) o las mezcla (k6); P10 con k6
le pasa la clase de transporte de un producto al otro.

## 5. Arquitectura, reproducibilidad y costo

Todo el software es open source (Docling, RapidOCR, Chroma, Ollama, llama3.2, bge-m3, RAGAS) y las
dependencias están fijadas con uv (`uv.lock`, `requirements.txt`). Equipo: portátil con Core
i5-13420H, 15,6 GB de RAM y RTX 3050 de 6 GB.

| Concepto | Valor |
|---|---|
| Extracción | 64 min en CPU (3,4 s por página), 1,8 GB de RAM como máximo |
| Índice y VRAM | 165 s; llama3.2 2,9 GB y bge-m3 0,66 GB |
| Respuesta (mediana) | 1,2 s con k3 y 1,6 s con k6 |
| Evaluación | 60 puntajes en 28 min |
| Costo | 0 |

El modelo de 3B cabe con los embeddings en la GPU y responde en 1 a 2 s, a costa de fallar más en
las preguntas integradoras y en las citas.

## 6. Limitaciones y siguientes pasos

- El generador de 3B falla en comparaciones; un modelo de 7-8B podría mejorarlo, pero no se probó.
- La cuadrícula de la sección 14 de Pintuco y Corona queda desordenada, y coincidió con el error de
  P10.
- El umbral de similitud no separó las preguntas sin respuesta; la protección vino del prompt.
- El evaluador del plan gratuito es un modelo pequeño: su Context Precision no sirve para las
  integradoras, y Faithfulness no mide si la respuesta está completa. Por eso la revisión manual es
  parte de la evaluación.

## 7. Conclusión

El sistema incorpora las 101 FDS con extracción verificada y cita fabricante, producto, versión,
documento, página y sección. Se selecciona top-k 3: la evidencia ya llega con 3 fragmentos, y el
generador de 3B, principal fuente de errores, se equivoca menos con menos contexto. Con 12
preguntas, la comparación es didáctica y no demuestra que k3 sea mejor en general.
