# Revisión manual de respuestas

Corrida 2 (`eval/resultados/respuestas_k3.json` y `respuestas_k6.json`). Cada caso se revisó
leyendo la respuesta, los fragmentos recuperados en su orden y la referencia, y comparando los
valores con la capa de texto del PDF. Se revisan las 2 preguntas sin respuesta y 4 respondibles,
P07 a P10, que son las de menor puntaje de RAGAS en las dos configuraciones (evaluador
`gemini-3.5-flash-lite`; puntajes en `eval/resultados/evaluador_gemini-3.5-flash-lite/`).

Los puntajes se escriben como F (Faithfulness) · CP (Context Precision) · CR (Context Recall).

Clasificación de cada respuesta:

- **correcta**: responde lo pedido, sin datos que no estén en la ficha;
- **incompleta**: lo que dice es cierto, pero falta parte de la referencia;
- **incorrecta**: afirma algo que contradice la ficha o que viene de otro producto o versión.

## Preguntas sin respuesta

### P11. Tiempo de secado al tacto del VINILTEX PRO 650 BLANCO (Pintuco)

- **Verificación de que no hay respuesta**: el producto está en el corpus (FDS045), pero en la
  capa de texto de las 101 fichas no aparecen "secado" ni "al tacto" (0 resultados). Es un dato
  de ficha técnica, no de FDS.
- **Recuperación**: la detección filtró por FDS045. Con k3 recuperó dos fragmentos de la sección 8 y uno de la 7; con
  k6, además uno de la sección 0 y dos de la 9. Similitud máxima 0,74: el umbral de 0,40 no se
  activó.
- **Respuesta (k3 y k6)**: "No hay evidencia suficiente en las FDS disponibles para responder esta
  pregunta."
- **Resultado: correcta en las dos configuraciones.** El generador no completó el dato con
  conocimiento propio, aunque tenía delante la sección 9 del mismo producto.

### P12. Punto de inflamación de la pintura ProMar 200 de Sherwin-Williams

- **Verificación de que no hay respuesta**: "Sherwin" y "ProMar" no aparecen en la capa de texto
  de las 101 fichas (0 resultados).
- **Recuperación**: no se detectó producto ni fabricante, así que se buscó en todo el corpus.
  Se recuperaron fragmentos de otros productos que sí traen puntos de inflamación: la sección 9 de
  FDS006, FDS010 y FDS075 con k3; con k6, además la sección 9 de FDS042 y FDS068 y la sección 5
  de FDS001. Similitud máxima 0,57: el umbral
  no se activó.
- **Respuesta (k3 y k6)**: "No hay evidencia suficiente en las FDS disponibles para responder esta
  pregunta."
- **Resultado: correcta en las dos configuraciones.** Es el caso más riesgoso: los fragmentos
  tenían valores de punto de inflamación de otras pinturas y el generador no los usó.

## Preguntas respondibles

### P07. Cambio de las indicaciones de peligro de la Pintura Primera Mano & Acabado (Corona), 2019 frente a 2021

- **Referencia**: versión 1 (2019): H351 y H317. Versión 3 (2021): solo H351 (Inhalación); la H317
  ya no figura.
- **Recuperación**: correcta. Con k3 y con k6 los fragmentos [1] (FDS029, sección 2) y [2] (FDS043,
  sección 2) contienen las indicaciones de cada versión; en [2] solo está "H351 - Susceptible de
  provocar cáncer (Inhalación)".
- **Respuesta k3**: dice que la versión 3 tiene H351 y H317 y que el cambio fue agregar
  "(Inhalación)". El modelo solo puso "[1][3]" al final; las citas [1] y [2] de las oraciones las agregó la
  revisión automática.
- **Respuesta k6**: también atribuye la H317 a la versión 3. Agrega que en 2021 aparecen P202 y
  P405, lo cual es cierto (están en FDS043 y no en FDS029), aunque son consejos de prudencia, no
  indicaciones de peligro.
- **Resultado: incorrecta en las dos configuraciones.** Error del generador: copió la H317 de
  la versión 1 a la versión 3, es decir, mezcló versiones teniendo la evidencia correcta.
- **RAGAS**: k3 0,80 · 0,00 · 1,00; k6 0,67 · 0,00 · 1,00. Faithfulness baja por la H317 atribuida
  a la versión 3, de acuerdo con la revisión. Context Precision 0,00 no coincide con la revisión:
  según el diagnóstico, el evaluador descartó el fragmento [1] porque "no contiene información
  sobre la ficha de 2021", y el [2] porque solo trae la de 2021. Juzga cada fragmento como si
  tuviera que bastar solo para toda la comparación. `gemini-3.5-flash` dio 0,50 en la prueba.

### P08. Punto de inflamación del TOPCOAT 400 1793 BLANCO frente al TOPCOAT 400 1772 NEGRO (International)

- **Referencia**: blanco 49 °C (120.2 °F) y negro 56 °C (132.8 °F), vaso cerrado; el blanco tiene el
  más bajo.
- **Recuperación**: correcta. Con k3, el fragmento [2] (FDS081, sección 9) trae "Punto de
  inflamación: Vaso cerrado: 56°C (132.8°F)" y el [3] (FDS080, sección 9) "Vaso cerrado: 49°C
  (120.2°F)". Con k6 están los mismos y 3 fragmentos más.
- **Respuesta k3**: "No hay evidencia suficiente … para comparar …, ya que solo se menciona el
  punto de inflamación para el TOPCOAT 400 1793 BLANCO [1] y para el TOPCOAT 400 1772 NEGRO [2]."
  Se contradice: reconoce que los dos datos están y aun así no los da.
- **Respuesta k6**: "No hay evidencia suficiente en las FDS disponibles para responder esta
  pregunta."
- **Resultado: incorrecta en las dos configuraciones** (falso "sin evidencia"). Error del
  generador con preguntas de comparación. Pasó igual en la corrida 1 y no se usó para ajustar el
  prompt.
- **RAGAS**: k3 0,00 · 0,00 · 1,00; k6 0,00 · 0,00 · 1,00. Faithfulness 0 coincide con la
  revisión: decir que no hay evidencia no se deduce de unos fragmentos que sí traen los datos.
  Context Precision 0 no coincide: el evaluador descartó los fragmentos [2] y [3] porque cada uno
  trae solo uno de los dos productos.

### P09. Indicación de peligro ambiental y transporte del EPOXY ZINC PRIMER 10056 (AkzoNobel/International)

- **Referencia**: H410 (peligro acuático agudo y crónico, categoría 1). Transporte: UN3077,
  SUSTANCIA SÓLIDA PELIGROSA PARA EL MEDIO AMBIENTE, N.E.P., clase 9, grupo de embalaje III.
- **Recuperación**: correcta en las dos configuraciones. El filtro escogió FDS083. Con k3 y con k6,
  el fragmento [1] (sección 14) trae el número ONU, la designación, la clase 9 y el grupo III, y el
  [2] (sección 2) trae la clasificación y "H410 - Muy tóxico para los organismos acuáticos, con
  efectos nocivos duraderos". Con k6 se suman fragmentos de las secciones 16 y 11, que no aportan.
- **Respuesta k3**: da las categorías de peligro acuático (agudo y crónico, categoría 1) [2], que
  son ciertas, pero no da la H410 ni nada del transporte.
- **Respuesta k6**: llama "indicación de peligro ambiental" a la designación de transporte
  ("SUSTANCIA SÓLIDA PELIGROSA PARA EL MEDIO AMBIENTE…") y dice que las categorías de peligro
  acuático son una clasificación "de la ONU", cuando son de la clasificación SGA de la sección 2.
  Tampoco da la H410, el UN3077 ni la clase 9. Una de las citas [1] la agregó la revisión
  automática, porque el punto de "N.E.P." partió la oración en dos.
- **Resultado: incompleta con k3 e incorrecta con k6.** Error del generador: la pregunta junta dos
  secciones y el modelo responde solo una (k3) o las mezcla (k6), teniendo los dos fragmentos.
- **RAGAS**: k3 1,00 · 0,00 · 1,00; k6 0,33 · 0,00 · 1,00. Faithfulness 1,00 con k3 es coherente
  con lo que mide (lo que dice es cierto), pero no ve que la respuesta está incompleta. Con k6 baja
  por la mezcla de secciones. Context Precision 0 no coincide con la revisión, por la misma razón
  que en P07. Además, en el diagnóstico el fragmento [1] con k6 salió "útil", aunque la razón del
  evaluador dice que no menciona la H410: el evaluador no es del todo estable en estos casos.

### P10. Número ONU y clase de transporte: ESMALTE TOP QUALITY NEGRO 1772 (Pintuco) frente a PINTURA EXTERIORES de Corona, versión de 2025

- **Referencia**: esmalte UN1263, clase 3, grupo III. Pintura Exteriores (versión 9, 2025)
  UN3082, clase 9, grupo III.
- **Recuperación**: correcta en las dos configuraciones (sección 14 de FDS035 y de FDS091). En la
  corrida 1 la detección descartaba el esmalte (ver README, "Dos corridas de generación").
- **Extracción**: en FDS091 la etiqueta de la clase quedó partida por su valor ("14.3 Clase(s)
  relativas al 9" / "transporte:" / "Etiquetas: 9"). Está registrado en
  `data/verificacion/revision_manual.csv`.
- **Respuesta k3**: esmalte UN1263 y clase 3 [1]; Pintura Exteriores UN3082 y clase 9 [2].
- **Respuesta k6**: Pintura Exteriores UN3082 y "clase 3" [2]; dice que la ficha del esmalte "no
  menciona el número ONU" [1], aunque el fragmento [1] dice "14.1 Número ONU: UN1263".
- **Resultado: correcta con k3 e incorrecta con k6.** Con más contexto, el modelo de 3B le pasó la
  clase de un producto al otro, en la ficha donde la etiqueta de la clase quedó desordenada.
- **RAGAS**: k3 1,00 · 0,00 · 1,00; k6 0,50 · 0,00 · 1,00. Faithfulness coincide con la revisión
  (baja con k6 por la clase equivocada y el "no menciona el número ONU"). Context Precision 0 no
  coincide: cada fragmento de la sección 14 trae solo uno de los dos productos.

## Otras respuestas revisadas contra la referencia

- **P04, primeros auxilios por contacto con la piel (Sikafloor-161 Part A)**: con k3 solo dice
  "lavar con abundante agua", aunque el fragmento [1] (sección 4) trae también quitarse la ropa
  contaminada, usar jabón y llamar a un médico si persisten los síntomas. Con k6 responde
  completo. RAGAS: 1,00 · 1,00 · 1,00 en las dos. **Incompleta con k3 y correcta con k6.**
- **P06, protección respiratoria (Pintura Lavable Antibacterial de Corona)**: las dos respuestas
  dan la máscara autofiltrante para gases y vapores, pero no dicen cuándo reemplazarla ni el uso
  de equipos aislantes. Con k6 cita también [2], que es de FDS088, otra ficha (ver la limitación de
  nombres contenidos en otros, en el README). RAGAS: 1,00 · 1,00 · 1,00 en las dos.
  **Incompleta en las dos.**

## Conclusiones de la revisión

1. En las 6 preguntas revisadas en detalle la recuperación trajo la evidencia necesaria. Los
   errores son del generador: mezcla de versiones (P07), falso "sin evidencia" en una comparación
   (P08), respuesta a una sola de dos secciones o mezcla de ambas (P09) y atribución de un dato al
   producto equivocado (P10 con k6).
2. Las dos preguntas sin respuesta se respondieron bien, aunque el umbral de similitud no se
   activó en ninguna (0,74 y 0,57). La protección vino del prompt.
3. Las preguntas integradoras son el punto débil del modelo de 3B, y aumentar k no lo arregla:
   P09 y P10 empeoran con k6.
4. Un error de extracción (etiqueta partida en FDS091) coincide con uno de los errores del
   generador. Una extracción más limpia de la cuadrícula de la sección 14 reduciría ese riesgo.
5. Faithfulness de RAGAS coincide con la revisión en las afirmaciones falsas (P07, P08, P09 y P10
   con k6), pero no detecta respuestas incompletas (P04 con k3, P06, P09 con k3).
6. Context Precision de RAGAS no sirve para las integradoras con este evaluador: da 0 en P07 a
   P10 porque descarta cada fragmento que trae solo una parte de la referencia. Contando como útil
   el fragmento que trae una parte, con k3 sería 1,00 en P07, P09 y P10 y 0,58 en P08
   (fragmentos útiles en los puestos 2 y 3). Este cálculo es manual y no reemplaza los puntajes.
