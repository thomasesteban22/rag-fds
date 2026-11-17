# Resultados RAGAS

Evaluador: gemini-3.5-flash-lite (temperatura 0). Promedios solo sobre casos válidos.

| Configuración | Métrica | Promedio | Casos válidos | Fallos | Sin evaluar |
|---|---|---|---|---|---|
| k3 | faithfulness | 0.88 | 10/10 | 0 | 0 |
| k3 | context_precision | 0.533 | 10/10 | 0 | 0 |
| k3 | context_recall | 0.975 | 10/10 | 0 | 0 |
| k6 | faithfulness | 0.75 | 10/10 | 0 | 0 |
| k6 | context_precision | 0.533 | 10/10 | 0 | 0 |
| k6 | context_recall | 1.0 | 10/10 | 0 | 0 |

## Por pregunta

| Pregunta | Tipo | k | Faithfulness | Context Precision | Context Recall | Insuficiente |
|---|---|---|---|---|---|---|
| P01 | directa | 3 | 1.000 | 1.000 | 1.000 | no |
| P01 | directa | 6 | 1.000 | 1.000 | 1.000 | no |
| P02 | directa | 3 | 1.000 | 0.333 | 1.000 | no |
| P02 | directa | 6 | 1.000 | 0.333 | 1.000 | no |
| P03 | directa | 3 | 1.000 | 1.000 | 0.750 | no |
| P03 | directa | 6 | 1.000 | 1.000 | 1.000 | no |
| P04 | directa | 3 | 1.000 | 1.000 | 1.000 | no |
| P04 | directa | 6 | 1.000 | 1.000 | 1.000 | no |
| P05 | directa | 3 | 1.000 | 1.000 | 1.000 | no |
| P05 | directa | 6 | 1.000 | 1.000 | 1.000 | no |
| P06 | directa | 3 | 1.000 | 1.000 | 1.000 | no |
| P06 | directa | 6 | 1.000 | 1.000 | 1.000 | no |
| P07 | integradora: compara versiones | 3 | 0.800 | 0.000 | 1.000 | no |
| P07 | integradora: compara versiones | 6 | 0.667 | 0.000 | 1.000 | no |
| P08 | integradora: compara productos | 3 | 0.000 | 0.000 | 1.000 | no |
| P08 | integradora: compara productos | 6 | 0.000 | 0.000 | 1.000 | sí |
| P09 | integradora: dos secciones | 3 | 1.000 | 0.000 | 1.000 | no |
| P09 | integradora: dos secciones | 6 | 0.333 | 0.000 | 1.000 | no |
| P10 | integradora: compara productos de dos fabricantes | 3 | 1.000 | 0.000 | 1.000 | no |
| P10 | integradora: compara productos de dos fabricantes | 6 | 0.500 | 0.000 | 1.000 | no |
| P11 | sin respuesta en el corpus | 3 | — | — | — | sí |
| P11 | sin respuesta en el corpus | 6 | — | — | — | sí |
| P12 | sin respuesta en el corpus | 3 | — | — | — | sí |
| P12 | sin respuesta en el corpus | 6 | — | — | — | sí |
