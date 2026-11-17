"""Paso 5: evaluación del RAG con RAGAS usando Gemini como evaluador.

Uso (en este orden):
    uv run python -m rag_fds.evaluate desarrollo # preguntas de desarrollo (distintas): revisar prompt y umbral
    uv run python -m rag_fds.evaluate generar    # corre el RAG con top-k 3 y 6 sobre las 12 preguntas
    uv run python -m rag_fds.evaluate prueba     # prueba el evaluador con 2 casos y registra su configuración
    uv run python -m rag_fds.evaluate puntuar    # Faithfulness, Context Precision y Context Recall (10 respondibles x 2)
    uv run python -m rag_fds.evaluate diagnostico # veredicto y razón por fragmento en Context Precision
    uv run python -m rag_fds.evaluate resumen    # CSV/JSON por pregunta, promedios y casos válidos

Las dos configuraciones solo cambian top-k (3 y 6); corpus, chunking, embeddings, generador,
prompt y evaluador son los mismos. Las respuestas de referencia (eval/preguntas.json) solo las
usa el evaluador: no están en el índice ni en el prompt del generador.

El evaluador es un LLM de Gemini (API gratuita, autorizado por el profesor), con temperatura 0,
una llamada a la vez y reintentos con espera exponencial. Un fallo de evaluación se guarda como
fallo (valor nulo y mensaje de error), nunca como cero. El modelo se cambia con la variable de
entorno RAGAS_MODELO, y cada modelo guarda sus resultados en su propia carpeta
(eval/resultados/evaluador_<modelo>/), para no mezclar puntajes de evaluadores distintos.

Si se agota la cuota diaria del plan gratuito, la corrida se detiene sin marcar como fallo los
casos que no alcanzó a evaluar: quedan "sin evaluar" y 'puntuar' los retoma al volver a correrlo.

La clave va en .env como GOOGLE_API_KEY (ese archivo está en .gitignore).
"""
import argparse
import asyncio
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from importlib.metadata import version

from dotenv import load_dotenv

from rag_fds import config

CONFIGURACIONES = {"k3": 3, "k6": 6}
MODELO_EVALUADOR = os.environ.get("RAGAS_MODELO", "gemini-3.5-flash-lite")  # ver README, "Evaluador de RAGAS"
TEMPERATURA_EVALUADOR = 0.0
REINTENTOS = 6
ESPERA_BASE_S = 10
ESPERA_MAX_S = 120
TIMEOUT_S = 900             # una métrica puede hacer varias llamadas de 20 a 50 s (p. ej. 6 contextos)
PAUSA_ENTRE_LLAMADAS_S = 6   # plan gratuito: pocas solicitudes por minuto
METRICAS = ["faithfulness", "context_precision", "context_recall"]

DIR_RES = config.DIR_EVAL / "resultados"
DIR_EVALUADOR = DIR_RES / f"evaluador_{MODELO_EVALUADOR}"
RUTA_PREGUNTAS = config.DIR_EVAL / "preguntas.json"
RUTA_PUNTAJES = DIR_EVALUADOR / "puntuaciones.json"
RUTA_CONFIG_EVALUADOR = DIR_EVALUADOR / "evaluador_config.json"


class CuotaDiariaAgotada(Exception):
    """El plan gratuito llegó a su límite de solicitudes por día para este modelo."""


def _preguntas() -> list[dict]:
    return json.loads(RUTA_PREGUNTAS.read_text(encoding="utf-8"))


def _guardar(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- generación
def generar():
    from rag_fds.rag import RAG

    for nombre, k in CONFIGURACIONES.items():
        rag = RAG(top_k=k)
        rag.responder("¿Qué es una ficha de datos de seguridad?")  # calentamiento: carga los modelos en la GPU
        filas = []
        for p in _preguntas():
            r = rag.responder(p["pregunta"])
            filas.append({
                "configuracion": nombre,
                "top_k": k,
                "id": p["id"],
                "tipo": p["tipo"],
                "respondible": p["respondible"],
                "user_input": p["pregunta"],
                "retrieved_contexts": [f.texto for f in r.fragmentos],
                "retrieved_ids": [f.id for f in r.fragmentos],
                "similitudes": [f.similitud for f in r.fragmentos],
                "similitud_max": r.similitud_max,
                "response": r.respuesta,
                "response_modelo": r.respuesta_modelo,
                "reference": p.get("referencia"),
                "fuentes_recuperadas": r.fuentes(solo_citadas=False),
                "fuentes_citadas": r.fuentes() if r.citados else [],
                "insuficiente": r.insuficiente,
                "filtro": r.filtro,
                "advertencias": r.advertencias,
                "segundos": r.segundos,
            })
            print(f"[{nombre}] {p['id']} ({r.segundos}) -> {r.respuesta[:110]!r}")
        _guardar(DIR_RES / f"respuestas_{nombre}.json", filas)
    _guardar(DIR_RES / "generador_config.json", {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "generador": config.MODELO_GENERADOR, "temperatura": config.TEMPERATURA_GENERADOR,
        "num_ctx": config.CONTEXTO_GENERADOR, "seed": 42,
        "embeddings": config.MODELO_EMBEDDINGS,
        "chunk_max_chars": config.CHUNK_MAX_CHARS, "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
        "configuraciones": CONFIGURACIONES,
    })


def desarrollo():
    """Preguntas de desarrollo (distintas de las 12 evaluadas) para revisar el prompt y el umbral."""
    from rag_fds.rag import RAG

    rag = RAG(top_k=CONFIGURACIONES["k6"])
    preguntas = json.loads((config.DIR_EVAL / "preguntas_desarrollo.json").read_text(encoding="utf-8"))
    filas = []
    for p in preguntas:
        r = rag.responder(p["pregunta"])
        filas.append({"id": p["id"], "respondible": p["respondible"], "pregunta": p["pregunta"],
                      "respuesta": r.respuesta, "insuficiente": r.insuficiente, "filtro": r.filtro,
                      "similitud_max": r.similitud_max,
                      "similitudes": [f.similitud for f in r.fragmentos], "fuentes": r.fuentes(),
                      "advertencias": r.advertencias})
        print(f"{p['id']} (respondible={p['respondible']}) sim={filas[-1]['similitud_max']} "
              f"insuficiente={r.insuficiente} -> {r.respuesta[:120]!r}")
    _guardar(DIR_RES / "desarrollo.json", filas)


# ---------------------------------------------------------------- evaluador
def _cliente():
    load_dotenv(config.RAIZ / ".env")
    clave = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not clave:
        sys.exit("Falta GOOGLE_API_KEY en el archivo .env")
    from google import genai

    return genai.Client(api_key=clave)


def _crear_metricas(cliente):
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    llm = llm_factory(MODELO_EVALUADOR, provider="google", client=cliente, temperature=TEMPERATURA_EVALUADOR)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def _version_servida(cliente) -> str:
    """Versión del modelo que reporta la API en una respuesta real (puede diferir del alias pedido)."""
    try:
        resp = cliente.models.generate_content(model=MODELO_EVALUADOR, contents="Responde solo: OK",
                                               config={"temperature": TEMPERATURA_EVALUADOR})
        return resp.model_version or "no reportada"
    except Exception as e:  # noqa: BLE001
        if _es_cuota_diaria(e):
            raise CuotaDiariaAgotada(str(e)[:300]) from e
        return f"no disponible ({type(e).__name__})"


def _es_cuota_diaria(e: Exception) -> bool:
    texto = str(e)
    return "429" in texto and ("PerDay" in texto or "per day" in texto.lower())


def _entradas(metrica: str, fila: dict) -> dict:
    if metrica == "faithfulness":
        return {"user_input": fila["user_input"], "response": fila["response"],
                "retrieved_contexts": fila["retrieved_contexts"]}
    return {"user_input": fila["user_input"], "reference": fila["reference"],
            "retrieved_contexts": fila["retrieved_contexts"]}


async def _puntuar_una(scorer, entradas: dict) -> dict:
    """Una métrica sobre un caso, con reintentos. Devuelve valor o fallo (nunca 0 por error).

    Si la API responde que se agotó la cuota diaria, no reintenta: lanza CuotaDiariaAgotada.
    """
    errores = []
    for intento in range(1, REINTENTOS + 1):
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(scorer.ascore(**entradas), timeout=TIMEOUT_S)
            valor = res.value
            if valor is None or (isinstance(valor, float) and valor != valor):
                raise ValueError(f"el evaluador devolvió un valor no numérico: {valor!r}")
            return {"estado": "ok", "valor": float(valor), "intentos": intento,
                    "segundos": round(time.perf_counter() - t0, 1),
                    "razon": getattr(res, "reason", None), "errores_previos": errores}
        except Exception as e:  # noqa: BLE001 - se registra cualquier fallo del evaluador
            if _es_cuota_diaria(e):
                raise CuotaDiariaAgotada(str(e)[:300]) from e
            errores.append(f"intento {intento}: {type(e).__name__}: {str(e)[:300]}")
            if intento < REINTENTOS:
                espera = min(ESPERA_MAX_S, ESPERA_BASE_S * 2 ** (intento - 1)) * (0.75 + random.random() / 2)
                print(f"      fallo ({type(e).__name__}); reintento en {espera:.0f} s")
                await asyncio.sleep(espera)
    return {"estado": "fallo", "valor": None, "intentos": REINTENTOS, "errores": errores}


def _versiones() -> dict:
    v = {}
    for p in ("ragas", "google-genai", "ollama", "chromadb", "docling"):
        try:
            v[p] = version(p)
        except Exception:  # noqa: BLE001
            v[p] = "no instalado"
    return v


def _guardar_config_evaluador(version_servida: str | None = None):
    """Guarda la configuración del evaluador; conserva la versión servida medida en 'prueba'."""
    previa = json.loads(RUTA_CONFIG_EVALUADOR.read_text(encoding="utf-8")) if RUTA_CONFIG_EVALUADOR.exists() else {}
    if version_servida is None:
        version_servida = previa.get("evaluador", {}).get("version_servida", "no medida")
    _guardar(RUTA_CONFIG_EVALUADOR, {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "evaluador": {"proveedor": "Google Gemini API (plan gratuito)", "modelo": MODELO_EVALUADOR,
                      "version_servida": version_servida,
                      "temperatura": TEMPERATURA_EVALUADOR, "via": "ragas.llms.llm_factory(provider='google')"},
        "metricas": {"faithfulness": "ragas.metrics.collections.Faithfulness",
                     "context_precision": "ragas.metrics.collections.ContextPrecision (con referencia)",
                     "context_recall": "ragas.metrics.collections.ContextRecall"},
        "ejecucion": {"llamadas_simultaneas": 1, "reintentos": REINTENTOS, "espera_base_s": ESPERA_BASE_S,
                      "espera_max_s": ESPERA_MAX_S, "timeout_s": TIMEOUT_S, "pausa_entre_llamadas_s": PAUSA_ENTRE_LLAMADAS_S},
        "versiones": _versiones(),
        "python": platform.python_version(),
    })


def _aviso_cuota(e: CuotaDiariaAgotada, pendientes: int):
    print(f"\nCuota diaria agotada para {MODELO_EVALUADOR}. Quedan {pendientes} puntajes sin evaluar "
          f"(no se marcan como fallo). Vuelve a correr el mismo paso cuando se reinicie la cuota.\n  {e}")


async def _prueba():
    cliente = _cliente()
    metricas = _crear_metricas(cliente)
    filas = json.loads((DIR_RES / "respuestas_k3.json").read_text(encoding="utf-8"))
    casos = [f for f in filas if f["id"] in ("P01", "P07")]
    out = []
    try:
        version_servida = _version_servida(cliente)
        print(f"  {MODELO_EVALUADOR}: versión servida {version_servida}")
        _guardar_config_evaluador(version_servida)
        for f in casos:
            for m, scorer in metricas.items():
                r = await _puntuar_una(scorer, _entradas(m, f))
                print(f"  prueba {f['id']} {m}: {r['estado']} {r.get('valor')}")
                out.append({"id": f["id"], "configuracion": f["configuracion"], "metrica": m, **r})
                _guardar(DIR_EVALUADOR / "prueba_evaluador.json", out)
                await asyncio.sleep(PAUSA_ENTRE_LLAMADAS_S)
    except CuotaDiariaAgotada as e:
        _aviso_cuota(e, len(casos) * len(metricas) - len(out))
        sys.exit(2)


async def _puntuar(forzar: bool):
    metricas = _crear_metricas(_cliente())
    puntajes = json.loads(RUTA_PUNTAJES.read_text(encoding="utf-8")) if RUTA_PUNTAJES.exists() and not forzar else {}
    pendientes = []
    for nombre in CONFIGURACIONES:
        filas = json.loads((DIR_RES / f"respuestas_{nombre}.json").read_text(encoding="utf-8"))
        for f in filas:
            if not f["respondible"]:
                continue
            for m in metricas:
                clave = f"{nombre}|{f['id']}|{m}"
                if not (clave in puntajes and puntajes[clave]["estado"] == "ok"):
                    pendientes.append((clave, m, f))
    print(f"{MODELO_EVALUADOR}: {len(pendientes)} puntajes pendientes")
    try:
        for i, (clave, m, f) in enumerate(pendientes):
            print(f"[{clave}] ...")
            previo = puntajes.get(clave)
            puntajes[clave] = await _puntuar_una(metricas[m], _entradas(m, f))
            if previo and previo["estado"] == "fallo":  # el fallo anterior se conserva, no se borra
                anteriores = previo.pop("corridas_fallidas_previas", [])
                puntajes[clave]["corridas_fallidas_previas"] = anteriores + [previo]
            print(f"      {puntajes[clave]['estado']} {puntajes[clave].get('valor')}")
            _guardar(RUTA_PUNTAJES, puntajes)
            await asyncio.sleep(PAUSA_ENTRE_LLAMADAS_S)
    except CuotaDiariaAgotada as e:
        _guardar_config_evaluador()
        _aviso_cuota(e, len(pendientes) - i)
        sys.exit(2)
    _guardar_config_evaluador()


async def _diagnostico(ids: list[str]):
    """Veredicto y razón del evaluador por fragmento en Context Precision (RAGAS solo guarda el promedio).

    Repite la misma consulta que hace ContextPrecision (mismo prompt, modelo y temperatura) para
    ver qué fragmentos juzgó útiles y por qué. No cambia los puntajes guardados.
    """
    from ragas.metrics.collections.context_precision.util import (
        ContextPrecisionInput, ContextPrecisionOutput, ContextPrecisionPrompt)

    llm = _crear_metricas(_cliente())["context_precision"].llm
    prompt = ContextPrecisionPrompt()
    ruta = DIR_EVALUADOR / "diagnostico_context_precision.json"
    out = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    try:
        for nombre in CONFIGURACIONES:
            for f in json.loads((DIR_RES / f"respuestas_{nombre}.json").read_text(encoding="utf-8")):
                if f["id"] not in ids:
                    continue
                veredictos = []
                for cid, contexto in zip(f["retrieved_ids"], f["retrieved_contexts"]):
                    texto = prompt.to_string(ContextPrecisionInput(
                        question=f["user_input"], context=contexto, answer=f["reference"]))
                    for intento in range(1, REINTENTOS + 1):
                        try:
                            r = await llm.agenerate(texto, ContextPrecisionOutput)
                            veredictos.append({"fragmento": cid, "veredicto": r.verdict, "razon": r.reason})
                            break
                        except Exception as e:  # noqa: BLE001
                            if _es_cuota_diaria(e):
                                raise CuotaDiariaAgotada(str(e)[:300]) from e
                            if intento == REINTENTOS:
                                veredictos.append({"fragmento": cid, "veredicto": None,
                                                   "razon": f"fallo: {type(e).__name__}: {str(e)[:200]}"})
                            else:
                                await asyncio.sleep(min(ESPERA_MAX_S, ESPERA_BASE_S * 2 ** (intento - 1)))
                    await asyncio.sleep(PAUSA_ENTRE_LLAMADAS_S)
                out[f"{nombre}|{f['id']}"] = veredictos
                _guardar(ruta, out)
                print(f"[{nombre}] {f['id']}: " + " ".join(str(v["veredicto"]) for v in veredictos))
    except CuotaDiariaAgotada as e:
        _aviso_cuota(e, 0)
        sys.exit(2)


# ---------------------------------------------------------------- resumen
def resumen():
    puntajes = json.loads(RUTA_PUNTAJES.read_text(encoding="utf-8")) if RUTA_PUNTAJES.exists() else {}
    filas_csv, todas = [], []
    for nombre in CONFIGURACIONES:
        for f in json.loads((DIR_RES / f"respuestas_{nombre}.json").read_text(encoding="utf-8")):
            fila = dict(f)
            for m in METRICAS:
                p = puntajes.get(f"{nombre}|{f['id']}|{m}")
                if not f["respondible"]:
                    fila[m], fila[f"{m}_estado"] = None, "no aplica (pregunta sin respuesta)"
                elif p is None:
                    fila[m], fila[f"{m}_estado"] = None, "sin evaluar"
                else:
                    fila[m], fila[f"{m}_estado"] = p["valor"], p["estado"]
            todas.append(fila)
            filas_csv.append({
                **{k: fila[k] for k in ("configuracion", "top_k", "id", "tipo", "respondible", "user_input")},
                "retrieved_contexts": json.dumps(fila["retrieved_contexts"], ensure_ascii=False),
                "retrieved_ids": ";".join(fila["retrieved_ids"]),
                "response": fila["response"],
                "reference": fila["reference"] or "",
                "fuentes_citadas": " || ".join(fila["fuentes_citadas"]),
                "fuentes_recuperadas": " || ".join(fila["fuentes_recuperadas"]),
                "insuficiente": fila["insuficiente"],
                **{m: fila[m] for m in METRICAS},
                **{f"{m}_estado": fila[f"{m}_estado"] for m in METRICAS},
            })
    _guardar(DIR_EVALUADOR / "resultados.json", todas)
    with open(DIR_EVALUADOR / "resultados.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_csv[0].keys()))
        w.writeheader()
        w.writerows(filas_csv)

    lineas = ["# Resultados RAGAS", "", f"Evaluador: {MODELO_EVALUADOR} (temperatura 0). Promedios solo sobre casos válidos.", "",
              "| Configuración | Métrica | Promedio | Casos válidos | Fallos | Sin evaluar |", "|---|---|---|---|---|---|"]
    promedios = {}
    for nombre in CONFIGURACIONES:
        for m in METRICAS:
            casos = [t for t in todas if t["configuracion"] == nombre and t["respondible"]]
            vals = [t[m] for t in casos if t[f"{m}_estado"] == "ok"]
            fallos = sum(1 for t in casos if t[f"{m}_estado"] == "fallo")
            sin_evaluar = sum(1 for t in casos if t[f"{m}_estado"] == "sin evaluar")
            prom = round(sum(vals) / len(vals), 3) if vals else None
            promedios[f"{nombre}|{m}"] = {"promedio": prom, "validos": len(vals), "fallos": fallos,
                                          "sin_evaluar": sin_evaluar}
            lineas.append(f"| {nombre} | {m} | {prom if prom is not None else '—'} | {len(vals)}/{len(casos)} "
                          f"| {fallos} | {sin_evaluar} |")
    lineas += ["", "## Por pregunta", "", "| Pregunta | Tipo | k | Faithfulness | Context Precision | Context Recall | Insuficiente |",
               "|---|---|---|---|---|---|---|"]
    fmt = lambda v, e: f"{v:.3f}" if v is not None else ("fallo" if e == "fallo" else "—")  # noqa: E731
    for t in sorted(todas, key=lambda t: (t["id"], t["top_k"])):
        lineas.append(f"| {t['id']} | {t['tipo']} | {t['top_k']} | " + " | ".join(fmt(t[m], t[f'{m}_estado']) for m in METRICAS)
                      + f" | {'sí' if t['insuficiente'] else 'no'} |")
    (DIR_EVALUADOR / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    _guardar(DIR_EVALUADOR / "promedios.json", promedios)
    print("\n".join(lineas))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=["desarrollo", "generar", "prueba", "puntuar", "diagnostico", "resumen"])
    ap.add_argument("--force", action="store_true", help="en 'puntuar', vuelve a evaluar todo")
    ap.add_argument("--ids", default="P02,P07,P08,P09,P10",
                    help="en 'diagnostico', preguntas a revisar (separadas por coma)")
    args = ap.parse_args()
    if args.paso == "desarrollo":
        desarrollo()
    elif args.paso == "generar":
        generar()
    elif args.paso == "prueba":
        asyncio.run(_prueba())
    elif args.paso == "puntuar":
        asyncio.run(_puntuar(args.force))
    elif args.paso == "diagnostico":
        asyncio.run(_diagnostico(args.ids.split(",")))
    else:
        resumen()


if __name__ == "__main__":
    main()
