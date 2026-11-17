"""Registro de hardware y consumo de recursos del proyecto.

Uso:
    uv run python -m rag_fds.recursos

Escribe docs/recursos.json con: CPU, RAM, GPU (nvidia-smi), modelos cargados en Ollama y la
memoria que ocupan, tiempos de extracción (data/metadata/extraccion_log.csv), tiempos por
consulta del RAG (eval/resultados/respuestas_*.json) y espacio en disco de cada artefacto.
"""
import csv
import json
import platform
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import ollama
import psutil

from rag_fds import config


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"no disponible: {e}"


def _cpu() -> str:
    if platform.system() == "Windows":
        salida = _cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        return salida or platform.processor()
    return platform.processor()


def _tamano_mb(ruta: Path) -> float:
    if not ruta.exists():
        return 0.0
    return round(sum(p.stat().st_size for p in ruta.rglob("*") if p.is_file()) / 2**20, 1)


def main():
    gpu = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
    try:
        cargados = [{"modelo": m.model, "vram_mb": round(m.size_vram / 2**20), "total_mb": round(m.size / 2**20)}
                    for m in ollama.Client(host=config.OLLAMA_URL).ps().models]
        version_ollama = ollama.Client(host=config.OLLAMA_URL)._client.get("/api/version").json()["version"]
    except Exception as e:  # noqa: BLE001
        cargados, version_ollama = f"no disponible: {e}", "?"

    extraccion = {}
    ruta_log = config.DIR_META / "extraccion_log.csv"
    if ruta_log.exists():
        with open(ruta_log, newline="", encoding="utf-8") as f:
            filas = [r for r in csv.DictReader(f) if r.get("seg_docling")]
        seg = [float(r["seg_docling"]) + float(r["seg_imagenes_ocr"]) for r in filas]
        pags = sum(int(r["paginas"]) for r in filas)
        extraccion = {
            "documentos": len(filas), "paginas": pags,
            "minutos_total": round(sum(seg) / 60, 1),
            "segundos_por_pagina": round(sum(seg) / pags, 2) if pags else None,
            "segundos_docling_total": round(sum(float(r["seg_docling"]) for r in filas), 1),
            "segundos_imagenes_ocr_total": round(sum(float(r["seg_imagenes_ocr"]) for r in filas), 1),
            "ram_proceso_max_mb": max((int(r["ram_proceso_mb"]) for r in filas if r.get("ram_proceso_mb")), default=None),
        }

    consultas = {}
    for ruta in sorted((config.DIR_EVAL / "resultados").glob("respuestas_*.json")):
        filas = json.loads(ruta.read_text(encoding="utf-8"))
        rec = [f["segundos"]["recuperacion"] for f in filas]
        gen = [f["segundos"]["generacion"] for f in filas if f["segundos"]["generacion"] > 0]
        consultas[ruta.stem] = {
            "preguntas": len(filas),
            "recuperacion_s_mediana": round(statistics.median(rec), 2) if rec else None,
            "generacion_s_mediana": round(statistics.median(gen), 2) if gen else None,
            "generacion_s_max": round(max(gen), 2) if gen else None,
        }

    disco = {
        "pdf_originales_mb": _tamano_mb(config.DIR_PDF),
        "markdown_e_imagenes_mb": _tamano_mb(config.DIR_MD),
        "docling_json_mb": _tamano_mb(config.DIR_DOCLING),
        "indice_chroma_mb": _tamano_mb(config.DIR_CHROMA),
        "entorno_python_mb": _tamano_mb(config.RAIZ / ".venv"),
        "disco_libre_gb": round(shutil.disk_usage(config.RAIZ).free / 2**30, 1),
    }

    datos = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "equipo": {
            "so": platform.platform(),
            "cpu": _cpu(),
            "nucleos_fisicos": psutil.cpu_count(logical=False),
            "hilos": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "gpu": gpu,
        },
        "ollama": {"version": version_ollama, "modelos_cargados": cargados},
        "extraccion": extraccion,
        "consultas_rag": consultas,
        "disco": disco,
    }
    destino = config.RAIZ / "docs" / "recursos.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(datos, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
