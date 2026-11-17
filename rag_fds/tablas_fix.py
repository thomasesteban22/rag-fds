"""Reparación de tablas con filas fusionadas.

Uso:
    uv run python -m rag_fds.tablas_fix      # sobre todo el corpus ya extraído (se puede repetir)

En tablas sin líneas horizontales entre filas (p. ej. composición o toxicidad en fichas de
Axalta, Sika y PPG), Docling junta todas las filas en una: el texto llega completo, pero se
pierde qué porcentaje o resultado corresponde a qué sustancia.

Reconstrucción:
  1. Las columnas y la fila de encabezado se toman de las celdas que detectó Docling.
  2. Las palabras de la capa de texto del PDF dentro del área de datos se asignan a su columna
     por posición horizontal y se agrupan en renglones.
  3. La columna con menos renglones (al menos 2) marca dónde empieza cada fila; cada renglón de
     las demás columnas va a la fila cuyo inicio está a su altura o justo encima.
  4. Solo se reemplaza la tabla si las palabras de la versión reconstruida son exactamente las
     mismas que las de la tabla original; si no, se deja la original y se registra.
La tabla reconstruida queda marcada en el Markdown y en data/metadata/tablas_reconstruidas.csv.
"""
import csv
import gzip
import json
import re
from collections import Counter

import pypdfium2 as pdfium

from rag_fds import config
from rag_fds.pdftexto import palabras_de_pagina, renglones
from rag_fds.utils import doc_id_de, listar_pdfs, sin_tildes

NOTA = "*Tabla reconstruida por posición de renglones: Docling había fusionado sus filas (ver data/metadata/tablas_reconstruidas.csv).*"
TOLERANCIA_PT = 3.0


_RE_VALOR_UNIDAD = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|mg/l|µg/l|mg/kg|mg/m³|mg/m3|ppm|horas|días|h\b)", re.I)
_RE_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


def es_tabla_fusionada(markdown: str, filas: int, columnas: int) -> bool:
    """Señal de filas fusionadas: 3 filas o menos, al menos 2 columnas y una celda que reúne varios
    registros (4 o más valores con unidad, o 2 o más números CAS). Un texto largo con cifras
    sueltas (p. ej. 'Almacénese entre 0 y 35°C... ver Sección 10') no cuenta."""
    if filas > 3 or columnas < 2:
        return False
    for linea in markdown.splitlines():
        s = linea.strip()
        if not s.startswith("|") or set(s) <= set("|-: "):
            continue
        for celda in s.strip("|").split("|"):
            if len(_RE_VALOR_UNIDAD.findall(celda)) >= 4 or len(_RE_CAS.findall(celda)) >= 2:
                return True
    return False


def _tokens(texto: str) -> Counter:
    return Counter(re.findall(r"[a-zñ0-9]+", sin_tildes(texto)))


def _celda(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).replace("|", "/").strip()


def reconstruir(tabla_json: dict, page) -> tuple[str | None, str]:
    """Devuelve (Markdown de la tabla reconstruida, motivo) o (None, motivo) si no se pudo validar."""
    celdas = tabla_json["data"]["table_cells"]
    ncol = tabla_json["data"]["num_cols"]
    if not celdas or any(c["bbox"].get("coord_origin") != "TOPLEFT" for c in celdas):
        return None, "celdas sin posición utilizable"
    encabezado = [""] * ncol
    rangos: dict[int, list[float]] = {}
    datos_t, datos_b = float("inf"), 0.0
    for c in celdas:
        col = c["start_col_offset_idx"]
        if c["end_col_offset_idx"] - col == 1:
            r = rangos.setdefault(col, [c["bbox"]["l"], c["bbox"]["r"]])
            r[0], r[1] = min(r[0], c["bbox"]["l"]), max(r[1], c["bbox"]["r"])
        if c["start_row_offset_idx"] == 0:
            encabezado[col] = _celda(c["text"])
        else:
            datos_t, datos_b = min(datos_t, c["bbox"]["t"]), max(datos_b, c["bbox"]["b"])
    if len(rangos) != ncol or datos_b <= datos_t:
        return None, "no se pudieron ubicar todas las columnas"

    izq = min(r[0] for r in rangos.values()) - 3
    der = max(r[1] for r in rangos.values()) + 3
    palabras = [p for p in palabras_de_pagina(page)
                if izq <= p.cx <= der and datos_t - TOLERANCIA_PT <= p.cy <= datos_b + TOLERANCIA_PT]

    def columna(p):
        for col, (a, b) in rangos.items():
            if a - 2 <= p.cx <= b + 2:
                return col
        return min(rangos, key=lambda col: min(abs(p.cx - rangos[col][0]), abs(p.cx - rangos[col][1])))

    por_col = {col: renglones([p for p in palabras if columna(p) == col]) for col in range(ncol)}
    candidatas = {col: rs for col, rs in por_col.items() if len(rs) >= 2}
    if not candidatas:
        return None, "ninguna columna tiene más de un renglón"
    ancla = min(candidatas, key=lambda col: len(candidatas[col]))
    inicios = [r[0].t for r in candidatas[ancla]]

    filas = [[[] for _ in range(ncol)] for _ in inicios]
    for col, rs in por_col.items():
        for r in rs:
            idx = max([i for i, ini in enumerate(inicios) if ini - TOLERANCIA_PT <= r[0].t] or [0])
            filas[idx][col].append(" ".join(p.texto for p in r))

    original = " ".join(c["text"] for c in celdas if c["start_row_offset_idx"] > 0)
    nuevo = " ".join(t for f in filas for col in f for t in col)
    if _tokens(original) != _tokens(nuevo):
        dif = (_tokens(original) - _tokens(nuevo)) + (_tokens(nuevo) - _tokens(original))
        return None, f"las palabras no coinciden con la tabla original ({sum(dif.values())} diferencias)"
    if len(filas) < 2 or any(not col for f in filas for col in f):
        return None, "alguna fila reconstruida queda con celdas vacías (la división no es confiable)"

    lineas = ["| " + " | ".join(encabezado) + " |", "|" + "|".join("---" for _ in range(ncol)) + "|"]
    lineas += ["| " + " | ".join(_celda(" ".join(col)) for col in f) + " |" for f in filas]
    return "\n".join(lineas), f"{len(filas)} filas; columna ancla {ancla + 1} ({encabezado[ancla] or 'sin título'})"


def reparar_documento(doc_id: str, pdf_path) -> list[dict]:
    dir_doc = config.DIR_MD / doc_id
    ruta_tablas = dir_doc / "tablas.json"
    ruta_md = dir_doc / f"{doc_id}.md"
    tablas = json.loads(ruta_tablas.read_text(encoding="utf-8"))
    pendientes = [t for t in tablas if not t.get("reconstruida") and es_tabla_fusionada(t["markdown"], t["filas"], t["columnas"])]
    if not pendientes:
        return []
    with gzip.open(config.DIR_DOCLING / f"{doc_id}.json.gz", "rt", encoding="utf-8") as f:
        d = json.load(f)
    md = ruta_md.read_text(encoding="utf-8")
    pdf = pdfium.PdfDocument(str(pdf_path))
    registros = []
    for t in pendientes:
        tj = d["tables"][int(t["ref"].split("/")[-1])]
        nuevo, motivo = reconstruir(tj, pdf[t["pagina"] - 1])
        original = "\n".join(l for l in t["markdown"].splitlines() if l.strip().startswith("|"))
        estado = "reconstruida"
        if nuevo is None:
            estado = "sin reconstruir"
        elif original not in md:
            estado, motivo = "sin reconstruir", "no se encontró la tabla original en el Markdown"
        else:
            md = md.replace(original, nuevo + "\n\n" + NOTA, 1)
            t["reconstruida"] = True
            t["detalle_reconstruccion"] = motivo
            t["markdown_original"] = t["markdown"]
            t["markdown"] = nuevo
            t["filas"] = nuevo.count("\n") - 1
        registros.append({"doc_id": doc_id, "ref": t["ref"], "pagina": t["pagina"], "seccion": t["seccion"] or "",
                          "estado": estado, "detalle": motivo})
    ruta_md.write_text(md, encoding="utf-8")
    ruta_tablas.write_text(json.dumps(tablas, ensure_ascii=False, indent=1), encoding="utf-8")
    return registros


def main():
    filas = []
    for pdf in listar_pdfs():
        doc_id = doc_id_de(pdf.name)
        ruta_tablas = config.DIR_MD / doc_id / "tablas.json"
        if not ruta_tablas.exists():
            continue
        nuevos = reparar_documento(doc_id, pdf)
        # Las tablas que ya reconstruyó la extracción quedan marcadas en tablas.json.
        hechas = {r["ref"] for r in nuevos}
        for t in json.loads(ruta_tablas.read_text(encoding="utf-8")):
            if t.get("reconstruida") and t["ref"] not in hechas:
                filas.append({"doc_id": doc_id, "ref": t["ref"], "pagina": t["pagina"], "seccion": t["seccion"] or "",
                              "estado": "reconstruida",
                              "detalle": t.get("detalle_reconstruccion", f"{t['filas']} filas")})
        filas += nuevos
    filas.sort(key=lambda r: (r["doc_id"], int(r["ref"].split("/")[-1])))
    with open(config.DIR_META / "tablas_reconstruidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "ref", "pagina", "seccion", "estado", "detalle"])
        w.writeheader()
        w.writerows(filas)
    for r in filas:
        print(f"{r['doc_id']} {r['ref']} p{r['pagina']} s{r['seccion']}: {r['estado']} — {r['detalle']}")
    print(f"{sum(r['estado'] == 'reconstruida' for r in filas)} tablas reconstruidas, "
          f"{sum(r['estado'] != 'reconstruida' for r in filas)} sin reconstruir")


if __name__ == "__main__":
    main()
