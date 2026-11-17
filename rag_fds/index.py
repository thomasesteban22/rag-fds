"""Paso 4: chunking por sección de la FDS e indexado en Chroma con embeddings bge-m3 (Ollama).

Uso:
    uv run python -m rag_fds.index

Estrategia de chunking:
  1. Cada Markdown se recorre en orden y cada línea queda asignada a su página y a su sección
     canónica (1-16). Lo que va antes de la sección 1 es la sección 0 (encabezado de la ficha)
     y el bloque de encabezados y pies de página es la sección 17 (texto repetido de márgenes).
  2. Dentro de una sección, el texto se divide en bloques: párrafos separados por línea en blanco
     y tablas completas. Los enlaces de imagen se quitan del texto (en una fila de tabla, solo el
     enlace) pero el archivo de la imagen queda en los metadatos del fragmento; las notas con el
     texto del PDF o el texto OCR de las imágenes sí se conservan.
  3. Los bloques se agrupan en fragmentos de hasta CHUNK_MAX_CHARS caracteres. Un fragmento
     nunca mezcla dos secciones ni dos fichas. Entre fragmentos consecutivos de la misma sección
     se repiten los últimos bloques hasta CHUNK_OVERLAP_CHARS caracteres (solapamiento).
     Una tabla demasiado larga se parte por filas repitiendo la fila de encabezado.
  4. Cada fragmento lleva al inicio una cabecera con fabricante, producto, versión, documento y
     sección, para que el modelo de embeddings y el generador distingan productos y versiones.

Salidas: data/chroma/ (índice) y data/index/fragmentos.jsonl (los mismos fragmentos, para revisión).
"""
import csv
import json
import re
import time

import chromadb
import ollama

from rag_fds import config
from rag_fds.sections import MARCA_PIES, RE_LINEA_PIE, SECCIONES, lista_de_paginas, paginas_de_lista, recorrer_markdown

SECCION_ENCABEZADO = 0
SECCION_PIES = 17
TITULOS = {**SECCIONES, SECCION_ENCABEZADO: "Encabezado de la ficha", SECCION_PIES: "Encabezados y pies de página"}

_RE_IMG = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_RE_FRONT = re.compile(r"^---\n.*?\n---\n", re.S)
_RE_SEPARADOR_TABLA = re.compile(r"\|[\s\-:|]+\|")


def _compactar_fila(linea: str) -> str:
    """Quita el relleno de espacios que Docling pone para alinear columnas en las tablas.

    También une celdas idénticas consecutivas: Docling repite el texto de una celda combinada en
    cada columna que abarca (p. ej. FDS071 p. 7) y en el fragmento solo ocuparía espacio.
    """
    s = linea.strip()
    if _RE_SEPARADOR_TABLA.fullmatch(s):
        return "|" + "|".join("---" for _ in s.strip("|").split("|")) + "|"
    celdas = [re.sub(r"\s{2,}", " ", c).strip() for c in s.strip("|").split("|")]
    unidas = [c for i, c in enumerate(celdas) if i == 0 or not c or c != celdas[i - 1]]
    return "| " + " | ".join(unidas) + " |"


def cargar_inventario() -> dict[str, dict]:
    with open(config.DIR_META / "inventario.csv", newline="", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def fecha_version(fila: dict) -> str:
    return fila.get("fecha_revision") or fila.get("fecha_emision") or fila.get("fecha_actualizacion") or ""


def _bloques(lineas: list[tuple[int, str]]):
    """Agrupa líneas (página, texto) en bloques: tablas completas o párrafos. Devuelve (texto, páginas, imágenes)."""
    actual, paginas, imagenes, es_tabla = [], set(), [], False

    def cerrar():
        nonlocal actual, paginas, imagenes
        if actual or imagenes:
            yield "\n".join(actual).strip(), sorted(paginas), imagenes
        actual, paginas, imagenes = [], set(), []

    for pagina, linea in lineas:
        enlaces = _RE_IMG.findall(linea)
        if enlaces:
            imagenes += enlaces
            if pagina:
                paginas.add(pagina)
            # Dentro de una fila de tabla (pictogramas en celdas) solo se quita el enlace.
            linea = _RE_IMG.sub("", linea)
            if not linea.lstrip().startswith("|"):
                continue
        fila_tabla = linea.lstrip().startswith("|")
        if not linea.strip() or fila_tabla != es_tabla:
            yield from cerrar()
            es_tabla = fila_tabla
            if not linea.strip():
                continue
        actual.append(_compactar_fila(linea) if fila_tabla else linea)
        if pagina:
            paginas.add(pagina)
    yield from cerrar()


def _partir_tabla(texto: str, maximo: int) -> list[str]:
    filas = texto.splitlines()
    cab = filas[:2] if len(filas) > 2 and set(filas[1].replace("|", "").strip()) <= set("-: ") else filas[:1]
    largo_cab = len("\n".join(cab))
    partes, actual = [], list(cab)
    for f in filas[len(cab):]:
        if len(f) + largo_cab > maximo:
            # Fila gigante (p. ej. varias filas que Docling fusionó en una): se parte como texto.
            if len(actual) > len(cab):
                partes.append("\n".join(actual))
            actual = list(cab)
            partes += ["\n".join(cab) + "\n" + t for t in _partir_texto(f, maximo - largo_cab - 1)]
            continue
        if len("\n".join(actual + [f])) > maximo and len(actual) > len(cab):
            partes.append("\n".join(actual))
            actual = list(cab)
        actual.append(f)
    if len(actual) > len(cab):
        partes.append("\n".join(actual))
    return partes


def _partir_texto(texto: str, maximo: int) -> list[str]:
    oraciones = re.split(r"(?<=[.;:])\s+", texto)
    partes, actual = [], ""
    for o in oraciones:
        while len(o) > maximo:  # oración sin puntos: corte duro en el último espacio
            if actual:
                partes.append(actual)
                actual = ""
            corte = o.rfind(" ", 0, maximo)
            corte = corte if corte > 0 else maximo
            partes.append(o[:corte])
            o = o[corte:].strip()
        if len(actual) + len(o) + 1 > maximo and actual:
            partes.append(actual)
            actual = o
        else:
            actual = (actual + " " + o).strip()
    if actual:
        partes.append(actual)
    return partes


def _agrupar(bloques, maximo: int, solape: int):
    """Empaqueta bloques en fragmentos de hasta `maximo` caracteres con solapamiento por bloques."""
    unidades = []
    for texto, paginas, imagenes in bloques:
        if not texto:
            if unidades:
                unidades[-1][2].extend(imagenes)
            continue
        if len(texto) <= maximo:
            unidades.append((texto, paginas, list(imagenes)))
        else:
            trozos = _partir_tabla(texto, maximo) if texto.lstrip().startswith("|") else _partir_texto(texto, maximo)
            for i, t in enumerate(trozos):
                unidades.append((t, paginas, list(imagenes) if i == 0 else []))

    fragmentos, actual = [], []
    for u in unidades:
        if actual and sum(len(x[0]) + 2 for x in actual) + len(u[0]) > maximo:
            fragmentos.append(actual)
            # solapamiento: últimos bloques del fragmento anterior hasta `solape` caracteres
            previo, total = [], 0
            for x in reversed(actual):
                if total + len(x[0]) > solape:
                    break
                previo.insert(0, (x[0], x[1], []))
                total += len(x[0]) + 2
            if not previo and solape > 0:
                cola = actual[-1][0][-solape:]
                cola = cola[cola.find(" ") + 1:] if " " in cola else cola
                previo = [(cola, actual[-1][1], [])]
            actual = previo
        actual.append(u)
    if actual:
        fragmentos.append(actual)
    return fragmentos


def cabecera(meta: dict, seccion: int) -> str:
    version = meta["version"] or "no indicada"
    fecha = meta["fecha"] or "no indicada"
    return (f"[Fabricante: {meta['fabricante']} | Producto: {meta['producto']} | Versión: {version} | "
            f"Fecha: {fecha} | Documento: {meta['doc_id']} | Sección {seccion}: {TITULOS[seccion]}]")


def fragmentar_documento(doc_id: str, md: str, fila: dict) -> list[dict]:
    cuerpo = _RE_FRONT.sub("", md, count=1)
    meta = {
        "doc_id": doc_id,
        "fabricante": fila["fabricante"],
        "producto": fila["producto"],
        "version": fila["version"],
        "fecha": fecha_version(fila),
        "archivo": fila["archivo_normalizado"],
        "carpeta": fila["carpeta_origen"],
    }
    por_seccion: dict[int, list[tuple[int, str]]] = {}
    for _, pagina, canon, _, linea in recorrer_markdown(cuerpo):
        por_seccion.setdefault(canon or SECCION_ENCABEZADO, []).append((pagina, linea))
    if MARCA_PIES in cuerpo:
        pies = []
        for linea in cuerpo.split(MARCA_PIES, 1)[1].splitlines():
            m = RE_LINEA_PIE.match(linea)
            if m:
                ps = paginas_de_lista(m.group(1))
                pies.append((ps[0], f"- (págs. {m.group(1)}) {m.group(2)}"))
        if pies:
            por_seccion[SECCION_PIES] = pies

    fragmentos = []
    for seccion in sorted(por_seccion):
        grupos = _agrupar(_bloques(por_seccion[seccion]), config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS)
        for k, grupo in enumerate(grupos):
            texto = "\n\n".join(x[0] for x in grupo).strip()
            if len(texto) < 20:
                continue
            paginas = sorted({p for x in grupo for p in x[1] if p})
            imagenes = [i for x in grupo for i in x[2]]
            fragmentos.append({
                "id": f"{doc_id}-s{seccion:02d}-{k:03d}",
                "texto": cabecera(meta, seccion) + "\n" + texto,
                "metadatos": {
                    **meta,
                    "seccion": seccion,
                    "seccion_titulo": TITULOS[seccion],
                    "pagina_inicio": paginas[0] if paginas else 0,
                    "pagina_fin": paginas[-1] if paginas else 0,
                    "paginas": lista_de_paginas(paginas) if paginas else "",
                    "imagenes": ";".join(f"data/md/{doc_id}/{i}" for i in dict.fromkeys(imagenes)),
                    "caracteres": len(texto),
                },
            })
    return fragmentos


def embeber(cliente: ollama.Client, textos: list[str]) -> list[list[float]]:
    return cliente.embed(model=config.MODELO_EMBEDDINGS, input=textos).embeddings


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-fragmentos", action="store_true", help="genera fragmentos.jsonl sin calcular embeddings")
    args = ap.parse_args()
    inventario = cargar_inventario()
    fragmentos = []
    for doc_id, fila in inventario.items():
        md = (config.RAIZ / fila["ruta_md"]).read_text(encoding="utf-8")
        fragmentos += fragmentar_documento(doc_id, md, fila)

    dir_idx = config.RAIZ / "data" / "index"
    dir_idx.mkdir(parents=True, exist_ok=True)
    with open(dir_idx / "fragmentos.jsonl", "w", encoding="utf-8") as f:
        for fr in fragmentos:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")
    if args.solo_fragmentos:
        largos = sorted(f["metadatos"]["caracteres"] for f in fragmentos)
        print(f"{len(fragmentos)} fragmentos; mediana {largos[len(largos) // 2]} y máximo {largos[-1]} caracteres")
        return

    cliente = ollama.Client(host=config.OLLAMA_URL)
    db = chromadb.PersistentClient(path=str(config.DIR_CHROMA))
    if config.COLECCION in [c.name for c in db.list_collections()]:
        db.delete_collection(config.COLECCION)
    col = db.create_collection(config.COLECCION, metadata={"hnsw:space": "cosine", "modelo_embeddings": config.MODELO_EMBEDDINGS})

    t0 = time.perf_counter()
    lote = 32
    for i in range(0, len(fragmentos), lote):
        parte = fragmentos[i:i + lote]
        col.add(
            ids=[f["id"] for f in parte],
            documents=[f["texto"] for f in parte],
            embeddings=embeber(cliente, [f["texto"] for f in parte]),
            metadatas=[f["metadatos"] for f in parte],
        )
    dt = time.perf_counter() - t0

    largos = sorted(f["metadatos"]["caracteres"] for f in fragmentos)
    resumen = {
        "fragmentos": len(fragmentos),
        "documentos": len(inventario),
        "chunk_max_chars": config.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
        "caracteres_mediana": largos[len(largos) // 2],
        "caracteres_max": largos[-1],
        "modelo_embeddings": config.MODELO_EMBEDDINGS,
        "segundos_embeddings": round(dt, 1),
    }
    (dir_idx / "resumen_indice.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
