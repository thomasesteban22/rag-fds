"""Paso 1: PDF -> Markdown con Docling, imágenes enlazadas y OCR de las imágenes.

Uso:
    uv run python -m rag_fds.extract            # todo el corpus (salta los ya extraídos)
    uv run python -m rag_fds.extract --force    # vuelve a extraer todo
    uv run python -m rag_fds.extract --solo FDS029 FDS083

Salida por ficha en data/md/<doc_id>/:
    <doc_id>.md         Markdown con marcas <!-- pagina: N --> y enlaces relativos a images/
    images/             imágenes únicas de la ficha (PNG)
    tablas.json         tablas detectadas: página, bbox, sección y Markdown de cada una
Y en data/metadata/: imagenes.csv (una fila por aparición de imagen) y extraccion_log.csv.
"""
import argparse
import csv
import gzip
import hashlib
import json
import logging
import platform
import re
import shutil
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import psutil
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer, ImageRefMode, PictureItem

from rag_fds import config
from rag_fds.kv_fix import reemparejar_etiquetas
from rag_fds.sections import MARCA_PIES, RE_PAGINA, SeguidorSecciones, lista_de_paginas
from rag_fds.tablas_fix import reparar_documento
from rag_fds.utils import doc_id_de, listar_pdfs, normalizar_nombre, sha256_archivo

log = logging.getLogger("extract")

CLASES_ES = {
    "icon": "ícono o pictograma",
    "logo": "logo",
    "stamp": "sello",
    "signature": "firma",
    "bar_chart": "gráfico",
    "pie_chart": "gráfico",
    "line_chart": "gráfico",
    "flow_chart": "diagrama",
    "table": "tabla como imagen",
    "picture_group": "grupo de imágenes",
}
_CJK = re.compile(r"[　-鿿가-힯＀-￯]")
_RE_IMG = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_RE_PAGINACION = re.compile(r"(?i)(p[áa]g(ina)?\.?\s*)?\d+\s*(de|/)\s*\d+|\d+")
# Título "SECCIÓN N: TÍTULO EN MAYÚSCULAS" pegado al texto anterior y al siguiente. Pasa en
# FDS100 y FDS101: la marca de agua "COPIA CONTROLADA" hace que Docling junte la página entera
# en un solo párrafo. Solo se agregan saltos de línea; el texto no cambia.
_RE_SECCION_INCRUSTADA = re.compile(
    r"(?<=[^\s#|])\s+(SECCI[OÓ]N\s+\d{1,2}\s*:\s*[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ/(),\- ]*?[A-ZÁÉÍÓÚÜÑ)])"
    r"(?:\s+(?=\d{1,2}\.\d|[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ])|\s*$)"
)


def crear_convertidor() -> DocumentConverter:
    opts = PdfPipelineOptions()
    # Las 101 FDS tienen capa de texto: el OCR de página solo agregaría tiempo (65 de 76 s
    # en la prueba con FDS 29). El OCR se aplica aparte a las imágenes extraídas.
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    opts.generate_picture_images = True
    opts.images_scale = config.IMAGES_SCALE
    opts.do_picture_classification = True
    opts.accelerator_options = AcceleratorOptions(num_threads=config.DOCLING_THREADS, device="cpu")
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


class OCRImagenes:
    """RapidOCR (PP-OCRv6 multilingüe, ONNX en CPU) con caché por contenido de la imagen."""

    def __init__(self):
        from rapidocr import RapidOCR

        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        self.motor = RapidOCR()
        self.cache: dict[str, list[str]] = {}

    def __call__(self, ruta: Path, clave: str) -> list[str]:
        if clave not in self.cache:
            r = self.motor(str(ruta))
            textos = []
            for t, s in zip(r.txts or (), r.scores or ()):
                t = t.strip()
                # Sobre pictogramas el modelo a veces "lee" caracteres chinos sueltos: se descartan.
                if t and s >= config.OCR_MIN_SCORE and not _CJK.search(t):
                    textos.append(t)
            self.cache[clave] = textos
        return self.cache[clave]


def _clasificacion(pic: PictureItem) -> tuple[str, float | None]:
    meta = getattr(pic, "meta", None)
    clas = getattr(meta, "classification", None) if meta else None
    if clas and clas.predictions:
        mejor = clas.predictions[0]
        return mejor.class_name, round(float(mejor.confidence), 3)
    return "", None


def _bbox_tl(item, doc):
    """bbox de la primera procedencia, con origen arriba-izquierda, en puntos PDF."""
    if not item.prov:
        return None, None
    pv = item.prov[0]
    alto = doc.pages[pv.page_no].size.height
    return pv.page_no, pv.bbox.to_top_left_origin(page_height=alto)


def _tabla_asociada(pic, doc, tablas_por_pagina):
    """Asocia la imagen a una tabla solo si su área cae, al menos en un 50 %, dentro de la tabla."""
    pagina, bb = _bbox_tl(pic, doc)
    if bb is None or bb.area() <= 0:
        return "", ""
    for ref, bt in tablas_por_pagina.get(pagina, []):
        frac = bb.intersection_area_with(bt) / bb.area()
        if frac >= 0.5:
            return ref, f"imagen contenida en el área de la tabla en la página {pagina} ({frac:.0%} de su área)"
    return "", ""


def _texto_de_imagen(pic: PictureItem, doc) -> str:
    """Texto de la capa del PDF que Docling agrupó dentro del área de la imagen (no es OCR).

    Docling lo guarda como hijos de la imagen y no lo lleva al Markdown. Pasa con rótulos de
    pictogramas de EPP ("Protección obligatoria de las vías respiratorias") y con tablas que el
    modelo de diseño tomó por imagen (protección respiratoria de FDS055).
    """
    partes = []
    for hijo, _ in doc.iterate_items(root=pic, traverse_pictures=True, with_groups=False):
        texto = getattr(hijo, "text", "").strip() if hijo is not pic else ""
        if texto:
            partes.append(re.sub(r"\s+", " ", texto))
    return " | ".join(partes)


def _bloque_pies(doc) -> tuple[str, int]:
    """Encabezados y pies de página ("furniture" para Docling), sin repetir, con sus páginas.

    Docling no los lleva al cuerpo del Markdown, pero ahí suelen estar la versión, las fechas
    de emisión y revisión y la nota legal. Se omiten los números de página sueltos.
    """
    vistos: dict[str, list[int]] = {}
    items = [t for t in doc.texts if t.content_layer == ContentLayer.FURNITURE and t.prov]
    items += [t for t in doc.tables if t.content_layer == ContentLayer.FURNITURE and t.prov]
    items.sort(key=lambda it: (it.prov[0].page_no, -it.prov[0].bbox.t))
    for it in items:
        if isinstance(it, PictureItem):
            continue
        texto = it.export_to_markdown(doc) if it.label.value == "table" else it.text
        texto = re.sub(r"\s+", " ", texto).strip()
        if not texto or _RE_PAGINACION.fullmatch(texto):
            continue
        paginas = vistos.setdefault(texto, [])
        if it.prov[0].page_no not in paginas:
            paginas.append(it.prov[0].page_no)
    if not vistos:
        return "", 0
    lineas = [
        MARCA_PIES, "",
        "## Encabezados y pies de página", "",
        "*Texto del margen superior o inferior de las páginas, que Docling separa del cuerpo. "
        "Se lista una sola vez, con las páginas donde aparece.*", "",
    ]
    lineas += [f"- [p. {lista_de_paginas(ps)}] {t}" for t, ps in vistos.items()]
    return "\n".join(lineas) + "\n", len(vistos)


def _separar_secciones_incrustadas(texto: str) -> tuple[str, int]:
    """Pone en su propio renglón los títulos de sección que Docling dejó dentro de un párrafo."""
    n = 0
    lineas = []
    for linea in texto.splitlines():
        if not linea.startswith("|"):
            linea, k = _RE_SECCION_INCRUSTADA.subn(lambda m: f"\n\n{m.group(1)}\n\n", linea)
            n += k
        lineas.append(linea)
    return "\n".join(lineas), n


def extraer_documento(pdf: Path, conv: DocumentConverter, ocr: OCRImagenes) -> tuple[dict, list[dict]]:
    doc_id = doc_id_de(pdf.name)
    dir_doc = config.DIR_MD / doc_id
    dir_img = dir_doc / "images"
    if dir_doc.exists():
        shutil.rmtree(dir_doc)
    dir_img.mkdir(parents=True)

    t0 = time.perf_counter()
    res = conv.convert(pdf)
    doc = res.document
    t_docling = time.perf_counter() - t0
    bloques_kv = reemparejar_etiquetas(doc)

    # Tablas por página (para asociar imágenes con evidencia espacial).
    tablas_por_pagina: dict[int, list] = {}
    for t in doc.tables:
        pagina, bt = _bbox_tl(t, doc)
        if bt is not None:
            tablas_por_pagina.setdefault(pagina, []).append((t.self_ref, bt))

    # Imágenes: se guardan una vez por contenido y cada aparición se enlaza al archivo único.
    vistas: dict[str, str] = {}
    apariciones: list[dict] = []
    partes = []
    titulos_separados = 0
    t1 = time.perf_counter()
    for p in range(1, doc.num_pages() + 1):
        for item, _ in doc.iterate_items(page_no=p):
            if not isinstance(item, PictureItem):
                continue
            img = item.get_image(doc)
            if img is None:
                continue
            h = hashlib.sha1(img.tobytes() + repr(img.size).encode()).hexdigest()
            primera = h not in vistas
            if primera:
                vistas[h] = f"{doc_id}_p{p:02d}_img{len(vistas) + 1:02d}.png"
                img.save(dir_img / vistas[h])
            nombre = vistas[h]
            item.image.uri = Path("images") / nombre
            clase, conf = _clasificacion(item)
            tabla, evidencia = _tabla_asociada(item, doc, tablas_por_pagina)
            _, bb = _bbox_tl(item, doc)
            if min(img.size) < config.OCR_MIN_LADO_PX:
                textos_ocr, estado_ocr = [], f"omitido (lado menor a {config.OCR_MIN_LADO_PX} px)"
            else:
                textos_ocr = ocr(dir_img / nombre, h)
                estado_ocr = "con texto" if textos_ocr else "sin texto legible"
            apariciones.append({
                "doc_id": doc_id,
                "pagina": p,
                "seccion": "",
                "archivo": f"images/{nombre}",
                "primera_aparicion": primera,
                "hash": h[:12],
                "ancho_px": img.width,
                "alto_px": img.height,
                "bbox_pt": f"{bb.l:.0f},{bb.t:.0f},{bb.r:.0f},{bb.b:.0f}" if bb else "",
                "clase_docling": clase,
                "confianza_clase": conf,
                "ocr_estado": estado_ocr,
                "ocr": " | ".join(textos_ocr),
                "texto_pdf": _texto_de_imagen(item, doc),
                "leyenda": " ".join(c.resolve(doc).text for c in item.captions) if item.captions else "",
                "tabla_asociada": tabla,
                "evidencia_tabla": evidencia,
            })
        md_p = doc.export_to_markdown(
            page_no=p,
            image_mode=ImageRefMode.REFERENCED,
            escape_html=False,
            escape_underscores=False,
            include_picture_classification=False,
        )
        md_p, k = _separar_secciones_incrustadas(md_p)
        titulos_separados += k
        partes.append(f"<!-- pagina: {p} -->\n\n{md_p.strip()}\n")
    t_img = time.perf_counter() - t1

    # Se recorre el Markdown final para saber en qué sección cae cada imagen y enriquecer el enlace.
    md = "\n".join(partes)
    seg = SeguidorSecciones()
    pendientes = iter(apariciones)
    salida = []
    notas: list[str] = []

    def _enlace(m: re.Match) -> str:
        ap = next(pendientes)
        if ap["archivo"] != m.group(1):
            raise RuntimeError(f"{doc_id}: orden de imágenes inesperado ({ap['archivo']} vs {m.group(1)})")
        ap["seccion"] = seg.canonica or ""
        tipo = CLASES_ES.get(ap["clase_docling"], ap["clase_docling"] or "imagen")
        alt = f"Imagen ({tipo}), página {ap['pagina']}" + (f", sección {ap['seccion']}" if ap["seccion"] else "")
        # Si el PDF tiene texto en el área de la imagen se usa ese; el OCR queda solo en imagenes.csv.
        if ap["texto_pdf"]:
            notas.append(f"*Texto del PDF en el área de la imagen (capa de texto, no OCR): {ap['texto_pdf']}*")
        elif ap["ocr"]:
            notas.append(f"*Texto OCR de la imagen (puede contener errores): {ap['ocr']}*")
        if ap["tabla_asociada"]:
            notas.append(f"*Imagen asociada a la tabla {ap['tabla_asociada']}: {ap['evidencia_tabla']}.*")
        return f"![{alt}]({ap['archivo']})"

    for linea in md.splitlines():
        if not RE_PAGINA.match(linea.strip()):
            seg.procesar(linea)
        notas.clear()
        # Solo se reemplaza el enlace: una imagen puede estar dentro de una celda de tabla
        # (pictogramas de primeros auxilios en FDS099) y el resto del renglón es la fila.
        linea = _RE_IMG.sub(_enlace, linea)
        # Dentro de una fila de tabla no caben párrafos: ahí las notas quedan solo en imagenes.csv.
        if notas and not linea.lstrip().startswith("|"):
            linea += "\n\n" + "\n\n".join(notas)
        salida.append(linea)

    cabecera = (
        "---\n"
        f"doc_id: {doc_id}\n"
        f"archivo_original: \"{pdf.name}\"\n"
        f"archivo_normalizado: \"{normalizar_nombre(pdf.name)}\"\n"
        f"carpeta_origen: \"{pdf.parent.name}\"\n"
        f"sha256: {sha256_archivo(pdf)}\n"
        f"paginas: {doc.num_pages()}\n"
        f"extraccion: \"docling {version('docling')}, sin OCR de página; OCR RapidOCR {version('rapidocr')} en imágenes\"\n"
        f"fecha_extraccion: {date.today().isoformat()}\n"
        "---\n\n"
    )
    pies, n_pies = _bloque_pies(doc)
    (dir_doc / f"{doc_id}.md").write_text(cabecera + "\n".join(salida) + "\n" + pies, encoding="utf-8")

    # Tablas con su página, bbox y sección, para la verificación contra el PDF.
    tablas = []
    seccion_por_tabla = _seccion_de_tablas(doc)
    for t in doc.tables:
        pagina, bt = _bbox_tl(t, doc)
        tablas.append({
            "ref": t.self_ref,
            "pagina": pagina,
            "bbox_pt": [bt.l, bt.t, bt.r, bt.b] if bt else None,
            "filas": t.data.num_rows,
            "columnas": t.data.num_cols,
            "seccion": seccion_por_tabla.get(t.self_ref),
            "markdown": t.export_to_markdown(doc),
        })
    (dir_doc / "tablas.json").write_text(json.dumps(tablas, ensure_ascii=False, indent=1), encoding="utf-8")

    config.DIR_DOCLING.mkdir(parents=True, exist_ok=True)
    with gzip.open(config.DIR_DOCLING / f"{doc_id}.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(doc.export_to_dict(), ensure_ascii=False))
    tablas_reparadas = reparar_documento(doc_id, pdf)

    registro = {
        "doc_id": doc_id,
        "archivo": pdf.name,
        "estado": res.status.name,
        "errores": "; ".join(str(e.error_message) for e in res.errors),
        "paginas": doc.num_pages(),
        "tablas": len(doc.tables),
        "imagenes": len(apariciones),
        "imagenes_unicas": len(vistas),
        "bloques_etiqueta_valor": bloques_kv,
        "textos_encabezado_pie": n_pies,
        "titulos_separados": titulos_separados,
        "tablas_reconstruidas": sum(r["estado"] == "reconstruida" for r in tablas_reparadas),
        "seg_docling": round(t_docling, 1),
        "seg_imagenes_ocr": round(t_img, 1),
        "ram_proceso_mb": round(psutil.Process().memory_info().rss / 2**20),
    }
    return registro, apariciones


def _seccion_de_tablas(doc) -> dict[str, int | None]:
    """Sección canónica de cada tabla según el orden de lectura del documento."""
    seg = SeguidorSecciones()
    out = {}
    for item, _ in doc.iterate_items():
        if item.label.value == "table":
            out[item.self_ref] = seg.canonica
        elif hasattr(item, "text"):
            for linea in _separar_secciones_incrustadas(item.text)[0].splitlines():
                seg.procesar(linea)
    return out


def _escribir_csv(ruta: Path, filas: list[dict]):
    if not filas:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for fila in filas for k in fila))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def _leer_csv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-extrae aunque ya exista el .md")
    ap.add_argument("--solo", nargs="*", help="lista de doc_id a procesar (p. ej. FDS029)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("docling").setLevel(logging.WARNING)

    pdfs = listar_pdfs()
    if args.solo:
        pdfs = [p for p in pdfs if doc_id_de(p.name) in set(args.solo)]

    ruta_log = config.DIR_META / "extraccion_log.csv"
    ruta_img = config.DIR_META / "imagenes.csv"
    registros = {r["doc_id"]: r for r in _leer_csv(ruta_log)}
    imagenes = [r for r in _leer_csv(ruta_img)]

    conv = crear_convertidor()
    ocr = OCRImagenes()
    t_total = time.perf_counter()
    procesadas = 0
    for i, pdf in enumerate(pdfs, 1):
        doc_id = doc_id_de(pdf.name)
        if not args.force and (config.DIR_MD / doc_id / f"{doc_id}.md").exists() and doc_id in registros:
            continue
        procesadas += 1
        try:
            reg, aps = extraer_documento(pdf, conv, ocr)
        except Exception as e:  # se registra el fallo y se sigue con las demás fichas
            log.exception("Fallo en %s", doc_id)
            reg, aps = {"doc_id": doc_id, "archivo": pdf.name, "estado": "FALLO", "errores": repr(e)}, []
        registros[doc_id] = reg
        imagenes = [r for r in imagenes if r["doc_id"] != doc_id] + aps
        _escribir_csv(ruta_log, sorted(registros.values(), key=lambda r: r["doc_id"]))
        _escribir_csv(ruta_img, sorted(imagenes, key=lambda r: (r["doc_id"], int(r["pagina"]))))
        log.info("[%d/%d] %s %s: %s págs, %s tablas, %s imágenes (%s s, %s MB)", i, len(pdfs), doc_id, reg["estado"],
                 reg.get("paginas"), reg.get("tablas"), reg.get("imagenes"), reg.get("seg_docling"), reg.get("ram_proceso_mb"))

    segundos = round(time.perf_counter() - t_total, 1)
    ruta_resumen = config.DIR_META / "extraccion_resumen.json"
    if args.solo and ruta_resumen.exists():
        # Una re-extracción parcial no reemplaza el resumen de la corrida completa: se anota aparte.
        resumen = json.loads(ruta_resumen.read_text(encoding="utf-8"))
        resumen.setdefault("reextracciones_parciales", []).append(
            {"fecha": date.today().isoformat(), "doc_ids": sorted(args.solo), "segundos": segundos})
    else:
        # Si la corrida retoma una anterior, solo procesa las fichas que faltaban: el tiempo
        # total de extracción por ficha está en extraccion_log.csv (seg_docling + seg_imagenes_ocr).
        resumen = {
            "fecha": date.today().isoformat(),
            "segundos_total": segundos,
            "documentos": len(pdfs),
            "procesadas_en_esta_corrida": procesadas,
            "versiones": {p: version(p) for p in ("docling", "docling-core", "docling-ibm-models", "rapidocr", "torch")},
            "opciones": {"do_ocr_pagina": False, "tableformer": "ACCURATE", "images_scale": config.IMAGES_SCALE,
                         "clasificacion_imagenes": True, "hilos_cpu": config.DOCLING_THREADS,
                         "ocr_min_score": config.OCR_MIN_SCORE},
            "equipo": {"so": platform.platform(), "cpu": platform.processor(), "python": platform.python_version()},
        }
    ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Listo en %.1f s", segundos)


if __name__ == "__main__":
    main()
