"""Paso 2: inventario de procedencia y versión, secciones presentes y cobertura del texto.

Uso:
    uv run python -m rag_fds.inventory

Lee la capa de texto de cada PDF (pypdf) y su Markdown (data/md), y escribe en data/metadata/:
    inventario.csv          una fila por ficha: archivo, hash, fabricante, producto, versión y fechas,
                            cada dato con la página y el texto que lo respalda
    secciones.csv           una fila por ficha y sección (1-16) con su estado
    secciones_matriz.csv    ficha x sección: P presente, A ausente en el original,
                            F fallo de extracción (está en el PDF y no en el Markdown), R revisar
    cobertura_paginas.csv   proporción de palabras del PDF que llegan al Markdown, por página
Además reescribe el encabezado YAML de cada Markdown con fabricante, producto y versión.

Las correcciones hechas a mano, con su evidencia, se leen de data/metadata/correcciones_inventario.csv
y se aplican después de la extracción automática; nunca se completa un dato sin evidencia.
"""
import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from rag_fds import config
from rag_fds.sections import SECCIONES, encabezados, lineas_markdown, recorrer_markdown
from rag_fds.utils import doc_id_de, listar_pdfs, normalizar_nombre, numero_fds, sha256_archivo, sin_tildes

SEP = r"[\s|:]*"  # separadores entre etiqueta y valor en la capa de texto (espacios, barras, dos puntos)

# Fabricante: reglas en orden de prioridad. Cada una exige texto explícito en la ficha.
# AkzoNobel va antes que Pintuco porque sus fichas citan www.pintuco.com y un correo @akzonobel.com.
REGLAS_FABRICANTE = [
    (r"TECNI\s*&?\s*DECOR", "Tecnidecor"),
    (r"Pintubler|BLER®", "Pinturas Bler"),
    (r"QU[IÍ]MICA COSMOS|pintuland\.com", "Pintuland"),
    (r"Invesa S\.A\.|invesa\.com", "Invesa"),
    (r"PPG Industries|@ppg\.com", "PPG"),
    (r"Sika Colombia|sika\.com", "Sika"),
    (r"Axalta Coating Systems", "Axalta"),
    (r"CORLANC|corona\.com\.co|www\.corona\.co", "Corona"),
    (r"akzonobel\.com", "AkzoNobel"),
    (r"Pintuco|pintuco\.com", "Pintuco"),
    (r"IEDISA|graphenstone", "Graphenstone (IEDISA)"),
]
# Razón social del proveedor tal como aparece escrita.
RE_PROVEEDOR = re.compile(
    r"(TECNIDECOR S\.?A\.?S\.?|Pintubler de Colombia S\.A\.|QU[IÍ]MICA COSMOS S\.A\.|Invesa S\.A\."
    r"|PPG Industries [A-Za-z ]+Ltda|Sika Colombia S\.A\.S\.|Axalta Coating Systems[^\n|]*"
    r"|CORLANC S\.A\.S\.|Compañía Global de Pinturas S\.A\.S\.|Industria Española[^\n|]*\(IEDISA\)"
    r"|^\s*Pintuco\s*$)",
    re.M,
)
RE_PRODUCTO = [
    re.compile(r"Identificador SGA del\s*producto" + SEP + r"([^\n|]+)", re.I),
    re.compile(r"Nombre del producto" + SEP + r"([^\n|]+)", re.I),
    re.compile(r"Nombre comercial" + SEP + r"([^\n|]+)", re.I),
    re.compile(r"Identificador del producto" + SEP + r"([^\n|]+)", re.I),
    # Plantilla AkzoNobel/International cuando la etiqueta quedó fusionada con "SDS code": el
    # producto es el renglón que sigue al título de la ficha.
    re.compile(r"^#*\s*FICHA DE DATOS DE SEGURIDAD\s*\n\s*\n#*\s*([^\n|#]+)$", re.M),
]
_FECHA = r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-zé]+\s+\d{4}|[a-z]{3}-\d{2})"
RE_VERSION = re.compile(r"Versi[oó]n" + SEP + r"(\d+(?:[.,]\d+)*)(\s*\(sustituye a \d+\))?", re.I)
RE_EMISION_REVISION = re.compile(r"Fecha de emisi[oó]n\s*/\s*Fecha de revisi[oó]n" + SEP + _FECHA, re.I)
RE_EMISION = re.compile(r"(?:Fecha de )?emisi[oó]n" + SEP + _FECHA, re.I)
RE_REVISION = re.compile(r"(?:Fecha de )?revisi[oó]n" + SEP + r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})", re.I)
RE_ACTUALIZACION = re.compile(r"Fecha de actualizaci[oó]n" + SEP + r"([A-Za-zé]+\s+de\s+\d{4}|\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})", re.I)
NORMAS = {
    "Decreto 1496 de 2018 (Colombia)": r"Decreto 1496 de 2018",
    "Resolución 773 de 2021 (Colombia)": r"Resoluci[oó]n 773 de 2021",
    "REACH 1907/2006 (UE)": r"1907/2006",
    "NOM-018-STPS-2015 (México)": r"NOM-018-STPS-2015",
    "OSHA HCS 29 CFR 1910.1200 (EE. UU.)": r"29 CFR 1910\.1200|OSHA",
    "NTE INEN 2266 (Ecuador)": r"INEN 2266",
}
_LINEA_PROPIA = re.compile(r"^!\[Imagen|^\*Texto OCR de la imagen|^\*Imagen asociada a la tabla")
# El texto del PDF que estaba dentro de una imagen sí cuenta para la cobertura, sin el rótulo de la nota.
_NOTA_TEXTO_PDF = re.compile(r"^\*Texto del PDF en el área de la imagen \(capa de texto, no OCR\): (.*)\*$")
# Números con punto del tipo "14.3" (numeración de subsecciones, a veces valores como "7.5").
_RE_NUMERACION = re.compile(r"(?<![\d.,])(?:1[0-6]|[1-9])\.\d{1,2}(?![\d.,]*\d)")


def _texto_paginas(pdf: Path) -> tuple[list[str], dict]:
    r = PdfReader(pdf)
    return [p.extract_text() or "" for p in r.pages], dict(r.metadata or {})


def _evidencia(texto: str, m: re.Match) -> str:
    return re.sub(r"\s+", " ", texto[m.start():m.end()]).strip()[:200]


def _buscar(fuentes, patron, max_pag=None):
    """Primera coincidencia en una lista de (página, texto). Devuelve (match, página, texto)."""
    for pagina, texto in fuentes:
        if max_pag and pagina > max_pag:
            continue
        m = patron.search(texto) if isinstance(patron, re.Pattern) else re.search(patron, texto, re.I)
        if m:
            return m, pagina, texto
    return None, None, None


def _valor_valido(v: str) -> bool:
    """Descarta valores que en realidad son otra etiqueta (formatos de dos columnas desordenados)."""
    v = v.strip()
    return len(v) >= 3 and not re.match(
        r"(?i)secci[oó]n|n[uú]mero|c[oó]digo|sds code|uso|datos|fecha|versi[oó]n|identificador|nombre|tipo|otros medios|:|\d+$", v)


_MARCAS_ARCHIVO = {"corona", "pintuco", "sika", "invesa", "pintuland", "ppg", "aksonobel", "akzonobel",
                   "international", "pinturas", "bler"}


def _producto_de_archivo(nombre: str) -> str:
    """Producto según el nombre del archivo: los tramos separados por " - " que no son solo marcas.

    La marca puede ir antes o después del producto ("FDS 95 - INTERNATIONAL - AKSONOBEL - SEALER F100
    PARTE A", "FDS 82 - ACRILICA MANTENIMIENTO 13883 ... - INTERNATIONAL - AKSONOBEL"). Un guion sin
    espacios ("HS-P-P-007", "catalizador-duraglass") no separa tramos.
    """
    s = normalizar_nombre(Path(nombre).stem)
    s = re.sub(r"^FDS\s*\d+\s*[-–]?\s*", "", s, flags=re.I)
    tramos = [t.strip(" -_") for t in re.split(r"\s*[-–]\s+|\s+[-–]\s*", s)]
    return " - ".join(t for t in tramos if t and not set(t.lower().split()) <= _MARCAS_ARCHIVO)


def extraer_metadatos(pdf: Path, paginas_pdf: list[str], md: str) -> dict:
    md_paginas: dict[int, list[str]] = {}
    for p, linea in lineas_markdown(md, incluir_pies=True):
        if p is not None and not _LINEA_PROPIA.match(linea):
            md_paginas.setdefault(p, []).append(linea)
    fuentes_md = [(p, "\n".join(ls)) for p, ls in sorted(md_paginas.items())]
    fuentes_pdf = list(enumerate(paginas_pdf, 1))
    fila = {}

    # Fabricante: primeras dos páginas del PDF (incluye encabezados y pies de página).
    fila.update(fabricante="", fabricante_evidencia="", fabricante_pagina="", fabricante_fuente="")
    for patron, nombre in REGLAS_FABRICANTE:
        m, pag, texto = _buscar(fuentes_pdf, patron, max_pag=2)
        if m:
            fila.update(fabricante=nombre, fabricante_evidencia=_evidencia(texto, m),
                        fabricante_pagina=pag, fabricante_fuente="regla sobre el texto de la ficha")
            break
    m, pag, texto = _buscar(fuentes_pdf, RE_PROVEEDOR, max_pag=2)
    fila["proveedor_segun_ficha"] = m.group(1).strip() if m else "no indicado en las 2 primeras páginas"
    fila["proveedor_pagina"] = pag or ""

    # Producto: primero el Markdown (etiquetas ya reemparejadas), luego la capa de texto.
    fila.update(producto="", producto_evidencia="", producto_pagina="", producto_fuente="")
    for fuentes, origen in ((fuentes_md, "Markdown"), (fuentes_pdf, "capa de texto del PDF")):
        for patron in RE_PRODUCTO:
            m, pag, texto = _buscar(fuentes, patron, max_pag=2)
            if m and _valor_valido(m.group(1)):
                valor = re.sub(r"\s+", " ", m.group(1)).strip(" :*")
                fila.update(producto=valor, producto_evidencia=_evidencia(texto, m), producto_pagina=pag,
                            producto_fuente=f"ficha ({origen})")
                break
        if fila["producto"]:
            break
    fila["producto_segun_archivo"] = _producto_de_archivo(pdf.name)
    if not fila["producto"]:
        fila.update(producto=fila["producto_segun_archivo"], producto_fuente="nombre de archivo")

    # Versión y fechas: todo el documento, Markdown y capa de texto (las versiones suelen ir en el pie).
    todas = fuentes_md + fuentes_pdf
    m, pag, texto = _buscar(todas, RE_VERSION)
    fila["version"] = (m.group(1) + (m.group(2) or "")).strip() if m else ""
    fila["version_evidencia"] = _evidencia(texto, m) if m else ""
    fila["version_pagina"] = pag or ""
    fila["version_estado"] = "indicada en la ficha" if m else "no encontrada en el texto de la ficha"
    fechas_ev = []
    for campo, patron in (("fecha_emision", RE_EMISION_REVISION), ("fecha_emision", RE_EMISION),
                          ("fecha_revision", RE_REVISION), ("fecha_actualizacion", RE_ACTUALIZACION)):
        if fila.get(campo):
            continue
        m, pag, texto = _buscar(todas, patron)
        fila[campo] = m.group(1).strip() if m else ""
        if m:
            fechas_ev.append(f"p{pag}: {_evidencia(texto, m)}")
    fila["fechas_evidencia"] = " || ".join(fechas_ev)
    pag1 = " ".join(paginas_pdf[:2])
    fila["marco_normativo"] = "; ".join(n for n, p in NORMAS.items() if re.search(p, pag1, re.I))
    return fila


def estado_secciones(doc_id: str, paginas_pdf: list[str], md: str) -> list[dict]:
    enc_md = encabezados(lineas_markdown(md))
    enc_pdf = encabezados((i, l) for i, t in enumerate(paginas_pdf, 1) for l in t.splitlines())
    texto_pdf = sin_tildes("\n".join(paginas_pdf))
    filas = []
    for n, titulo in SECCIONES.items():
        en_md = [e for e in enc_md if e["canonica"] == n]
        en_pdf = [e for e in enc_pdf if e["canonica"] == n]
        if en_md:
            estado = "presente"
        elif en_pdf:
            estado = "fallo de extracción"
        elif re.search(rf"(?m)^\s*(secci[oó]n\s*)?{n}\s*[.:\-]", texto_pdf, re.I):
            estado = "revisar"  # hay una línea que empieza con el número pero no se reconoció el título
        else:
            estado = "ausente en el original"
        filas.append({
            "doc_id": doc_id,
            "seccion": n,
            "titulo_canonico": titulo,
            "estado": estado,
            "paginas_md": ",".join(sorted({str(e["pagina"]) for e in en_md})),
            "encabezado_md": en_md[0]["linea"][:120] if en_md else "",
            "forma_encabezado": en_md[0]["forma"] if en_md else (en_pdf[0]["forma"] if en_pdf else ""),
            "numero_en_documento": en_md[0]["numero_doc"] if en_md else (en_pdf[0]["numero_doc"] if en_pdf else ""),
            "encabezado_pdf": en_pdf[0]["linea"][:120] if en_pdf else "",
            "pagina_pdf": en_pdf[0]["pagina"] if en_pdf else "",
        })
    return filas


def _tokens(texto: str) -> Counter:
    return Counter(t for t in re.findall(r"[a-zñ0-9]+(?:[.,][0-9]+)?", sin_tildes(texto)) if len(t) >= 2)


def cobertura(doc_id: str, paginas_pdf: list[str], md: str) -> list[dict]:
    md_paginas: dict[int, list[str]] = {}
    for p, linea in lineas_markdown(md, incluir_pies=True):
        if p is None or _LINEA_PROPIA.match(linea):
            continue
        m = _NOTA_TEXTO_PDF.match(linea)
        # Los enlaces de imagen dentro de filas de tabla no son texto del PDF.
        md_paginas.setdefault(p, []).append(m.group(1) if m else _RE_ALT.sub("", linea))
    filas = []
    for i, t in enumerate(paginas_pdf, 1):
        texto_md = "\n".join(md_paginas.get(i, []))
        tp, tm = _tokens(t), _tokens(texto_md)
        comun = sum((tp & tm).values())
        # La numeración de subsecciones pesa poco en la cobertura, pero se pierde fácil (marcadores de lista).
        num_faltante = Counter(_RE_NUMERACION.findall(t)) - Counter(_RE_NUMERACION.findall(texto_md))
        filas.append({
            "doc_id": doc_id, "pagina": i,
            "palabras_pdf": sum(tp.values()), "palabras_md": sum(tm.values()),
            "cobertura": round(comun / sum(tp.values()), 3) if tp else "",
            "precision": round(comun / sum(tm.values()), 3) if tm else "",
            "numeracion_faltante": " ".join(sorted(num_faltante.elements())),
        })
    return filas


def aplicar_correcciones(filas: dict[str, dict]):
    ruta = config.DIR_META / "correcciones_inventario.csv"
    if not ruta.exists():
        return
    with open(ruta, newline="", encoding="utf-8") as f:
        for c in csv.DictReader(f):
            fila = filas.get(c["doc_id"])
            if not fila:
                continue
            campo = c["campo"]
            fila[campo] = c["valor"]
            nota = f"{campo}: {c['motivo']} (p. {c['pagina']}: {c['evidencia']})"
            fila["correcciones_manuales"] = (fila.get("correcciones_manuales", "") + " || " + nota).strip(" |")
            if f"{campo}_evidencia" in fila:
                fila[f"{campo}_evidencia"] = c["evidencia"]
            if f"{campo}_pagina" in fila:
                fila[f"{campo}_pagina"] = c["pagina"]
            if f"{campo}_fuente" in fila:
                fila[f"{campo}_fuente"] = f"revisión manual: {c['motivo']}"
            if campo.startswith("fecha"):
                fila["fechas_evidencia"] = (fila["fechas_evidencia"] + f" || p{c['pagina']} (manual): {c['evidencia']}").strip(" |")


_RE_ALT = re.compile(r"!\[Imagen \(([^)]*)\), página (\d+)(?:, sección \d+)?\]\((images/[^)]+)\)")


def sincronizar_imagenes(doc_id: str, ruta_md: Path, imagenes: list[dict]) -> int:
    """Recalcula la sección de cada imagen con el detector de secciones final.

    Reescribe el texto alternativo de los enlaces en el Markdown y la columna 'seccion' de
    imagenes.csv (las filas de la ficha están en el mismo orden en que aparecen en el Markdown).
    Devuelve cuántas imágenes cambiaron de sección.
    """
    md = ruta_md.read_text(encoding="utf-8")
    lineas = md.split("\n")
    secciones = []
    for i, _, canon, _, linea in recorrer_markdown(md):
        # Puede haber más de un enlace por renglón (imágenes dentro de una fila de tabla).
        for m in _RE_ALT.finditer(linea):
            secciones.append(canon or "")
        lineas[i] = _RE_ALT.sub(
            lambda m: f"![Imagen ({m.group(1)}), página {m.group(2)}"
                      + (f", sección {canon}" if canon else "") + f"]({m.group(3)})", linea)
    ruta_md.write_text("\n".join(lineas), encoding="utf-8")
    filas = [r for r in imagenes if r["doc_id"] == doc_id]
    if len(filas) != len(secciones):
        print(f"{doc_id}: {len(filas)} imágenes en el CSV y {len(secciones)} en el Markdown; no se sincroniza el CSV")
        return 0
    cambios = 0
    for fila, s in zip(filas, secciones):
        cambios += str(fila["seccion"]) != str(s)
        fila["seccion"] = s
    return cambios


CAMPOS_YAML = ["fabricante", "proveedor_segun_ficha", "producto", "version", "fecha_emision",
               "fecha_revision", "fecha_actualizacion", "marco_normativo"]


def actualizar_cabecera(ruta_md: Path, fila: dict):
    md = ruta_md.read_text(encoding="utf-8")
    _, cab, cuerpo = md.split("---\n", 2)
    lineas = [l for l in cab.strip().splitlines() if l.split(":", 1)[0] not in CAMPOS_YAML]
    for c in CAMPOS_YAML:
        valor = str(fila.get(c, "")).replace('"', "'")
        lineas.append(f'{c}: "{valor}"')
    ruta_md.write_text("---\n" + "\n".join(lineas) + "\n---\n" + cuerpo, encoding="utf-8")


def _escribir(ruta: Path, filas: list[dict]):
    campos = list(dict.fromkeys(k for fila in filas for k in fila))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def main():
    inventario, secciones, cober = {}, [], []
    for pdf in listar_pdfs():
        doc_id = doc_id_de(pdf.name)
        ruta_md = config.DIR_MD / doc_id / f"{doc_id}.md"
        if not ruta_md.exists():
            print(f"{doc_id}: falta el Markdown, se omite")
            continue
        md = ruta_md.read_text(encoding="utf-8")
        paginas_pdf, meta = _texto_paginas(pdf)
        fila = {
            "doc_id": doc_id,
            "numero_fds": numero_fds(pdf.name),
            "carpeta_origen": pdf.parent.name,
            "archivo_original": pdf.name,
            "archivo_normalizado": normalizar_nombre(pdf.name),
            "ruta_pdf": pdf.relative_to(config.RAIZ).as_posix(),
            "ruta_md": ruta_md.relative_to(config.RAIZ).as_posix(),
            "sha256": sha256_archivo(pdf),
            "bytes": pdf.stat().st_size,
            "paginas": len(paginas_pdf),
            "pdf_creador": meta.get("/Creator", ""),
            "pdf_productor": meta.get("/Producer", ""),
            "pdf_fecha_creacion": meta.get("/CreationDate", ""),
            "pdf_autor": meta.get("/Author", ""),
        }
        fila.update(extraer_metadatos(pdf, paginas_pdf, md))
        sec = estado_secciones(doc_id, paginas_pdf, md)
        cob = cobertura(doc_id, paginas_pdf, md)
        valores = [c["cobertura"] for c in cob if c["cobertura"] != ""]
        fila["secciones_presentes"] = sum(s["estado"] == "presente" for s in sec)
        fila["secciones_no_presentes"] = ";".join(f"{s['seccion']}:{s['estado']}" for s in sec if s["estado"] != "presente")
        fila["cobertura_media"] = round(sum(valores) / len(valores), 3) if valores else ""
        fila["cobertura_min"] = min(valores) if valores else ""
        fila["pagina_cobertura_min"] = cob[[c["cobertura"] for c in cob].index(min(valores))]["pagina"] if valores else ""
        inventario[doc_id] = fila
        secciones += sec
        cober += cob

    aplicar_correcciones(inventario)
    ruta_img = config.DIR_META / "imagenes.csv"
    with open(ruta_img, newline="", encoding="utf-8") as f:
        imagenes = list(csv.DictReader(f))
    cambios = 0
    for doc_id, fila in inventario.items():
        actualizar_cabecera(config.DIR_MD / doc_id / f"{doc_id}.md", fila)
        cambios += sincronizar_imagenes(doc_id, config.DIR_MD / doc_id / f"{doc_id}.md", imagenes)
    _escribir(ruta_img, imagenes)
    print(f"Imágenes que cambiaron de sección al aplicar el detector final: {cambios}")

    config.DIR_META.mkdir(parents=True, exist_ok=True)
    _escribir(config.DIR_META / "inventario.csv", list(inventario.values()))
    _escribir(config.DIR_META / "secciones.csv", secciones)
    _escribir(config.DIR_META / "cobertura_paginas.csv", cober)
    codigo = {"presente": "P", "ausente en el original": "A", "fallo de extracción": "F", "revisar": "R"}
    matriz = {}
    for s in secciones:
        matriz.setdefault(s["doc_id"], {"doc_id": s["doc_id"]})[f"S{s['seccion']}"] = codigo[s["estado"]]
    _escribir(config.DIR_META / "secciones_matriz.csv", list(matriz.values()))

    estados = Counter(s["estado"] for s in secciones)
    print(f"{len(inventario)} fichas. Secciones: {dict(estados)}")
    sin_fab = [d for d, f in inventario.items() if not f["fabricante"]]
    print("Sin fabricante:", sin_fab or "ninguna")
    print("Sin versión:", [d for d, f in inventario.items() if not f["version"]] or "ninguna")


if __name__ == "__main__":
    main()
