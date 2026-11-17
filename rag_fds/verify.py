"""Paso 3: verificación de una tabla (o bloque con datos) por ficha contra el PDF original.

Uso:
    uv run python -m rag_fds.verify

Para cada ficha se elige una tabla con datos, por prioridad de sección: 3 (composición),
9 (propiedades), 8 (exposición), 14 (transporte), 11 (toxicología) y luego cualquiera.
Si la ficha no tiene tablas, se usa el bloque de texto de la sección 9.

La comparación usa la capa de texto del PDF dentro del área (bbox) del objeto en la página:
    cobertura  = palabras del PDF en esa área que aparecen en el Markdown
    precisión  = palabras del Markdown que aparecen en el PDF
    números faltantes = cifras, CAS o códigos del PDF que no llegaron al Markdown
Además se guarda un recorte de la página (PNG) para la revisión visual.

Salidas en data/verificacion/:
    verificacion_tablas.csv   métricas y diferencias por ficha
    verificacion_tablas.md    reporte con el recorte del PDF junto al Markdown extraído
    recortes/                 imágenes de las zonas verificadas
    errores_extraccion.csv    registro consolidado de errores (automáticos + revisión manual)
    evidencias_evaluacion.csv la evidencia de cada pregunta de evaluación, buscada en el Markdown
"""
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from rag_fds import config
from rag_fds.sections import SECCIONES, recorrer_markdown
from rag_fds.tablas_fix import es_tabla_fusionada
from rag_fds.utils import doc_id_de, listar_pdfs, sin_tildes

PRIORIDAD_SECCIONES = [3, 9, 8, 14, 11]
MARGEN_PT = 2.0
UMBRAL_COBERTURA_PAGINA = 0.85


def _tokens(texto: str) -> Counter:
    texto = sin_tildes(texto.replace("|", " "))
    return Counter(t for t in re.findall(r"[a-zñ0-9]+(?:[.,\-][0-9]+)*", texto) if len(t) >= 2 or t.isdigit())


def _es_numero(t: str) -> bool:
    return bool(re.search(r"\d", t))


def _elegir_tabla(tablas: list[dict]) -> dict | None:
    utiles = [t for t in tablas if t["bbox_pt"] and t["filas"] >= 2 and t["columnas"] >= 2 and len(t["markdown"]) > 60]
    for s in PRIORIDAD_SECCIONES:
        for t in utiles:
            if t["seccion"] == s:
                return t
    return utiles[0] if utiles else None


def _bloque_seccion(md: str, seccion: int) -> tuple[int | None, str]:
    """Primer bloque de texto (hasta 1500 caracteres) de una sección, con su página."""
    pagina, lineas = None, []
    for _, p, canon, _, linea in recorrer_markdown(md):
        if canon == seccion and linea.strip() and not linea.startswith("![") and not linea.startswith("*Texto OCR"):
            if pagina is None:
                pagina = p
            if p != pagina:
                break
            lineas.append(linea)
            if sum(len(l) for l in lineas) > 1500:
                break
    return pagina, "\n".join(lineas)


def _recortar(pdf, pagina: int, bbox, destino: Path):
    page = pdf[pagina - 1]
    w, h = page.get_width(), page.get_height()
    if bbox is None:
        page.render(scale=1.5).to_pil().save(destino)
        return
    l, t, r, b = bbox
    m = 6
    crop = (max(0, l - m), max(0, h - b - m), max(0, w - r - m), max(0, t - m))
    page.render(scale=2.0, crop=crop).to_pil().save(destino)


def _texto_en_area(pdf, pagina: int, bbox) -> str:
    """Texto de la capa del PDF dentro del área, palabra por palabra.

    Una palabra cuenta solo si al menos el 60 % de sus caracteres cae dentro del área; así no
    entran letras sueltas de los renglones que quedan en el borde de la tabla.
    """
    page = pdf[pagina - 1]
    tp = page.get_textpage()
    if bbox is None:
        return tp.get_text_range()
    h = page.get_height()
    l, t, r, b = bbox
    l, r, arriba, abajo = l - MARGEN_PT, r + MARGEN_PT, h - t + MARGEN_PT, h - b - MARGEN_PT
    palabras, actual, dentro = [], [], 0
    for i in range(tp.count_chars()):
        c = tp.get_text_range(i, 1)
        if not c or c.isspace():
            if actual and dentro / len(actual) >= 0.6:
                palabras.append("".join(actual))
            actual, dentro = [], 0
            continue
        x0, y0, x1, y1 = tp.get_charbox(i)
        actual.append(c)
        dentro += l <= (x0 + x1) / 2 <= r and abajo <= (y0 + y1) / 2 <= arriba
    if actual and dentro / len(actual) >= 0.6:
        palabras.append("".join(actual))
    return " ".join(palabras)


def _bbox_tabla(doc_id: str, ref: str, pagina: int, respaldo):
    """Unión de todas las áreas de la tabla en la página, según el JSON completo de Docling."""
    ruta = config.DIR_DOCLING / f"{doc_id}.json.gz"
    if not ruta.exists():
        return respaldo
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        d = json.load(f)
    tabla = d["tables"][int(ref.split("/")[-1])]
    alto = d["pages"][str(pagina)]["size"]["height"]
    cajas = []
    for pv in tabla["prov"]:
        if pv["page_no"] != pagina:
            continue
        bb = pv["bbox"]
        if bb.get("coord_origin", "BOTTOMLEFT") == "BOTTOMLEFT":
            cajas.append((bb["l"], alto - bb["t"], bb["r"], alto - bb["b"]))
        else:
            cajas.append((bb["l"], bb["t"], bb["r"], bb["b"]))
    if not cajas:
        return respaldo
    return [min(c[0] for c in cajas), min(c[1] for c in cajas), max(c[2] for c in cajas), max(c[3] for c in cajas)]


def verificar_documento(pdf_path: Path) -> dict:
    doc_id = doc_id_de(pdf_path.name)
    dir_doc = config.DIR_MD / doc_id
    tablas = json.loads((dir_doc / "tablas.json").read_text(encoding="utf-8"))
    md = (dir_doc / f"{doc_id}.md").read_text(encoding="utf-8")
    pdf = pdfium.PdfDocument(str(pdf_path))

    tabla = _elegir_tabla(tablas)
    if tabla:
        objeto, ref, seccion, pagina, extraido = "tabla", tabla["ref"], tabla["seccion"], tabla["pagina"], tabla["markdown"]
        bbox = _bbox_tabla(doc_id, ref, pagina, tabla["bbox_pt"])
        filas, columnas = tabla["filas"], tabla["columnas"]
    else:
        seccion = 9
        pagina, extraido = _bloque_seccion(md, 9)
        if pagina is None:
            seccion = 3
            pagina, extraido = _bloque_seccion(md, 3)
        objeto, ref, bbox, filas, columnas = "bloque de texto (sin tablas útiles)", "", None, "", ""

    if pagina is None:
        return {"doc_id": doc_id, "objeto": "ninguno", "estado": "no verificable: sin tabla ni sección 9/3 detectada"}

    original = _texto_en_area(pdf, pagina, bbox)
    # De una tabla se comparan solo sus filas: Docling exporta también la leyenda y las notas
    # al pie asociadas, que en el PDF están fuera del área de la tabla.
    cuerpo = "\n".join(l for l in extraido.splitlines() if l.strip().startswith("|")) if tabla else extraido
    tp, tm = _tokens(original), _tokens(cuerpo)
    comun = tp & tm
    # Precisión sobre palabras únicas: Docling repite el texto de una celda combinada en cada
    # columna que ocupa, y eso no es contenido inventado.
    sobran = Counter({k: 1 for k in tm if k not in tp})
    precision = round(len(set(tm) & set(tp)) / max(1, len(set(tm))), 3)
    if bbox is None:
        # Sin bbox se compara contra la página completa: solo tiene sentido la precisión.
        faltan, num_faltan, cobertura = Counter(), [], ""
    else:
        faltan = tp - tm
        num_faltan = sorted(t for t in faltan if _es_numero(t))
        cobertura = round(sum(comun.values()) / max(1, sum(tp.values())), 3)

    dir_rec = config.DIR_VERIF / "recortes"
    dir_rec.mkdir(parents=True, exist_ok=True)
    nombre_rec = f"{doc_id}_p{pagina:02d}.png"
    _recortar(pdf, pagina, bbox, dir_rec / nombre_rec)

    ok_cobertura = cobertura == "" or cobertura >= 0.98
    estado = "sin diferencias" if ok_cobertura and precision >= 0.98 and not num_faltan else "con diferencias"
    return {
        "doc_id": doc_id,
        "objeto": objeto,
        "ref": ref,
        "seccion": seccion,
        "pagina": pagina,
        "filas": filas,
        "columnas": columnas,
        "palabras_pdf": sum(tp.values()),
        "palabras_md": sum(tm.values()),
        "cobertura": cobertura,
        "precision": precision,
        "numeros_faltantes": " ".join(num_faltan[:20]),
        "palabras_faltantes": " ".join(sorted(faltan)[:25]),
        "palabras_sobrantes": " ".join(sorted(sobran)[:25]),
        "estado": estado,
        "recorte": f"recortes/{nombre_rec}",
        "_extraido": extraido,
    }


def _leer(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _escribir(ruta: Path, filas: list[dict]):
    campos = []
    for f in filas:
        campos += [k for k in f if k not in campos]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def filas_fusionadas(doc_id: str) -> list[dict]:
    """Tablas donde Docling fusionó varias filas en una (ver rag_fds.tablas_fix), reparadas o no."""
    tablas = json.loads((config.DIR_MD / doc_id / "tablas.json").read_text(encoding="utf-8"))
    out = []
    for t in tablas:
        if t.get("reconstruida"):
            out.append({"doc_id": doc_id, "pagina": t["pagina"], "seccion": t["seccion"] or "", "ref": t["ref"],
                        "reparada": True})
        elif es_tabla_fusionada(t["markdown"], t["filas"], t["columnas"]):
            out.append({"doc_id": doc_id, "pagina": t["pagina"], "seccion": t["seccion"] or "", "ref": t["ref"],
                        "reparada": False})
    return out


def calidad_markdown(doc_id: str) -> dict:
    """Indicadores simples de calidad del Markdown de una ficha.

    valores_sin_etiqueta: renglones que empiezan con ':' (en las plantillas de dos columnas,
    un valor que quedó separado de su etiqueta).
    """
    md = (config.DIR_MD / doc_id / f"{doc_id}.md").read_text(encoding="utf-8")
    lineas = [l.strip() for _, _, _, _, l in recorrer_markdown(md)]
    tablas = json.loads((config.DIR_MD / doc_id / "tablas.json").read_text(encoding="utf-8"))
    return {
        "doc_id": doc_id,
        "caracteres": len(md),
        "renglones": sum(1 for l in lineas if l),
        "valores_sin_etiqueta": sum(1 for l in lineas if l.startswith(":")),
        "solo_dos_puntos": sum(1 for l in lineas if l and set(l) <= set(": ")),
        "pares_reemparejados": sum(1 for l in lineas if l.startswith("- **") and ":**" in l),
        "tablas": len(tablas),
        "tablas_reconstruidas": sum(1 for t in tablas if t.get("reconstruida")),
        "imagenes_enlazadas": sum(l.count("](images/") for l in lineas),
        "imagenes_con_texto_pdf": sum(1 for l in lineas if l.startswith("*Texto del PDF en el área de la imagen")),
    }


def registro_errores(verif: list[dict], fusiones: list[dict]) -> list[dict]:
    """Une en un solo registro los errores detectados automáticamente y los de la revisión manual."""
    errores = []
    for s in fusiones:
        errores.append({
            "doc_id": s["doc_id"], "pagina": s["pagina"], "seccion": s["seccion"],
            "tipo": "filas de tabla fusionadas" + (" (corregido)" if s["reparada"] else " (sin corregir)"),
            "detalle": f"tabla {s['ref']}: Docling juntó varias filas en una; el texto estaba completo pero sin "
                       "correspondencia entre filas. " + ("Se reconstruyó por posición de renglones (rag_fds.tablas_fix)."
                                                          if s["reparada"] else "No se pudo reconstruir con validación."),
            "detectado_por": "automático (forma de la tabla)",
        })
    for v in verif:
        if v.get("estado") == "con diferencias":
            errores.append({
                "doc_id": v["doc_id"], "pagina": v["pagina"], "seccion": v["seccion"],
                "tipo": f"diferencias en {v['objeto']} verificada",
                "detalle": f"cobertura {v['cobertura']}, precisión {v['precision']}; números faltantes: "
                           f"{v['numeros_faltantes'] or 'ninguno'}; palabras faltantes: {v['palabras_faltantes'] or 'ninguna'}",
                "detectado_por": "automático (capa de texto en el área de la tabla)",
            })
    for s in _leer(config.DIR_META / "secciones.csv"):
        if s["estado"] in ("fallo de extracción", "revisar"):
            errores.append({
                "doc_id": s["doc_id"], "pagina": s["pagina_pdf"], "seccion": s["seccion"],
                "tipo": f"sección {s['estado']}",
                "detalle": f"encabezado en el PDF: '{s['encabezado_pdf']}'" if s["encabezado_pdf"] else "sin encabezado reconocible",
                "detectado_por": "automático (detector de secciones)",
            })
    for c in _leer(config.DIR_META / "cobertura_paginas.csv"):
        if c["cobertura"] and float(c["cobertura"]) < UMBRAL_COBERTURA_PAGINA:
            errores.append({
                "doc_id": c["doc_id"], "pagina": c["pagina"], "seccion": "",
                "tipo": "texto del PDF no presente en el Markdown",
                "detalle": f"cobertura de palabras de la página {c['cobertura']} ({c['palabras_md']} de {c['palabras_pdf']})",
                "detectado_por": "automático (cobertura por página)",
            })
        if c.get("numeracion_faltante"):
            errores.append({
                "doc_id": c["doc_id"], "pagina": c["pagina"], "seccion": "",
                "tipo": "números con punto del PDF ausentes en el Markdown",
                "detalle": f"en la capa de texto pero no en el Markdown de la página: {c['numeracion_faltante']}",
                "detectado_por": "automático (números del tipo 14.3 por página)",
            })
    for r in _leer(config.DIR_VERIF / "revision_manual.csv"):
        # Solo errores y límites del OCR; las observaciones y los datos ausentes en el original no son errores.
        if r["resultado"].startswith("error") or r["resultado"].startswith("limitación"):
            errores.append({
                "doc_id": r["doc_id"], "pagina": r["pagina"], "seccion": r["seccion"],
                "tipo": f"{r['tipo_error']} ({r['resultado']})", "detalle": r["descripcion"],
                "detectado_por": "revisión manual contra el PDF renderizado",
            })
    return errores


def verificar_evidencias() -> list[dict]:
    """Comprueba que la evidencia de cada pregunta de evaluación esté en el Markdown, en su página y sección.

    Solo lee eval/preguntas.json para esta comprobación: las referencias no pasan al índice.
    """
    ruta = config.DIR_EVAL / "preguntas.json"
    if not ruta.exists():
        return []
    filas = []
    for q in json.loads(ruta.read_text(encoding="utf-8")):
        for ev in q.get("evidencias", []):
            md = (config.DIR_MD / ev["doc_id"] / f"{ev['doc_id']}.md").read_text(encoding="utf-8")
            lineas = [(canon, l) for _, p, canon, _, l in recorrer_markdown(md) if p == ev["pagina"]]
            te = _tokens(ev["texto"])
            tp = _tokens("\n".join(l for _, l in lineas))
            ts = _tokens("\n".join(l for c, l in lineas if c == ev["seccion"]))
            filas.append({
                "id": q["id"], "doc_id": ev["doc_id"], "pagina": ev["pagina"], "seccion": ev["seccion"],
                "cobertura_pagina": round(sum((te & tp).values()) / sum(te.values()), 3),
                "cobertura_seccion": round(sum((te & ts).values()) / sum(te.values()), 3),
                "faltan": " ".join(sorted((te - tp).elements())),
            })
    return filas


def main():
    config.DIR_VERIF.mkdir(parents=True, exist_ok=True)
    verif = []
    for pdf in listar_pdfs():
        if not (config.DIR_MD / doc_id_de(pdf.name) / "tablas.json").exists():
            continue
        verif.append(verificar_documento(pdf))

    reporte = ["# Verificación de una tabla o bloque por ficha", "",
               "Cada entrada muestra el recorte del PDF original y lo que quedó en el Markdown. "
               "Las métricas comparan las palabras de la capa de texto del PDF dentro del área del objeto "
               "con las del Markdown extraído.", ""]
    for v in verif:
        if v["objeto"] == "ninguno":
            reporte += [f"## {v['doc_id']}", "", v["estado"], ""]
            continue
        titulo = SECCIONES.get(v["seccion"], "?")
        reporte += [
            f"## {v['doc_id']} — {v['objeto']}, página {v['pagina']}, sección {v['seccion']} ({titulo})", "",
            f"![Recorte del PDF]({v['recorte']})", "",
            f"Cobertura {v['cobertura']} · precisión {v['precision']} · estado: **{v['estado']}**", "",
        ]
        if v["numeros_faltantes"]:
            reporte += [f"Números del PDF que no aparecen en el Markdown: `{v['numeros_faltantes']}`", ""]
        if v["palabras_faltantes"]:
            reporte += [f"Palabras del PDF que faltan: `{v['palabras_faltantes']}`", ""]
        if v["palabras_sobrantes"]:
            reporte += [f"Palabras del Markdown que no están en el área del PDF: `{v['palabras_sobrantes']}`", ""]
        reporte += ["Markdown extraído:", "", v["_extraido"], ""]
    (config.DIR_VERIF / "verificacion_tablas.md").write_text("\n".join(reporte), encoding="utf-8")

    for v in verif:
        v.pop("_extraido", None)
    _escribir(config.DIR_VERIF / "verificacion_tablas.csv", verif)
    _escribir(config.DIR_VERIF / "calidad_markdown.csv", [calidad_markdown(v["doc_id"]) for v in verif])
    fusiones = [s for v in verif for s in filas_fusionadas(v["doc_id"])]
    errores = registro_errores(verif, fusiones)
    if errores:
        _escribir(config.DIR_VERIF / "errores_extraccion.csv", errores)
    evidencias = verificar_evidencias()
    if evidencias:
        _escribir(config.DIR_VERIF / "evidencias_evaluacion.csv", evidencias)
    estados = Counter(v["estado"] for v in verif)
    print(f"{len(verif)} fichas verificadas: {dict(estados)}. Tablas con filas fusionadas: {len(fusiones)} "
          f"({sum(s['reparada'] for s in fusiones)} reconstruidas). "
          f"Errores registrados: {len(errores)} ({dict(Counter(e['tipo'] for e in errores))})")
    if evidencias:
        completas = sum(e["cobertura_seccion"] >= 0.95 for e in evidencias)
        print(f"Evidencias de evaluación en su página y sección: {completas} de {len(evidencias)}")


if __name__ == "__main__":
    main()
