"""Reemparejado de etiquetas y valores en fichas con formato de dos columnas.

En las plantillas "Etiqueta : valor" el modelo de diseño de Docling separa la columna de
etiquetas de la de valores, y el Markdown queda con varias etiquetas seguidas y luego sus
valores. Aquí se vuelven a unir usando la posición vertical de cada bloque en la página: un valor
pertenece a la etiqueta cuya parte superior está a su altura o justo encima.

Se reconocen dos variantes:
  A. Los ":" están en la columna de valores (AkzoNobel/International, Axalta, PPG, Sika 2023+):
     bloques que son solo ":" o que empiezan con ":".
  B. Los ":" van pegados a la etiqueta (plantilla Chemeter de Pintuco, Corona, Invesa,
     Pintuland, Bler y fichas Sika 2019-2020): etiquetas cortas que terminan en ":" con los
     valores a su derecha.

El borde entre columnas se estima con esos ":". Cada bloque queda clasificado como etiqueta (a
la izquierda del borde), valor (a la derecha) o bloque que cruza las dos columnas. Los que cruzan
se dejan intactos y separan los tramos que se reemparejan. Un tramo se toca solo si tiene al
menos dos etiquetas y un valor. El texto no se modifica: solo se reordena y se une.

Docling guarda la numeración de un elemento de lista ("14.3") como marcador, fuera del texto.
Al unir bloques, ese marcador se pasa al comienzo del texto de su etiqueta o valor; si no, se
perdería con el bloque que se borra.

Limitación conocida: si Docling ya juntó en un mismo bloque texto de las dos columnas
(p. ej. una etiqueta con parte de un valor), ese bloque no se puede separar aquí.
"""
import re
import statistics

from docling_core.types.doc import DocItemLabel, ListItem, TextItem

_ETIQUETAS_TEXTO = {DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.PARAGRAPH}
_VINETAS = {"", "-", "·", "•", "*", "–", "o"}
_SOLO_DOS_PUNTOS = re.compile(r"^[:\s]+$")
_EMPIEZA_DOS_PUNTOS = re.compile(r"^\s*:\s*")
TOLERANCIA_PT = 4.0       # desfase vertical admitido entre una etiqueta y su valor
MARGEN_COLUMNA_PT = 3.0   # holgura horizontal alrededor del borde entre columnas
SALTO_RENGLON_PT = 6.0    # distancia máxima para considerar que una etiqueta sigue en el renglón siguiente
LARGO_ETIQUETA = 80       # una etiqueta de la variante B tiene a lo sumo este largo


def _bbox(item, doc):
    pv = item.prov[0]
    return pv.bbox.to_top_left_origin(page_height=doc.pages[pv.page_no].size.height)


def _texto(it) -> str:
    """Texto del bloque con su numeración de lista (p. ej. '14.3'), que Docling guarda aparte."""
    marca = (getattr(it, "marker", "") or "").strip()
    if isinstance(it, ListItem) and marca not in _VINETAS:
        return f"{marca} {it.text}"
    return it.text


def _corridas(doc, pagina):
    """Secuencias de bloques de texto consecutivos (en orden de lectura) de una página."""
    actual = []
    for item, _ in doc.iterate_items(page_no=pagina):
        if isinstance(item, TextItem) and item.label in _ETIQUETAS_TEXTO and item.prov:
            actual.append(item)
        else:
            if actual:
                yield actual
            actual = []
    if actual:
        yield actual


def _borde_columnas(corrida, doc) -> tuple[float, str] | None:
    """Coordenada x del borde entre etiquetas y valores, y la variante, o None si no aplica."""
    marcas = [it for it in corrida if _SOLO_DOS_PUNTOS.match(it.text) or _EMPIEZA_DOS_PUNTOS.match(it.text)]
    if len(marcas) >= 2:
        return statistics.median(_bbox(it, doc).l for it in marcas), "A"
    etiquetas = [it for it in corrida if it.text.strip().endswith(":") and len(it.text.strip()) <= LARGO_ETIQUETA]
    if len(etiquetas) >= 2:
        borde = statistics.median(_bbox(it, doc).r for it in etiquetas)
        # Debe haber al menos dos bloques a la derecha de ese borde (los valores).
        if sum(_bbox(it, doc).l >= borde - MARGEN_COLUMNA_PT for it in corrida) >= 2:
            return borde, "B"
    return None


def _emparejar(etiquetas, valores) -> list[str]:
    etiquetas.sort(key=lambda e: e[0].t)
    asignados: list[list[str]] = [[] for _ in etiquetas]
    sueltos = []
    for bb, it in sorted(valores, key=lambda v: v[0].t):
        idx = None
        for i, (be, _) in enumerate(etiquetas):
            if be.t <= bb.t + TOLERANCIA_PT:
                idx = i
            else:
                break
        texto = _EMPIEZA_DOS_PUNTOS.sub("", _texto(it)).strip()
        (asignados[idx] if idx is not None else sueltos).append(texto)

    # Un renglón de etiqueta sin valor, justo debajo de otro que no termina en ":", es la
    # continuación de esa etiqueta (p. ej. "Clase(s) relativas al" + "transporte:").
    filas = []  # [texto_etiqueta, [valores], bbox, texto_original]
    for (be, et), vals in zip(etiquetas, asignados):
        etiqueta, _, en_linea = _texto(et).partition(":")
        vals = [v for v in [en_linea.strip(), *vals] if v]
        if (filas and not vals and be.t - filas[-1][2].b <= SALTO_RENGLON_PT
                and not filas[-1][3].strip().endswith(":")):
            filas[-1][0] += " " + etiqueta.strip()
            filas[-1][2], filas[-1][3] = be, et.text
            continue
        filas.append([etiqueta.strip(), vals, be, et.text])
    lineas = [s for s in sueltos if s]
    lineas += [f"- **{e}:** {' '.join(v)}".rstrip() for e, v, _, _ in filas]
    return lineas


def _reemparejar(corrida, doc) -> tuple[int, list]:
    """Reescribe los tramos etiqueta-valor de la corrida. Devuelve (tramos cambiados, elementos que sobran)."""
    borde = _borde_columnas(corrida, doc)
    if borde is None:
        return 0, []
    x, _variante = borde
    tramos, actual = [], []
    for it in corrida:
        bb = _bbox(it, doc)
        if _SOLO_DOS_PUNTOS.match(it.text):
            actual.append(("marca", bb, it))
        elif bb.r <= x + MARGEN_COLUMNA_PT or (bb.l < x - MARGEN_COLUMNA_PT and it.text.strip().endswith(":")
                                                and len(it.text.strip()) <= LARGO_ETIQUETA):
            actual.append(("etiqueta", bb, it))
        elif bb.l >= x - MARGEN_COLUMNA_PT:
            actual.append(("valor", bb, it))
        else:  # cruza las dos columnas: se deja intacto y corta el tramo
            if actual:
                tramos.append(actual)
            actual = []
    if actual:
        tramos.append(actual)

    sobrantes, cambiados = [], 0
    for tramo in tramos:
        etiquetas = [(bb, it) for tipo, bb, it in tramo if tipo == "etiqueta"]
        valores = [(bb, it) for tipo, bb, it in tramo if tipo == "valor"]
        if len(etiquetas) < 2 or not valores:
            continue
        lineas = _emparejar(etiquetas, valores)
        primero = tramo[0][2]
        if isinstance(primero, ListItem):
            # Docling ya escribe "- " (y el marcador) antes de un elemento de lista: el marcador
            # quedó dentro del texto, así que se quita de ahí y del primer renglón.
            primero.marker, primero.enumerated = "", False
            if lineas[0].startswith("- "):
                lineas[0] = lineas[0][2:]
        primero.text = "\n".join(lineas)
        sobrantes += [it for _, _, it in tramo[1:]]
        cambiados += 1
    return cambiados, sobrantes


def reemparejar_etiquetas(doc) -> int:
    """Corrige en el documento los bloques etiqueta-valor separados. Devuelve cuántos tramos cambió."""
    n = 0
    for p in range(1, doc.num_pages() + 1):
        sobrantes = []
        for corrida in list(_corridas(doc, p)):
            cambiados, resto = _reemparejar(corrida, doc)
            n += cambiados
            sobrantes += resto
        if sobrantes:
            doc.delete_items(node_items=sobrantes)
    return n
