"""Las 16 secciones de una FDS y su detección en el texto extraído.

Formas de encabezado reconocidas:
  - "SECCIÓN N: TÍTULO" (SGA, Decreto 1496 de 2018): el número es la sección canónica.
  - "N. TÍTULO" en mayúsculas (fichas antiguas de TECNIDECOR, 2018, con orden ANSI donde la 2
    es Composición y la 3 es Riesgos): la sección canónica se decide por el título.
  - Solo el título, sin número (p. ej. "Primeros auxilios:" en FDS 2): se acepta únicamente si
    la línea entera es un título de sección conocido.
"""
import re

from rag_fds.utils import sin_tildes

SECCIONES = {
    1: "Identificación del producto y del proveedor",
    2: "Identificación del peligro o peligros",
    3: "Composición / información sobre los componentes",
    4: "Primeros auxilios",
    5: "Medidas de lucha contra incendios",
    6: "Medidas en caso de vertido accidental",
    7: "Manipulación y almacenamiento",
    8: "Controles de exposición / protección personal",
    9: "Propiedades físicas y químicas",
    10: "Estabilidad y reactividad",
    11: "Información toxicológica",
    12: "Información ecotoxicológica",
    13: "Consideraciones sobre la disposición final",
    14: "Información relativa al transporte",
    15: "Información sobre la reglamentación",
    16: "Otra información",
}

# Se evalúan en este orden y gana la primera coincidencia (12 va antes que 11 porque
# "ecotoxicológica" contiene "toxicológ"; 1 va de último porque "identificación" es genérico).
_PALABRAS = [
    (2, r"peligro|riesgo"),
    (4, r"primeros\s+auxilios"),
    (5, r"incendio|extincion"),
    (6, r"vertido|derrame|accidental|fuga"),
    (7, r"manipulacion|almacenamiento|manejo"),
    (8, r"exposicion|proteccion\s+personal"),
    (9, r"propiedades\s+fisic|fisicas\s+y\s+quimicas|fisicoquimic"),
    (10, r"estabilidad|reactividad"),
    (12, r"ecolog|ecotox|ambiental"),
    (11, r"toxicolog"),
    (13, r"disposicion|eliminacion|desecho|residuo"),
    (14, r"transporte"),
    (15, r"reglament|regulat|normativ|legislac"),
    (16, r"otra(s)?\s+informacion"),
    (3, r"composicion|ingrediente|componentes"),
    (1, r"identificacion|identificador"),
]

# Títulos que se aceptan solos, sin número, si ocupan la línea completa.
_TITULOS_SOLOS = [
    (4, r"primeros auxilios"),
    (5, r"(medidas de )?(lucha|combate) contra (el |los )?incendios?|combate de incendios"),
    (6, r"(medidas en caso de )?(vertido|derrame|liberacion|liberaciones) accidental(es)?"),
    (7, r"(manipulacion|manejo) y almacenamiento"),
    (8, r"controles? de exposicion( ?/ ?proteccion personal)?"),
    (9, r"propiedades fisicas y quimicas"),
    (10, r"estabilidad y reactividad"),
    (12, r"informacion (eco ?toxicologica|ecologica)"),
    (11, r"informacion toxicologica"),
    (13, r"consideraciones (sobre|relativas a|de) (la )?(disposicion|eliminacion)( final)?"),
    (14, r"informacion (relativa al|sobre el|de|para el) transporte"),
    (15, r"informacion (sobre la )?reglament(aria|acion)"),
    (16, r"otras? informacion(es)?"),
]

_RE_SECCION = re.compile(r"^secci[oó]n\s*(\d{1,2})\s*(?:[:.\-–—)]\s*|\s+|$)(.*)$", re.I)
_RE_NUMERADA = re.compile(r"^(\d{1,2})\s*[.)\-–]\s*(\S.*)$")
RE_PAGINA = re.compile(r"^<!-- pagina: (\d+) -->$")
# Bloque final con encabezados y pies de página; cada línea es "- [p. 1-3, 5] texto".
MARCA_PIES = "<!-- encabezados y pies de pagina -->"
RE_LINEA_PIE = re.compile(r"^- \[p\. ([\d, \-]+)\] (.*)$")


def paginas_de_lista(texto: str) -> list[int]:
    """'1-3, 5' -> [1, 2, 3, 5]"""
    out = []
    for parte in texto.split(","):
        a, _, b = parte.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def lista_de_paginas(paginas: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3, 5'"""
    rangos, ps = [], sorted(set(paginas))
    for p in ps:
        if rangos and p == rangos[-1][1] + 1:
            rangos[-1][1] = p
        else:
            rangos.append([p, p])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in rangos)


def canonica_por_titulo(titulo: str) -> int | None:
    t = sin_tildes(titulo)
    for n, patron in _PALABRAS:
        if re.search(patron, t):
            return n
    return None


def limpiar_linea(linea: str) -> str:
    s = linea.strip()
    if s.startswith("|"):
        # Fila de tabla: solo cuenta la primera celda no vacía (Docling a veces mete el título
        # de la sección en el encabezado de una tabla, repetido en cada columna).
        celdas = [c.strip() for c in s.strip("|").split("|")]
        s = next((c for c in celdas if c), "")
    s = re.sub(r"^#+\s*", "", s)
    s = s.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", s).strip()


def detectar_encabezado(linea: str) -> tuple[int, int, str, str] | None:
    """Si la línea es un encabezado de sección devuelve (número en el documento, sección canónica, título, forma)."""
    s = limpiar_linea(linea)
    if not s or len(s) > 160:
        return None
    m = _RE_SECCION.match(s)
    if m:
        n = int(m.group(1))
        return (n, n, m.group(2).strip(), "SECCIÓN N") if 1 <= n <= 16 else None
    m = _RE_NUMERADA.match(s)
    if m:
        n, titulo = int(m.group(1)), m.group(2).strip()
        if not 1 <= n <= 16 or len(titulo) < 6:
            return None
        letras = [c for c in titulo if c.isalpha()]
        if not letras or sum(c.isupper() for c in letras) / len(letras) < 0.8:
            return None
        canon = canonica_por_titulo(titulo)
        return (n, canon, titulo, "N. TÍTULO") if canon else None
    if len(s) <= 60:
        t = sin_tildes(s).rstrip(" :.")
        for n, patron in _TITULOS_SOLOS:
            if re.fullmatch(patron, t):
                return n, n, s.rstrip(" :"), "solo título"
    return None


class SeguidorSecciones:
    """Recorre líneas en orden y lleva la sección actual.

    Solo acepta un encabezado si su número en el documento no retrocede respecto al último
    aceptado; así se descartan listas numeradas en mayúsculas que parecen encabezados.
    Un encabezado de solo título se acepta únicamente si es la sección siguiente a la actual:
    subtítulos como "Otra información" dentro de la sección 11 (plantilla PPG) no saltan a la 16.
    """

    def __init__(self):
        self.numero_doc = 0
        self.canonica: int | None = None
        self.titulo = ""
        self.forma = ""

    def procesar(self, linea: str) -> bool:
        enc = detectar_encabezado(linea)
        if not enc:
            return False
        n_doc, canon, titulo, forma = enc
        if n_doc < self.numero_doc or (forma == "solo título" and n_doc != self.numero_doc + 1):
            return False
        self.numero_doc, self.canonica, self.titulo, self.forma = n_doc, canon, titulo, forma
        return True


def encabezados(lineas_con_pagina) -> list[dict]:
    """Encabezados de sección aceptados, a partir de pares (página, línea) en orden de lectura."""
    seg = SeguidorSecciones()
    out = []
    for pagina, linea in lineas_con_pagina:
        if seg.procesar(linea):
            out.append({"canonica": seg.canonica, "numero_doc": seg.numero_doc, "titulo": seg.titulo,
                        "forma": seg.forma, "pagina": pagina, "linea": limpiar_linea(linea)})
    return out


def lineas_markdown(md: str, incluir_pies: bool = False):
    """Pares (página, línea) del cuerpo de un Markdown con marcas <!-- pagina: N -->.

    Con incluir_pies=True también entrega el texto del bloque de encabezados y pies de página,
    una vez por cada página donde aparece.
    """
    pagina = None
    en_pies = False
    for linea in md.splitlines():
        if linea.strip() == MARCA_PIES:
            en_pies = True
            continue
        if en_pies:
            m = RE_LINEA_PIE.match(linea)
            if incluir_pies and m:
                for p in paginas_de_lista(m.group(1)):
                    yield p, m.group(2)
            continue
        m = RE_PAGINA.match(linea.strip())
        if m:
            pagina = int(m.group(1))
        else:
            yield pagina, linea


def recorrer_markdown(md: str):
    """Genera (número de línea, página, sección canónica, número en el documento, línea)."""
    pagina = None
    seg = SeguidorSecciones()
    for i, linea in enumerate(md.splitlines()):
        if linea.strip() == MARCA_PIES:
            break
        m = RE_PAGINA.match(linea.strip())
        if m:
            pagina = int(m.group(1))
            continue
        seg.procesar(linea)
        yield i, pagina, seg.canonica, seg.numero_doc, linea
