"""Palabras de la capa de texto de un PDF con su posición (pypdfium2)."""
from dataclasses import dataclass


@dataclass
class Palabra:
    texto: str
    l: float   # coordenadas con origen arriba-izquierda, en puntos
    t: float
    r: float
    b: float

    @property
    def cx(self) -> float:
        return (self.l + self.r) / 2

    @property
    def cy(self) -> float:
        return (self.t + self.b) / 2


def palabras_de_pagina(page) -> list[Palabra]:
    """Agrupa los caracteres de la página en palabras separadas por espacios o saltos de línea."""
    tp = page.get_textpage()
    alto = page.get_height()
    palabras, letras, cajas = [], [], []

    def cerrar():
        if letras:
            palabras.append(Palabra("".join(letras), min(c[0] for c in cajas), min(c[1] for c in cajas),
                                    max(c[2] for c in cajas), max(c[3] for c in cajas)))
        letras.clear()
        cajas.clear()

    for i in range(tp.count_chars()):
        c = tp.get_text_range(i, 1)
        if not c or c.isspace():
            cerrar()
            continue
        x0, y0, x1, y1 = tp.get_charbox(i)
        letras.append(c)
        cajas.append((x0, alto - y1, x1, alto - y0))
    cerrar()
    return palabras


def palabras_en_area(palabras: list[Palabra], l: float, t: float, r: float, b: float, margen: float = 2.0) -> list[Palabra]:
    return [p for p in palabras if l - margen <= p.cx <= r + margen and t - margen <= p.cy <= b + margen]


def renglones(palabras: list[Palabra], tolerancia: float = 1.0) -> list[list[Palabra]]:
    """Agrupa palabras en renglones y las ordena de izquierda a derecha.

    Una palabra entra en el renglón actual si su centro vertical cae dentro de la franja del
    renglón; así un guion o un signo, cuya caja es más baja que la de las letras, no abre otro.
    """
    out: list[list[Palabra]] = []
    franja = (0.0, 0.0)
    for p in sorted(palabras, key=lambda p: (p.cy, p.l)):
        if out and franja[0] - tolerancia <= p.cy <= franja[1] + tolerancia:
            out[-1].append(p)
            franja = (min(franja[0], p.t), max(franja[1], p.b))
        else:
            out.append([p])
            franja = (p.t, p.b)
    return [sorted(r, key=lambda p: p.l) for r in out]
