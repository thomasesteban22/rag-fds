"""Funciones comunes: nombres de archivo, identificadores y hashes."""
import hashlib
import re
import unicodedata
from pathlib import Path

from rag_fds import config


def normalizar_nombre(nombre: str) -> str:
    """Decodifica escapes como #U00ed (í) o #U00d1 (Ñ), normaliza a NFC y colapsa espacios."""
    s = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), nombre)
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def sin_tildes(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar texto sin depender de la acentuación."""
    s = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def numero_fds(nombre: str) -> int:
    m = re.match(r"\s*FDS\s*(\d+)", normalizar_nombre(nombre), re.I)
    if not m:
        raise ValueError(f"No se encontró el número de FDS en: {nombre}")
    return int(m.group(1))


def doc_id_de(nombre: str) -> str:
    return f"FDS{numero_fds(nombre):03d}"


def listar_pdfs() -> list[Path]:
    """Todos los PDF del corpus, ordenados por número de FDS."""
    return sorted(config.DIR_PDF.rglob("*.pdf"), key=lambda p: numero_fds(p.name))


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()
