"""Rutas y parámetros del proyecto. Todo lo que define una configuración del RAG está aquí."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Datos
DIR_PDF = RAIZ / "data" / "fds_pdf" / "FDS EVALUADAS"
DIR_MD = RAIZ / "data" / "md"
DIR_DOCLING = RAIZ / "data" / "docling_json"
DIR_META = RAIZ / "data" / "metadata"
DIR_VERIF = RAIZ / "data" / "verificacion"
DIR_CHROMA = RAIZ / "data" / "chroma"
DIR_EVAL = RAIZ / "eval"

# Extracción (Docling + OCR de imágenes)
IMAGES_SCALE = 2.0          # 2 x 72 dpi = 144 dpi para las imágenes guardadas
DOCLING_THREADS = 8         # hilos de CPU para los modelos de Docling
OCR_MIN_SCORE = 0.7         # confianza mínima de RapidOCR para conservar un texto
OCR_MIN_LADO_PX = 40        # imágenes más pequeñas (adornos de 20 pt o menos) no pasan por OCR

# Modelos locales (Ollama)
OLLAMA_URL = "http://127.0.0.1:11434"
MODELO_GENERADOR = "llama3.2"
MODELO_EMBEDDINGS = "bge-m3"
TEMPERATURA_GENERADOR = 0.0
CONTEXTO_GENERADOR = 8192   # num_ctx de Ollama

# Chunking e índice
CHUNK_MAX_CHARS = 1500      # tamaño máximo de un fragmento (caracteres)
CHUNK_OVERLAP_CHARS = 200   # solapamiento entre fragmentos de una misma sección
COLECCION = "fds"
TOP_K = 3                   # 3 respondió mejor que 6 en la evaluación (README, "Resultados")
