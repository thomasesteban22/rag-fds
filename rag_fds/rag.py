"""RAG sobre las FDS: recuperación híbrida (embeddings + BM25), generación con llama3.2 (Ollama) y citas.

Flujo de una pregunta:
  1. Se detectan en la pregunta productos, número de FDS, fabricantes y años mencionados,
     comparando contra el inventario (nombres de producto según la ficha y según el archivo).
  2. Candidatos: si se detectó un producto (o varias versiones del mismo), sus fichas; si se
     detectaron dos o más productos distintos, cada uno por separado y los k fragmentos se
     reparten entre ellos; si solo se detectó un fabricante, sus fichas; si no, todo el corpus.
  3. Recuperación híbrida sobre los candidatos:
     - similitud coseno entre el embedding bge-m3 de la pregunta y los de los fragmentos
       (guardados en Chroma; con 3583 fragmentos la búsqueda exacta toma milisegundos);
     - BM25 sobre las palabras de los fragmentos (sin tildes y sin palabras vacías);
     - los dos órdenes se combinan con fusión por rangos recíprocos (RRF, k = 60).
     Motivo: dentro de una ficha todos los fragmentos comparten la cabecera (fabricante,
     producto, versión) y con solo embeddings una palabra clave de la pregunta ("densidad",
     "pH") pesa poco frente a esa cabecera; BM25 sí la tiene en cuenta.
  4. Si la mayor similitud coseno entre los candidatos no supera UMBRAL_SIMILITUD, se responde
     que no hay evidencia suficiente sin llamar al generador.
  5. El generador recibe los fragmentos numerados [1]..[k] con su fabricante, producto, versión,
     documento, página y sección, y debe citar [n] después de cada afirmación.
  6. Las citas se validan: se descartan números que no corresponden a un fragmento y se arma
     la lista de fuentes con fabricante, producto, documento, página y sección.
"""
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import chromadb
import numpy as np
import ollama

from rag_fds import config
from rag_fds.index import cargar_inventario, fecha_version
from rag_fds.utils import sin_tildes

UMBRAL_SIMILITUD = 0.40
K_RRF = 60          # constante de la fusión por rangos recíprocos (valor usual)
BM25_K1 = 1.2       # saturación de la frecuencia de un término (valor usual)
BM25_B = 0.75       # normalización por largo del fragmento (valor usual)
SIN_EVIDENCIA = "No hay evidencia suficiente en las FDS disponibles para responder esta pregunta."

PROMPT_SISTEMA = f"""Eres un asistente que responde preguntas sobre Fichas de Datos de Seguridad (FDS) de pinturas.
Usa ÚNICAMENTE la información de los fragmentos que se te entregan. No uses conocimiento propio.

Reglas:
1. Después de cada afirmación escribe entre corchetes el número del fragmento que la respalda, por ejemplo [2]. Si viene de dos fragmentos, cita ambos: [1][3].
2. Cada fragmento indica fabricante, producto, versión y documento. No mezcles datos de productos ni de versiones distintas. Si comparas, di a qué producto y versión corresponde cada dato y si los valores son iguales o distintos.
3. Copia cifras, unidades y códigos (CAS, H, P, ONU) exactamente como aparecen en el fragmento.
4. Si los fragmentos no contienen la información pedida, responde solamente: "{SIN_EVIDENCIA}"
5. Si la información está solo en parte, responde lo que sí está con su cita y di qué parte no aparece en los fragmentos.
6. Responde en español y de forma breve.

Ejemplo de formato:
Pregunta: ¿Cuál es el punto de ebullición del producto X?
Respuesta: Según la FDS del producto X (versión 2), el punto de ebullición es 100 °C [2]."""
RECORDATORIO = "Recuerda: termina cada oración con la cita del fragmento que la respalda, por ejemplo [1]."

_MARCAS = {
    "corona": "Corona", "pintuco": "Pintuco", "sika": "Sika", "ppg": "PPG", "invesa": "Invesa",
    "pintuland": "Pintuland", "akzonobel": "AkzoNobel", "aksonobel": "AkzoNobel", "international": "AkzoNobel",
    "axalta": "Axalta", "tecnidecor": "Tecnidecor", "bler": "Pinturas Bler", "graphenstone": "Graphenstone (IEDISA)",
}
_VACIAS = {"de", "la", "el", "los", "las", "y", "en", "para", "con", "por", "a", "del", "al", "su", "fds", "ficha",
           "part", "parte", "comp", "componente", "mod", "g", "lt", "x", "&"}
# Para BM25 se quitan además las palabras de las preguntas que no dicen de qué se pregunta.
_VACIAS_BM25 = _VACIAS | {"cual", "cuales", "que", "es", "son", "se", "un", "una", "o", "u", "como", "segun",
                          "indica", "indican", "tiene", "tienen", "hay", "lo", "le", "sus", "este", "esta",
                          "sobre", "entre", "fichas", "producto", "productos"}
_ECO_PREGUNTA = re.compile(r"^\s*Pregunta:.*?\n\s*Respuesta[^:\n]*:\s*", re.S)
_ECO_RESPUESTA = re.compile(r"^\s*Respuesta[^:\n]*:\s*")


def _normalizar(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", sin_tildes(texto)) if t not in _VACIAS]


def _tokens_bm25(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", sin_tildes(texto)) if t not in _VACIAS_BM25]


@dataclass
class Fragmento:
    id: str
    texto: str
    meta: dict
    similitud: float   # coseno entre la pregunta y el fragmento (embeddings)


@dataclass
class Respuesta:
    pregunta: str
    respuesta: str
    fragmentos: list[Fragmento]
    respuesta_modelo: str = ""   # texto tal como lo devolvió el generador, antes de agregar citas
    citados: list[int] = field(default_factory=list)
    insuficiente: bool = False
    filtro: str = ""
    similitud_max: float = 0.0   # mayor coseno entre los candidatos (lo que se compara con el umbral)
    advertencias: list[str] = field(default_factory=list)
    segundos: dict = field(default_factory=dict)

    def fuentes(self, solo_citadas: bool = True) -> list[str]:
        idx = self.citados if (solo_citadas and self.citados) else range(1, len(self.fragmentos) + 1)
        out = []
        for i in idx:
            m = self.fragmentos[i - 1].meta
            out.append(f"[{i}] {m['fabricante']} · {m['producto']} · {m['doc_id']} ({m['archivo']}) · "
                       f"versión {m['version'] or 'no indicada'}, fecha {m['fecha'] or 'no indicada'} · "
                       f"pág. {m['paginas'] or '?'} · sección {m['seccion']}: {m['seccion_titulo']}")
        return out


class RAG:
    def __init__(self, top_k: int = config.TOP_K):
        self.top_k = top_k
        self.ollama = ollama.Client(host=config.OLLAMA_URL)
        self.col = chromadb.PersistentClient(path=str(config.DIR_CHROMA)).get_collection(config.COLECCION)
        self.inventario = cargar_inventario()
        self._cargar_fragmentos()
        self._preparar_alias()

    def _preparar_alias(self):
        """Nombres de producto de cada ficha (sin marcas, que se detectan aparte) y sus palabras exclusivas."""
        self.alias: dict[str, list[list[str]]] = {}
        df = Counter()
        for doc_id, f in self.inventario.items():
            nombres = {tuple(t for t in _normalizar(f[c]) if t not in _MARCAS)
                       for c in ("producto", "producto_segun_archivo")}
            self.alias[doc_id] = [list(n) for n in nombres if n]
            for t in {t for n in nombres for t in n}:
                df[t] += 1
        n = len(self.inventario)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
        # Una palabra del nombre que en todo el corpus solo aparece en las fichas de ese producto
        # (todas sus versiones) lo identifica sola: "vinicryl", "13883".
        docs_de_palabra: dict[str, set] = {}
        for texto, m in zip(self.textos, self.metas):
            for t in set(re.findall(r"[a-z0-9]+", sin_tildes(texto))):
                docs_de_palabra.setdefault(t, set()).add(m["doc_id"])
        mismo_producto: dict[str, set] = {}
        for d, f in self.inventario.items():
            mismo_producto.setdefault(" ".join(_normalizar(f["producto"])), set()).add(d)
        self.exclusivas: dict[str, set] = {}
        for d, f in self.inventario.items():
            grupo = mismo_producto[" ".join(_normalizar(f["producto"]))]
            self.exclusivas[d] = {t for n in self.alias[d] for t in n
                                  if len(t) >= 4 and docs_de_palabra.get(t, {d}) <= grupo}

    def _cargar_fragmentos(self):
        """Lee de Chroma textos, metadatos y embeddings, y prepara las estadísticas de BM25."""
        todo = self.col.get(include=["documents", "metadatas", "embeddings"])
        self.ids = list(todo["ids"])
        self.textos = list(todo["documents"])
        self.metas = list(todo["metadatas"])
        emb = np.asarray(todo["embeddings"], dtype=np.float32)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.tf = [Counter(_tokens_bm25(t)) for t in self.textos]
        self.largo = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.largo_medio = float(self.largo.mean())
        df = Counter(t for c in self.tf for t in c)
        n = len(self.ids)
        self.idf_bm25 = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    # ---------- detección de entidades ----------
    def detectar(self, pregunta: str) -> dict:
        q = set(_normalizar(pregunta))
        docs = {}
        for m in re.finditer(r"FDS\s*0*(\d{1,3})\b", pregunta, re.I):
            d = f"FDS{int(m.group(1)):03d}"
            if d in self.inventario:
                docs[d] = 1.0
        for doc_id, nombres in self.alias.items():
            for toks in nombres:
                total = sum(self.idf[t] for t in toks)
                hecho = sum(self.idf[t] for t in toks if t in q)
                if total and hecho / total >= 0.75 and sum(t in q for t in toks) >= min(2, len(toks)):
                    docs[doc_id] = max(docs.get(doc_id, 0), hecho / total)
            if self.exclusivas[doc_id] & q:
                docs[doc_id] = max(docs.get(doc_id, 0), 0.75)
        if docs:
            mejor = max(docs.values())
            docs = {d: s for d, s in docs.items() if s >= mejor - 0.15}
        # Un año escoge entre versiones del mismo producto; no descarta los demás productos mencionados.
        anios = set(re.findall(r"\b(20(?:1[89]|2[0-5]))\b", pregunta))
        if anios and len(docs) > 1:
            grupos: dict[str, list[str]] = {}
            for d in docs:
                grupos.setdefault(" ".join(_normalizar(self.inventario[d]["producto"])), []).append(d)
            for ds in grupos.values():
                con_anio = [d for d in ds if any(a in self.inventario[d]["carpeta_origen"] or a in fecha_version(self.inventario[d]) for a in anios)]
                if con_anio:
                    for d in set(ds) - set(con_anio):
                        del docs[d]
        marcas = sorted({v for k, v in _MARCAS.items() if k in q})
        # Palabras que ya sirvieron para escoger las fichas: no se usan en BM25.
        usadas = {k for k in _MARCAS if k in q} | anios | {t for d in docs for n in self.alias[d] for t in n}
        return {"docs": sorted(docs), "fabricantes": marcas, "usadas": usadas}

    # ---------- recuperación ----------
    def _bm25(self, consulta: list[str], i: int) -> float:
        tf, s = self.tf[i], 0.0
        norma = BM25_K1 * (1 - BM25_B + BM25_B * self.largo[i] / self.largo_medio)
        for t in consulta:
            f = tf.get(t)
            if f:
                s += self.idf_bm25[t] * f * (BM25_K1 + 1) / (f + norma)
        return s

    def _consultar(self, emb, consulta, k, campo=None, valores=None) -> tuple[list[Fragmento], float]:
        """Top-k por RRF entre coseno y BM25, limitado a los fragmentos con meta[campo] en valores."""
        cand = [i for i, m in enumerate(self.metas) if campo is None or m[campo] in valores]
        if not cand:
            return [], 0.0
        sims = self.emb[cand] @ emb
        bm25 = np.array([self._bm25(consulta, i) for i in cand])
        rrf = np.zeros(len(cand))
        for puntajes in (sims, bm25):
            orden = np.argsort(-puntajes, kind="stable")
            rangos = np.empty(len(cand))
            rangos[orden] = np.arange(1, len(cand) + 1)
            rrf += np.where(puntajes > 0, 1.0 / (K_RRF + rangos), 0.0)
        mejores = np.argsort(-rrf, kind="stable")[:k]
        frs = [Fragmento(self.ids[cand[j]], self.textos[cand[j]], self.metas[cand[j]], round(float(sims[j]), 4))
               for j in mejores]
        return frs, round(float(sims.max()), 4)

    def recuperar(self, pregunta: str, k: int | None = None) -> tuple[list[Fragmento], str, float]:
        """Devuelve (fragmentos en orden, filtro aplicado, mayor coseno entre los candidatos)."""
        k = k or self.top_k
        emb = np.asarray(self.ollama.embed(model=config.MODELO_EMBEDDINGS, input=[pregunta]).embeddings[0],
                         dtype=np.float32)
        emb /= np.linalg.norm(emb)
        ent = self.detectar(pregunta)
        consulta = [t for t in _tokens_bm25(pregunta) if t not in ent["usadas"]]
        docs = ent["docs"]
        if docs:
            grupos: dict[str, list[str]] = {}
            for d in docs:  # versiones del mismo producto van juntas
                clave = " ".join(_normalizar(self.inventario[d]["producto"]))
                grupos.setdefault(clave, []).append(d)
            if len(grupos) == 1:
                frs, sim = self._consultar(emb, consulta, k, "doc_id", set(docs))
            else:
                por_grupo = math.ceil(k / len(grupos))
                partes = [self._consultar(emb, consulta, por_grupo, "doc_id", set(ds)) for ds in grupos.values()]
                # Se intercalan los grupos (1.º de cada uno, luego 2.º...) para que ninguno quede fuera al cortar en k.
                frs = [f for fila in zip(*[p[0] for p in partes]) for f in fila][:k]
                frs += [f for p in partes for f in p[0] if f not in frs][:k - len(frs)]
                sim = max(p[1] for p in partes)
            return frs, f"fichas {', '.join(docs)}", sim
        if ent["fabricantes"]:
            fab = ent["fabricantes"]
            frs, sim = self._consultar(emb, consulta, k, "fabricante", set(fab))
            return frs, f"fabricante {', '.join(fab)}", sim
        frs, sim = self._consultar(emb, _tokens_bm25(pregunta), k)
        return frs, "sin filtro", sim

    # ---------- generación ----------
    def _prompt_usuario(self, pregunta: str, frs: list[Fragmento]) -> str:
        partes = []
        for i, f in enumerate(frs, 1):
            m = f.meta
            partes.append(f"[{i}] (Fabricante: {m['fabricante']} | Producto: {m['producto']} | "
                          f"Versión: {m['version'] or 'no indicada'} | Fecha: {m['fecha'] or 'no indicada'} | "
                          f"Documento: {m['doc_id']}, pág. {m['paginas']} | Sección {m['seccion']}: {m['seccion_titulo']})\n"
                          f"{f.texto.split(chr(10), 1)[-1]}")
        return ("Fragmentos:\n\n" + "\n\n".join(partes) + f"\n\n{RECORDATORIO}\n\n"
                f"Pregunta: {pregunta}\nRespuesta (con citas [n]):")

    @staticmethod
    def _revisar_citas(texto: str, frs: list[Fragmento]) -> tuple[str, list[int], list[int]]:
        """Agrega citas por coincidencia de palabras y cifras; nunca quita las del modelo.

        Una oración sin cita recibe la del fragmento que contiene el 60 % o más de sus palabras.
        Una oración con cita recibe además la de otro fragmento si este contiene el 60 % o más y
        supera en 0,2 o más al mejor fragmento citado (el modelo de 3B suele citar el fragmento
        cuya cabecera tiene la versión y no el que tiene el dato).
        Devuelve (texto, citas agregadas a oraciones sin cita, citas agregadas a oraciones citadas).
        """
        oraciones = re.split(r"(?<=[.!?])\s+", texto)
        tok_frs = [set(_normalizar(f.texto)) for f in frs]
        salida, sin_cita, corregidas = [], [], []
        for o in oraciones:
            toks = set(_normalizar(re.sub(r"\[\d+\]", " ", o)))
            if len(toks) < 3:
                salida.append(o)
                continue
            cobertura = [len(toks & t) / len(toks) for t in tok_frs]
            mejor = max(range(len(frs)), key=lambda i: cobertura[i])
            citadas = [int(c) for c in re.findall(r"\[(\d+)\]", o) if 1 <= int(c) <= len(frs)]
            if cobertura[mejor] >= 0.6 and mejor + 1 not in citadas:
                if not citadas:
                    sin_cita.append(mejor + 1)
                elif cobertura[mejor] - max(cobertura[c - 1] for c in citadas) >= 0.2:
                    corregidas.append(mejor + 1)
                else:
                    salida.append(o)
                    continue
                o = re.sub(r"([.!?]?)\s*$", f" [{mejor + 1}]\\1", o, count=1)
            salida.append(o)
        return " ".join(salida), sorted(set(sin_cita)), sorted(set(corregidas))

    def responder(self, pregunta: str, k: int | None = None) -> Respuesta:
        t0 = time.perf_counter()
        frs, filtro, sim_max = self.recuperar(pregunta, k)
        t1 = time.perf_counter()
        r = Respuesta(pregunta=pregunta, respuesta="", fragmentos=frs, filtro=filtro, similitud_max=sim_max)
        if not frs or sim_max < UMBRAL_SIMILITUD:
            r.respuesta, r.insuficiente = SIN_EVIDENCIA, True
            r.advertencias.append(f"similitud máxima {sim_max} < umbral {UMBRAL_SIMILITUD}")
            r.segundos = {"recuperacion": round(t1 - t0, 2), "generacion": 0.0}
            return r
        salida = self.ollama.chat(
            model=config.MODELO_GENERADOR,
            messages=[{"role": "system", "content": PROMPT_SISTEMA},
                      {"role": "user", "content": self._prompt_usuario(pregunta, frs)}],
            options={"temperature": config.TEMPERATURA_GENERADOR, "seed": 42,
                     "num_ctx": config.CONTEXTO_GENERADOR, "num_predict": 600},
        )
        t2 = time.perf_counter()
        r.respuesta_modelo = salida.message.content.strip()
        # A veces el modelo repite el formato del ejemplo ("Pregunta: ... Respuesta: ..."): se quita ese eco.
        texto = _ECO_RESPUESTA.sub("", _ECO_PREGUNTA.sub("", r.respuesta_modelo, count=1), count=1).strip()
        # Es "insuficiente" si el modelo usó la frase de falta de evidencia y no respaldó nada con citas.
        r.insuficiente = (sin_tildes(SIN_EVIDENCIA[:30]) in sin_tildes(texto)
                          and not any(1 <= int(c) <= len(frs) for c in re.findall(r"\[(\d+)\]", texto)))
        if not r.insuficiente:
            texto, sin_cita, corregidas = self._revisar_citas(texto, frs)
            if sin_cita:
                r.advertencias.append(f"citas {sin_cita} agregadas automáticamente a oraciones sin cita, por coincidencia de palabras")
            if corregidas:
                r.advertencias.append(f"citas {corregidas} agregadas automáticamente: respaldan la oración mejor que la cita del modelo")
        r.respuesta = texto
        citas = [int(c) for c in re.findall(r"\[(\d+)\]", texto)]
        r.citados = sorted({c for c in citas if 1 <= c <= len(frs)})
        invalidas = sorted({c for c in citas if not 1 <= c <= len(frs)})
        if invalidas:
            r.advertencias.append(f"citas a fragmentos inexistentes: {invalidas}")
        if not r.citados and not r.insuficiente:
            r.advertencias.append("la respuesta no cita ningún fragmento; verifíquela contra las fuentes")
        r.segundos = {"recuperacion": round(t1 - t0, 2), "generacion": round(t2 - t1, 2)}
        return r
