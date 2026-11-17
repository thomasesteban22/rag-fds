"""Consulta al RAG desde la terminal.

Uso:
    uv run python -m rag_fds.ask "¿Cuál es el punto de inflamación del Esmalte Uretano Part A de Sika?"
    uv run python -m rag_fds.ask --k 6 "pregunta"
    uv run python -m rag_fds.ask            # modo interactivo
    uv run python -m rag_fds.ask --contextos "pregunta"   # muestra también los fragmentos recuperados
"""
import argparse

from rag_fds import config
from rag_fds.rag import RAG


def mostrar(rag: RAG, pregunta: str, contextos: bool):
    r = rag.responder(pregunta)
    print(f"\nPregunta: {pregunta}")
    print(f"Filtro de recuperación: {r.filtro} · top-k = {rag.top_k} · tiempos {r.segundos}")
    print("\nRespuesta:\n" + r.respuesta)
    if not r.insuficiente:
        print("\nFuentes citadas:")
        for f in r.fuentes():
            print("  " + f)
    for a in r.advertencias:
        print(f"\nAdvertencia: {a}")
    if contextos:
        print("\nFragmentos recuperados (en orden):")
        for i, f in enumerate(r.fragmentos, 1):
            print(f"\n--- [{i}] {f.id} · similitud {f.similitud}\n{f.texto[:800]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pregunta", nargs="?")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="número de fragmentos recuperados")
    ap.add_argument("--contextos", action="store_true", help="mostrar los fragmentos recuperados")
    args = ap.parse_args()
    rag = RAG(top_k=args.k)
    if args.pregunta:
        mostrar(rag, args.pregunta, args.contextos)
        return
    print("Escribe una pregunta (o 'salir').")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q.lower() in {"salir", "exit", "quit"}:
            break
        if q:
            mostrar(rag, q, args.contextos)


if __name__ == "__main__":
    main()
