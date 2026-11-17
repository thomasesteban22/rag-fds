"""Interfaz web mínima para la demostración (solo biblioteca estándar).

Uso:
    uv run python -m rag_fds.web            # abre http://127.0.0.1:8000
    uv run python -m rag_fds.web --puerto 8080

Escucha solo en 127.0.0.1. Muestra la respuesta, las fuentes citadas (fabricante, producto,
documento, versión, página y sección), los fragmentos recuperados en orden y las advertencias.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag_fds import config
from rag_fds.rag import RAG

_RAGS: dict[int, RAG] = {}

PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG de FDS</title>
<style>
:root { --fondo:#f7f7f5; --texto:#1d1d1b; --suave:#5f5f5a; --borde:#d9d9d4; --acento:#0b5cad; --aviso:#9a5b00; --tarjeta:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --fondo:#161615; --texto:#ececea; --suave:#a3a39d; --borde:#3a3a37; --acento:#6aa8ec; --aviso:#e0a748; --tarjeta:#1f1f1d; } }
body { margin:0; background:var(--fondo); color:var(--texto); font:16px/1.5 system-ui, sans-serif; }
main { max-width:900px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.4rem; margin:0 0 4px; } p.sub { color:var(--suave); margin:0 0 20px; }
form { display:flex; flex-direction:column; gap:10px; }
textarea { font:inherit; padding:10px; border:1px solid var(--borde); border-radius:8px; background:var(--tarjeta); color:var(--texto); min-height:70px; }
.fila { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
select, button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--borde); background:var(--tarjeta); color:var(--texto); }
button { background:var(--acento); color:#fff; border:none; cursor:pointer; } button:disabled { opacity:.6; }
.tarjeta { background:var(--tarjeta); border:1px solid var(--borde); border-radius:10px; padding:16px; margin-top:18px; }
.meta { color:var(--suave); font-size:.9rem; } .aviso { color:var(--aviso); font-size:.95rem; }
ol.fuentes li { margin-bottom:6px; } details { margin-top:8px; } pre { white-space:pre-wrap; font-size:.85rem; }
</style></head><body><main>
<h1>Consulta de Fichas de Datos de Seguridad</h1>
<p class="sub">101 FDS de pinturas · llama3.2 + bge-m3 en Ollama · cada afirmación cita su fuente</p>
<form id="f"><textarea id="q" placeholder="Ej.: ¿Cuál es el punto de inflamación del Esmalte Uretano Part A de Sika?"></textarea>
<div class="fila"><label>Fragmentos (top-k) <select id="k"><option selected>3</option><option>6</option></select></label>
<button id="b" type="submit">Preguntar</button><span id="estado" class="meta"></span></div></form>
<div id="r"></div>
<script>
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim(); if (!q) return;
  const b = document.getElementById('b'), est = document.getElementById('estado'), r = document.getElementById('r');
  b.disabled = true; est.textContent = 'Buscando y generando…'; r.innerHTML = '';
  try {
    const res = await fetch('/api/preguntar', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pregunta:q, k: parseInt(document.getElementById('k').value)})});
    const d = await res.json();
    if (d.error) { r.innerHTML = `<div class="tarjeta aviso">${esc(d.error)}</div>`; return; }
    let h = `<div class="tarjeta"><div>${esc(d.respuesta).replace(/\\n/g,'<br>')}</div>`;
    h += `<p class="meta">Filtro: ${esc(d.filtro)} · recuperación ${d.segundos.recuperacion} s · generación ${d.segundos.generacion} s</p>`;
    d.advertencias.forEach(a => h += `<p class="aviso">Advertencia: ${esc(a)}</p>`);
    if (d.fuentes.length) h += '<h3>Fuentes</h3><ol class="fuentes">' + d.fuentes.map(f => `<li>${esc(f)}</li>`).join('') + '</ol>';
    h += '<details><summary>Fragmentos recuperados (en orden)</summary>' + d.fragmentos.map((f,i) =>
      `<p class="meta">[${i+1}] ${esc(f.id)} · similitud ${f.similitud}</p><pre>${esc(f.texto)}</pre>`).join('') + '</details></div>';
    r.innerHTML = h;
  } catch (err) { r.innerHTML = `<div class="tarjeta aviso">Error: ${esc(String(err))}</div>`; }
  finally { b.disabled = false; est.textContent = ''; }
});
</script></main></body></html>"""


class Manejador(BaseHTTPRequestHandler):
    def _enviar(self, codigo: int, cuerpo: bytes, tipo: str):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._enviar(200, PAGINA.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._enviar(404, b"no encontrado", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/preguntar":
            self._enviar(404, b"no encontrado", "text/plain; charset=utf-8")
            return
        try:
            datos = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            pregunta = str(datos.get("pregunta", "")).strip()[:1000]
            k = int(datos.get("k", config.TOP_K))
            if k not in (3, 6) or not pregunta:
                raise ValueError("pregunta vacía o top-k no permitido")
            if k not in _RAGS:
                _RAGS[k] = RAG(top_k=k)
            rag = _RAGS[k]
            r = rag.responder(pregunta)
            salida = {
                "respuesta": r.respuesta, "filtro": r.filtro, "segundos": r.segundos,
                "advertencias": r.advertencias, "fuentes": r.fuentes() if not r.insuficiente else [],
                "fragmentos": [{"id": f.id, "similitud": f.similitud, "texto": f.texto} for f in r.fragmentos],
            }
        except Exception as e:  # noqa: BLE001 - se muestra el error en la página
            salida = {"error": f"{type(e).__name__}: {e}"}
        self._enviar(200, json.dumps(salida, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, formato, *args):
        print("web:", formato % args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8000)
    args = ap.parse_args()
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador)
    print(f"Abre http://127.0.0.1:{args.puerto} (Ctrl+C para salir)")
    servidor.serve_forever()


if __name__ == "__main__":
    main()
