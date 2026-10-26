"""Local web UI for live text-to-speech with the project's checkpoints (standard library HTTP server).

    python webui/server.py                     # http://127.0.0.1:8000, opens the browser
    python webui/server.py --port 8080 --no-browser --config webui/config.json

Uses webui/config.json when it exists, otherwise the committed webui/config.example.json.
Binds to 127.0.0.1 by default; the API has no authentication.
"""

import argparse
import json
import mimetypes
import sys
import traceback
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webui.engine import Engine  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 2_000_000
# Windows can map .js to text/plain through the registry, which browsers refuse for scripts.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")


class Handler(BaseHTTPRequestHandler):
    engine = None
    # Keep-alive: the agent demo makes many small requests; a new connection each time costs more
    # than a chunk's synthesis. Every response sets Content-Length.
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if not self.path.startswith("/static/"):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), format % args))

    def send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        body = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not getattr(self, "head_only", False):
            self.wfile.write(body)

    def do_HEAD(self):
        self.head_only = True
        self.do_GET()

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/status":
            return self.send_json(self.engine.status())
        relative = "index.html" if route == "/" else route.removeprefix("/static/")
        path = (STATIC / relative).resolve()
        if (route != "/" and not route.startswith("/static/")) or not path.is_relative_to(STATIC) \
                or not path.is_file():
            return self.send_json({"error": "Not found"}, 404)
        self.send_file(path)

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True  # the unread body must not be parsed as the next request
            return self.send_json({"error": "Request too large"}, 413)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            if route == "/api/load":
                return self.send_json(self.engine.load(request["model"]))
            if route == "/api/prepare":
                return self.send_json({"segments": self.engine.prepare(request["text"],
                                                                        request.get("normalizer", "english"))})
            if route == "/api/synthesize":
                return self.send_json(self.engine.synthesize(
                    request["model"], request["text"], noise_scale=request.get("noise_scale", 0.667),
                    length_scale=request.get("length_scale", 1.0), seed=request.get("seed"),
                    speaker=request.get("speaker")))
            if route == "/api/synthesize_chunk":
                return self.send_json(self.engine.synthesize_chunk(
                    request["model"], request["text"], request["start"], request["end"],
                    lookahead=request.get("lookahead", 0), mode=request.get("mode", "context"),
                    seed=request.get("seed"), noise_scale=request.get("noise_scale", 0.667),
                    length_scale=request.get("length_scale", 1.0), speaker=request.get("speaker")))
            return self.send_json({"error": "Not found"}, 404)
        except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
            return self.send_json({"error": f"{type(error).__name__}: {error}"}, 400)
        except Exception as error:  # report, keep serving
            traceback.print_exc()
            return self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).with_name("config.json")
    if not default_config.is_file():
        default_config = default_config.with_name("config.example.json")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda:0 (default: cuda:0 if available)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    engine = Engine(args.config, args.device)
    print(f"Config: {engine.config_path}\nCheckpoint root: {engine.checkpoint_root}\nDevice: {engine.device}")
    for row in engine.model_list():
        print(f"  {row['id']:<8} {'found' if row['available'] else 'MISSING':<8} {row['path'] or row['pattern']}")
    default = engine.config.get("default_model")
    if default and engine.entries.get(default) and any(r["id"] == default and r["available"]
                                                        for r in engine.model_list()):
        print(f"Loading {default} ...")
        engine.load(default)
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_port}/"
    print(f"Serving {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
