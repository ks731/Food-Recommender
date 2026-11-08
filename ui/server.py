"""Local web UI for the food recommender. Uses only the Python standard library.

Run from the project root:   python3 ui/server.py
Then open http://127.0.0.1:8000  (it opens automatically unless you pass --no-browser).

Needs data/processed/ to exist first (python3 src/clustering.py).
"""
import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR.parent / "src"))

try:
    import ui_service as service
except FileNotFoundError as err:
    sys.exit(
        "Missing data file: %s\n"
        "Put the USDA SR Legacy CSVs in data/raw/, then run  python3 src/clustering.py  "
        "from the project root to create data/processed/." % err.filename
    )

INDEX_FILE = UI_DIR / "index.html"


def parse_int(query, name, default):
    """Returns an int query parameter, or the default when it is missing. Raises ValueError if it is not a number"""
    values = query.get(name)
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        raise ValueError("%s must be a whole number" % name)


class Handler(BaseHTTPRequestHandler):
    server_version = "FoodRecommenderUI"

    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        self.send_body(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query, keep_blank_values=True)
        try:
            if url.path in ("/", "/index.html"):
                self.send_body(200, "text/html; charset=utf-8", INDEX_FILE.read_bytes())
            elif url.path == "/api/meta":
                self.send_json(200, service.meta())
            elif url.path == "/api/recommend":
                goal = query.get("goal", [service.DEFAULT_GOAL])[0]
                limit = parse_int(query, "limit", service.DEFAULT_LIMIT)
                top_n = parse_int(query, "top_n", service.DEFAULT_TOP_N)
                self.send_json(200, service.run(goal, limit, top_n))
            else:
                self.send_json(404, {"error": "Not found"})
        except ValueError as err:
            self.send_json(400, {"error": str(err)})
        except Exception:
            traceback.print_exc()
            self.send_json(500, {"error": "Something went wrong. See the terminal for details."})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description="Run the food recommender web UI")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as err:
        sys.exit("Could not start on port %d (%s). Try another one, e.g.  python3 ui/server.py --port %d"
                 % (args.port, err.strerror, args.port + 1))

    url = "http://%s:%d" % ("127.0.0.1" if args.host in ("0.0.0.0", "") else args.host, args.port)
    print("Food recommender UI running at %s  (press Ctrl+C to stop)" % url)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
