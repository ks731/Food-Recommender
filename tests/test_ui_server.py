import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import pytest

SERVER_FILE = Path(__file__).resolve().parent.parent / "ui" / "server.py"
spec = importlib.util.spec_from_file_location("ui_server", SERVER_FILE)
ui_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui_server)


@pytest.fixture(scope="module")
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.Handler)   # port 0 = any free port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get_content_type(), err.read()


def test_home_page_is_served(base_url):
    status, kind, body = get(base_url + "/")
    assert status == 200 and kind == "text/html"
    assert b"DASH Food Recommender" in body


def test_meta_endpoint(base_url):
    status, kind, body = get(base_url + "/api/meta")
    assert status == 200 and kind == "application/json"
    assert len(json.loads(body)["goals"]) == 5


def test_recommend_endpoint_uses_defaults_and_parameters(base_url):
    status, _, body = get(base_url + "/api/recommend")
    assert status == 200 and len(json.loads(body)["foods"]) == 5
    status, _, body = get(base_url + "/api/recommend?goal=low_carb&limit=2300&top_n=10")
    data = json.loads(body)
    assert status == 200 and len(data["foods"]) == 10 and data["profile"]["limit"] == 2300


@pytest.mark.parametrize("query", ["goal=nope", "limit=999", "top_n=7", "top_n=abc", "limit="])
def test_bad_parameters_give_400_with_a_message(base_url, query):
    status, _, body = get(base_url + "/api/recommend?" + query)
    assert status == 400 and "error" in json.loads(body)


@pytest.mark.parametrize("path", ["/missing", "/api", "/../README.md", "/src/ranking.py"])
def test_other_paths_are_404(base_url, path):
    assert get(base_url + path)[0] == 404
