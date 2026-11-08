# Web UI

A local web page for the food recommender. It uses only the Python standard library, so
there is nothing extra to install beyond `requirements.txt`.

## Run it

From the project root, after `python3 src/clustering.py` has created `data/processed/`:

```bash
python3 ui/server.py
```

It opens http://127.0.0.1:8000 in your browser. Press Ctrl+C in the terminal to stop.
Options: `--port 8001` (if 8000 is busy), `--no-browser`.

## What it shows

Pick a nutrition goal and a DASH sodium plan, then press **Get recommendations**:

1. the profile (goal, targets in real units, weights),
2. the nearest cluster and how many foods it holds,
3. how many of those foods the sodium filter removed,
4. the top 5 or top 10 foods by cosine similarity, and
5. the best-matching foods that the filter blocked.

Food group names (for example "Poultry Products") come from `data/raw/food_category.csv`.
If that file is missing, the page shows the cluster number only.

## Files

| File | Purpose |
| --- | --- |
| `ui/server.py` | Small HTTP server: serves the page and two JSON endpoints |
| `ui/index.html` | The page (HTML, CSS and JavaScript in one file, no external resources) |
| `src/ui_service.py` | Turns a choice into results using `make_profile` and `recommend` |
| `tests/test_ui_service.py`, `tests/test_ui_server.py` | Tests for the logic and the server |

Endpoints: `GET /api/meta` and `GET /api/recommend?goal=high_protein&limit=1500&top_n=5`.
The server listens on this computer only (127.0.0.1) unless you pass `--host`.
