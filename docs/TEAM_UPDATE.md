# Team Update — Week 5

Quick catch-up for the team. Add to this as things change — it's meant to stay current, not be a one-time post.

## Where things are
- Repo is live on GitHub, structure and README are set up.
- `src/preprocessing.py` has two working steps so far, both verified against the real USDA SR Legacy data (644,125 rows, no rows lost):
  - `load_table()` — loads any raw USDA CSV into a pandas DataFrame.
  - `merge_nutrient_names()` — attaches nutrient names/units to each food_nutrient row.
  - `merge_food_descriptions()` — attaches the food's description to that table.

## Getting set up
1. `git clone https://github.com/ks731/Food-Recommender.git`
2. `cd Food-Recommender && pip install -r requirements.txt`
3. Download the USDA SR Legacy dataset (FoodData Central) and put the CSVs in `data/raw/` — that folder is gitignored, so the data isn't in the repo and you'll each need your own copy. *(Kevin: drop the exact download link here)*
4. From the repo root, run `python3 src/preprocessing.py` — you should see a merged table print out. If it errors, it's probably a missing/misnamed file in `data/raw/`.

## What's next
- Reshape the merged table from long format (one row per food-nutrient pair) to wide format (one row per food, nutrients as columns) — needed before clustering.
- Start `clustering.py` — k-means over nutrient profiles, 8 clusters, per Sarker & Tanjim.
- After that: the hard DASH sodium constraint filter, ranking, synthetic user profiles, evaluation. Full checklist is in `README.md` under Features.

## Workflow
- Pull `main` before starting anything, work on your own branch, open a PR rather than pushing straight to `main`.
- Add your name under Team in `README.md` and a note on what you're picking up.
- Questions or mix-ups — post them here so the answer's around for whoever hits the same thing next.
