# Multi-Nutrient Personalized Food Recommendations

## What this is
Extends Sarker & Tanjim's k-means clustering food recommender (2026) with a
hard DASH sodium constraint, so hypertension safety is enforced as a filter
before ranking, not treated as a soft preference weighed against taste or
other nutrient goals.

## Core architecture decision
Pipeline order: filter DASH-violating foods OUT of the candidate pool first,
THEN cluster/rank the remainder by preference match. Never rank first and
truncate to K after — that can leave a K-sized list with fewer than K items,
or silently keep a food that ranks well but breaks the sodium limit.

## Tech stack
Python, pandas, NumPy, scikit-learn (KMeans, StandardScaler, cosine_similarity),
matplotlib/seaborn for eval plots, pytest for tests.

## Data
USDA FoodData Central "SR Legacy" (successor to SR28), ~8,790 foods across 23
nutrients. Raw files go in data/raw/ (gitignored — don't commit the dataset
itself; document the exact download source/version in README.md once fetched).

## Conventions
- Plain, conversational docstrings/comments — explain the "why", not just the "what".
- One function's responsibility per pipeline stage in src/ (preprocessing,
  clustering, constraints, ranking, user_profiles, evaluate) — keep each
  independently testable.
- Tests in tests/ mirror the src/ module they cover (e.g. test_constraints.py
  tests constraints.py).
- Team of 3 — commit under your own name/account so individual GitHub
  contribution stays visible (required by the course syllabus).

## Status
Repo scaffold created (folders, README, requirements.txt). No implementation
yet. Project Abstract and Literature Review are complete (see docs/,
Abstract_and_Literature_Review.pdf). Project Proposal is a SEPARATE
deliverable per the syllabus (due end of week 5) and has NOT been written yet
- confirm with the professor whether it's still required given what's
already been submitted, then add it to docs/ once done. Next up:
src/preprocessing.py — load the USDA SR Legacy dataset, clean it, and
z-score normalize the nutrient columns.
