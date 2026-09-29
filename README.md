# Multi-Nutrient Personalized Food Recommendations

## Overview
A food recommendation system that clusters foods by nutrient profile
(extending Sarker & Tanjim's 2026 multinutrient clustering framework) and
adds a hard sodium constraint based on the DASH (Dietary Approaches to Stop
Hypertension) eating plan. Foods that exceed the DASH sodium threshold are
filtered out before ranking, rather than just down-weighted as a soft
preference — so a hypertensive user's recommendations are guaranteed safe on
sodium, not just probably safe.

## Motivation
Existing nutrient-aware recommenders (including the system this project
extends) treat health conditions as one preference among many — weighed the
same way as, say, a preference for sweet food. That means a food that's a
great match on taste and other nutrients can still be recommended even if it
contains a dangerous amount of sodium for someone managing hypertension. This
project adds a hard, non-negotiable safety filter ahead of the ranking step
to close that gap.

## Features
(To be filled in as each module is built.)
- [ ] USDA nutrient data preprocessing and normalization
- [ ] K-means clustering over nutrient profiles
- [ ] Hard DASH sodium constraint filter
- [ ] Preference-based ranking (cosine similarity)
- [ ] Synthetic hypertensive user profile generation
- [ ] Evaluation (Precision@K, NDCG, average cosine similarity, violation rate)

## Requirements
- Python 3.x
- See requirements.txt for packages

## Installation
```bash
git clone https://github.com/<org-or-username>/Food-Recommender.git
cd Food-Recommender
pip install -r requirements.txt
```

## Usage
(To be filled in once preprocessing and the recommendation pipeline are
runnable — will include exact commands to preprocess data, generate
recommendations, and run the evaluation script.)

## Repository Structure
- `src/` — pipeline modules: preprocessing, clustering, constraints, ranking,
  user_profiles, evaluate
- `data/` — `raw/` (untouched USDA download, gitignored) and `processed/`
  (cleaned/derived data, gitignored)
- `notebooks/` — exploratory analysis
- `tests/` — unit tests, one file per src/ module
- `docs/` — Abstract_and_Literature_Review.pdf (submitted abstract + literature review); reference papers used for research are kept locally in docs/Capstone Cited Works/ but are gitignored, not part of the repo. Project Proposal not yet added — pending, see status below.

## Team
Kevin Sanches and teammates — Hunter College CSCI 49900-03 Capstone, Fall 2026.
(Add individual contributions here as they accumulate through the semester.)

## References
- Sarker & Tanjim, "A Multinutrient Clustering Framework for Personalized
  Food Recommendation," International Journal of Food Science, 2026.
- DASH (Dietary Approaches to Stop Hypertension) eating plan sodium
  guidelines: 2,300 mg/day general limit, 1,500 mg/day lower-sodium tier.
