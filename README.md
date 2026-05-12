# Kaycorp Intelligent Review Agent

A product-review system that builds user memory, item memory, product recommendations, and Nigerian-contextualized review generation.

## What It Does

- Builds item profiles with average rating, review count, sample review memory, and embeddings.
- Builds customer profiles with positivity, activity level, verbosity, and sample behavior when raw reviewer IDs are unavailable.
- Recommends semantically similar products using sentence-transformer embeddings and cosine similarity.
- Generates profile-aware product reviews with restrained Nigerian English style.
- Exposes the system through FastAPI and Streamlit.

## Project Structure

```text
app/
  api.py              FastAPI endpoints
  profile_builder.py  user/item memory builder
  recommender.py      semantic similarity retrieval
  review_agent.py     review generation logic
  service.py          shared app service layer
data/processed/
  clean_reviews.csv
  user_profiles.csv
  item_profiles.csv
  user_likes.csv
streamlit_app.py
main.py
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

`python main.py` ingests real review-level files from `data/raw/*.jsonl.gz`, builds `clean_reviews.csv`, then creates user/item profiles and embeddings.

## Run The API

```bash
uvicorn app.api:app --reload
```

Open:

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

Useful endpoints:

- `GET /health`
- `GET /products`
- `GET /users`
- `GET /recommendations/product/{product_id}`
- `GET /recommendations/user/{user_id}`
- `POST /reviews/generate`
- `POST /task-a/generate-review`
- `POST /task-b/recommend`
- `POST /task-b/conversation-turn`

Example review request:

```json
{
  "user_id": "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ",
  "product_id": "B08L6L3X1S"
}
```

Review generation request:

```json
{
  "persona": "A budget-conscious Nigerian student who writes short reviews and cares about price, delivery speed, and durability.",
  "product_details": {
    "product_id": "B08L6L3X1S",
    "product_name": "Clear protective phone case",
    "category": "Cell Phones & Accessories",
    "features": "Transparent slim protective case"
  }
}
```

Recommendation request:

```json
{
  "persona": "A Lagos student who wants affordable phone accessories with fast delivery.",
  "context": "Needs a durable phone case.",
  "product_history": [],
  "category": "Cell Phones",
  "top_k": 10
}
```

## Run The Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501.

## Docker

Run API and frontend together:

```bash
docker compose up --build
```

The compose file mounts both `data/raw` and `data/processed`. If processed CSVs are missing, the container runs `python -m app.ensure_data` and rebuilds them from `data/raw/*.jsonl.gz`.

Open:

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501

## Evaluation

```bash
python -m app.evaluate
```

This writes `outputs/evaluation_metrics.json` with profile coverage, recommendation similarity, and memory statistics.

Current evaluator includes:

- Review rating RMSE
- ROUGE-L review text score
- Recommendation Hit Rate@10
- Recommendation NDCG@10
- memory and embedding coverage

## Dataset

The project now uses real Amazon review-level files when present:

- `All_Beauty.jsonl.gz`
- `Appliances.jsonl.gz`
- `Gift_Cards.jsonl.gz`

Metadata files are used to enrich product IDs with readable names, categories, stores, prices, features, descriptions, and images where available.

## Solution Paper

The solution paper is available at `docs/solution_paper.md`. It covers architecture, dataset choices, experiments, ablations, metrics, Nigerian contextualization, limitations, and future work.
