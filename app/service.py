import ast
import contextlib
import io
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.recommender import recommend_similar_products
from app.review_agent import generate_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ITEM_PROFILES_PATH = PROCESSED_DIR / "item_profiles.csv"
USER_PROFILES_PATH = PROCESSED_DIR / "user_profiles.csv"
USER_LIKES_PATH = PROCESSED_DIR / "user_likes.csv"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _processed_path(filename):
    # Allow deployment environments to override the processed-data location.
    env_dir = os.getenv("KAYCORP_PROCESSED_DIR")
    if env_dir:
        return Path(env_dir) / filename
    return PROCESSED_DIR / filename


def _ensure_processed_file(path):
    if path.exists():
        return

    # Attempt to bootstrap processed CSVs from raw review files when available.
    from app.ensure_data import ensure_data

    try:
        ensure_data()
    except Exception as exc:
        raise FileNotFoundError(
            "Processed data is missing and auto-build failed. "
            f"Expected file: {path}. "
            "Set KAYCORP_PROCESSED_DIR to the folder containing item_profiles.csv, "
            "user_profiles.csv, and user_likes.csv, or add raw review files under data/raw "
            "and run `python -m app.ensure_data`."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(
            "Processed data build completed but required file is still missing. "
            f"Expected file: {path}."
        )


@lru_cache(maxsize=1)
def load_item_profiles():
    path = _processed_path("item_profiles.csv")
    _ensure_processed_file(path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_user_profiles():
    path = _processed_path("user_profiles.csv")
    _ensure_processed_file(path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_user_likes():
    path = _processed_path("user_likes.csv")

    if not path.exists():
        return pd.DataFrame(columns=["user_id", "liked_product_ids"])

    user_likes = pd.read_csv(path)
    if "liked_product_ids" in user_likes.columns:
        user_likes["liked_product_ids"] = user_likes["liked_product_ids"].apply(_safe_list)

    return user_likes


@lru_cache(maxsize=1)
def load_embedding_matrix():
    items = load_item_profiles()
    embeddings = np.vstack(
        items["embedding"].apply(lambda value: np.array(ast.literal_eval(value))).to_numpy()
    )

    return embeddings


@lru_cache(maxsize=1)
def load_sentence_model():
    from huggingface_hub.utils import disable_progress_bars
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    transformers_logging.disable_progress_bar()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer("all-MiniLM-L6-v2")


def _safe_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []

    return parsed if isinstance(parsed, list) else []


def _clean_value(value):
    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and math.isnan(value):
        return None

    if pd.isna(value):
        return None

    return value


def _clean_record(record):
    return {key: _clean_value(value) for key, value in record.items()}


def _clean_records(records):
    return [_clean_record(record) for record in records]


def list_products(limit=25):
    items = load_item_profiles()
    columns = [
        "product_id",
        "product_name",
        "category",
        "store",
        "price",
        "avg_rating",
        "review_count",
        "sample_reviews",
    ]
    columns = [column for column in columns if column in items.columns]

    return _clean_records(items[columns].head(limit).to_dict(orient="records"))


def list_users(limit=25):
    users = load_user_profiles()
    columns = [
        "user_id",
        "persona_label",
        "avg_rating",
        "review_count",
        "avg_review_length",
        "persona",
    ]
    columns = [column for column in columns if column in users.columns]

    return _clean_records(users[columns].head(limit).to_dict(orient="records"))


def get_product(product_id):
    items = load_item_profiles()
    product = items[items["product_id"] == product_id]

    if product.empty:
        return None

    columns = [
        "product_id",
        "product_name",
        "category",
        "store",
        "price",
        "avg_rating",
        "review_count",
        "sample_reviews",
        "features",
        "description",
        "image_url",
    ]
    columns = [column for column in columns if column in product.columns]
    return _clean_record(product.iloc[0][columns].to_dict())


def get_user(user_id):
    users = load_user_profiles()
    user = users[users["user_id"] == user_id]

    if user.empty:
        return None

    return _clean_record(user.iloc[0].to_dict())


def get_similar_products(product_id, top_k=5):
    items = load_item_profiles()
    recommendations = recommend_similar_products(items, product_id, top_k=top_k)

    if recommendations is None:
        return None

    return _clean_records(recommendations.to_dict(orient="records"))


def get_user_recommendations(user_id, top_k=5):
    likes = load_user_likes()
    user_row = likes[likes["user_id"] == user_id]

    if user_row.empty:
        return None

    liked_product_ids = user_row.iloc[0]["liked_product_ids"]
    if not liked_product_ids:
        return []

    seed_product_id = liked_product_ids[0]
    recommendations = get_similar_products(seed_product_id, top_k=top_k + 10)
    if recommendations is None:
        return None

    filtered = [
        item for item in recommendations if item["product_id"] not in set(liked_product_ids)
    ]

    return filtered[:top_k]


def _product_text(product_details):
    parts = []
    for key in ["product_name", "name", "title", "category", "store", "features", "description"]:
        value = product_details.get(key)
        if value:
            parts.append(str(value))

    return " ".join(parts).strip() or "product"


def predict_rating_from_persona(persona, product_details):
    users = load_user_profiles()
    items = load_item_profiles()

    persona_text = str(persona or "")
    product_text = _product_text(product_details)
    product_id = product_details.get("product_id")

    global_rating = float(items["avg_rating"].mean())
    product_rating = None
    if product_id:
        product = items[items["product_id"] == product_id]
        if not product.empty:
            product_rating = float(product.iloc[0]["avg_rating"])

    persona_embedding = load_sentence_model().encode(persona_text)
    user_embeddings = load_sentence_model().encode(users["persona"].fillna("").astype(str).tolist())
    user_similarity = cosine_similarity([persona_embedding], user_embeddings)[0]
    nearest_user_index = int(np.argmax(user_similarity))
    nearest_user = users.iloc[nearest_user_index]
    nearest_user_rating = float(nearest_user["avg_rating"])

    if product_rating is None:
        product_embedding = load_sentence_model().encode(product_text)
        item_similarities = cosine_similarity([product_embedding], load_embedding_matrix())[0]
        nearest_item_index = int(np.argmax(item_similarities))
        product_rating = float(items.iloc[nearest_item_index]["avg_rating"])

    predicted_rating = (0.45 * product_rating) + (0.4 * nearest_user_rating) + (0.15 * global_rating)
    predicted_rating = round(min(5.0, max(1.0, predicted_rating)), 1)

    return {
        "predicted_rating": predicted_rating,
        "nearest_user_id": nearest_user["user_id"],
        "nearest_user_similarity": float(user_similarity[nearest_user_index]),
    }


def generate_review_from_persona(persona, product_details):
    product_name = (
        product_details.get("product_name")
        or product_details.get("name")
        or product_details.get("title")
        or product_details.get("product_id")
        or "this product"
    )
    prediction = predict_rating_from_persona(persona, product_details)
    review = generate_review(persona, product_name, prediction["predicted_rating"])

    return _clean_record(
        {
            "product_name": product_name,
            "predicted_rating": prediction["predicted_rating"],
            "nearest_behavioral_user": prediction["nearest_user_id"],
            "nearest_user_similarity": prediction["nearest_user_similarity"],
            "review": review,
        }
    )


def recommend_from_persona(persona, context="", top_k=10, product_history=None, category=None):
    items = load_item_profiles().copy()
    query = " ".join(str(part) for part in [persona, context, " ".join(product_history or [])] if part)
    query_embedding = load_sentence_model().encode(query or "useful high quality product")
    similarities = cosine_similarity([query_embedding], load_embedding_matrix())[0]

    scored = items.copy()
    scored["similarity"] = similarities

    if category:
        category_mask = scored["category"].fillna("").str.contains(category, case=False, na=False)
        if category_mask.any():
            scored = scored[category_mask]

    if product_history:
        history = set(product_history)
        scored = scored[~scored["product_id"].isin(history)]

    columns = [
        "product_id",
        "product_name",
        "category",
        "store",
        "price",
        "avg_rating",
        "review_count",
        "similarity",
    ]
    columns = [column for column in columns if column in scored.columns]

    recommendations = scored.sort_values(
        ["similarity", "avg_rating", "review_count"], ascending=False
    ).head(top_k)

    reasoning = (
        "Matched the persona/context embedding against product review-memory embeddings, "
        "then ranked by semantic similarity with rating and review-count as tie breakers."
    )

    return {
        "reasoning": reasoning,
        "recommendations": _clean_records(recommendations[columns].to_dict(orient="records")),
    }


def generate_personalized_review(user_id, product_id=None, product_name=None, avg_rating=None):
    user = get_user(user_id)
    if user is None:
        return None

    product = None
    if product_id:
        product = get_product(product_id)

    resolved_name = product_name or product_id or "this product"
    resolved_rating = avg_rating

    if product is not None:
        resolved_name = (
            product_name
            or product.get("product_name")
            or product.get("sample_reviews", "")[:80]
            or product_id
        )
        resolved_rating = avg_rating if avg_rating is not None else product["avg_rating"]

    if resolved_rating is None:
        resolved_rating = user["avg_rating"]

    return {
        "user_id": user_id,
        "product_id": product_id,
        "product_name": resolved_name,
        "avg_rating": float(resolved_rating),
        "review": generate_review(user["persona"], resolved_name, float(resolved_rating)),
    }
