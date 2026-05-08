import json
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.review_agent import generate_review
from app.service import load_item_profiles, load_user_likes, load_user_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation_metrics.json"
CLEAN_REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"


def load_reviews():
    return pd.read_csv(CLEAN_REVIEWS_PATH)


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def rouge_l_score(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    rows = len(pred_tokens) + 1
    cols = len(ref_tokens) + 1
    table = [[0] * cols for _ in range(rows)]

    for i, pred_token in enumerate(pred_tokens, start=1):
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    lcs = table[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def evaluate_recommendations(sample_size=25, top_k=5):
    items = load_item_profiles()
    sample_size = min(sample_size, len(items))
    sample_ids = items["product_id"].head(sample_size).tolist()
    similarity_scores = []
    recommended_product_ids = set()

    embeddings = np.vstack(
        items["embedding"].apply(lambda value: np.array(ast.literal_eval(value))).to_numpy()
    )
    sample_embeddings = embeddings[:sample_size]
    similarity_matrix = cosine_similarity(sample_embeddings, embeddings)

    for sample_index, product_id in enumerate(sample_ids):
        ranked_indices = np.argsort(similarity_matrix[sample_index])[::-1]
        non_self = [index for index in ranked_indices if index != sample_index][:top_k]

        for index in non_self:
            similarity_scores.append(float(similarity_matrix[sample_index][index]))
            recommended_product_ids.add(items.iloc[index]["product_id"])

    if similarity_scores:
        mean_similarity = float(np.mean(similarity_scores))
        median_similarity = float(np.median(similarity_scores))
    else:
        mean_similarity = 0.0
        median_similarity = 0.0

    return {
        "sampled_seed_products": len(sample_ids),
        "top_k": top_k,
        "mean_non_self_similarity": mean_similarity,
        "median_non_self_similarity": median_similarity,
        "unique_recommended_products": len(recommended_product_ids),
    }


def evaluate_task_a(max_samples=150):
    reviews = load_reviews().copy()
    users = load_user_profiles()
    items = load_item_profiles()

    reviews = reviews.sort_values(["user_id", "timestamp"], na_position="last")
    test_rows = reviews.groupby("user_id").tail(1).head(max_samples)
    global_rating = float(reviews["rating"].mean())

    user_stats = reviews.groupby("user_id")["rating"].agg(["sum", "count"])
    item_stats = reviews.groupby("product_id")["rating"].agg(["sum", "count"])
    user_personas = users.set_index("user_id")["persona"].to_dict()
    item_names = items.set_index("product_id")["product_name"].to_dict()

    predictions = []
    actuals = []
    rouge_scores = []

    for _, row in test_rows.iterrows():
        user = user_stats.loc[row["user_id"]]
        item = item_stats.loc[row["product_id"]]

        user_count = max(0, int(user["count"]) - 1)
        item_count = max(0, int(item["count"]) - 1)

        user_avg = (
            (float(user["sum"]) - float(row["rating"])) / user_count
            if user_count > 0
            else global_rating
        )
        item_avg = (
            (float(item["sum"]) - float(row["rating"])) / item_count
            if item_count > 0
            else global_rating
        )
        prediction = round(min(5.0, max(1.0, 0.45 * item_avg + 0.4 * user_avg + 0.15 * global_rating)), 2)

        persona = user_personas.get(row["user_id"], "")
        product_name = item_names.get(row["product_id"], row["product_id"])
        generated_review = generate_review(persona, product_name, prediction)

        predictions.append(prediction)
        actuals.append(float(row["rating"]))
        rouge_scores.append(rouge_l_score(generated_review, row["review_text"]))

    rmse = float(np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)))

    return {
        "samples": int(len(test_rows)),
        "rating_rmse": rmse,
        "mean_rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "rating_model": "0.45 item mean + 0.40 user mean + 0.15 global mean with leave-one-out test rows",
        "review_model": "persona-conditioned deterministic template with Nigerian English phrase injection",
    }


def evaluate_task_b(max_users=300, top_k=10):
    reviews = load_reviews().copy()
    items = load_item_profiles().reset_index(drop=True)
    embeddings = np.vstack(
        items["embedding"].apply(lambda value: np.array(ast.literal_eval(value))).to_numpy()
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-12)
    product_to_index = {product_id: index for index, product_id in enumerate(items["product_id"])}

    liked = reviews[reviews["rating"] >= 4].sort_values(["user_id", "timestamp"], na_position="last")
    hits = []
    ndcgs = []
    evaluated = 0

    for user_id, group in liked.groupby("user_id"):
        user_items = [item for item in group["product_id"].tolist() if item in product_to_index]
        unique_items = list(dict.fromkeys(user_items))

        if len(unique_items) < 2:
            continue

        heldout = unique_items[-1]
        history = unique_items[:-1]
        seed_items = history[-3:]
        query = normalized_embeddings[[product_to_index[item] for item in seed_items]].mean(axis=0)
        query = query / max(np.linalg.norm(query), 1e-12)
        similarities = normalized_embeddings @ query

        for seen_item in history:
            similarities[product_to_index[seen_item]] = -np.inf

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        ranked_products = [items.iloc[index]["product_id"] for index in ranked_indices]

        if heldout in ranked_products:
            rank = ranked_products.index(heldout) + 1
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

        evaluated += 1
        if evaluated >= max_users:
            break

    return {
        "users_evaluated": int(evaluated),
        "top_k": int(top_k),
        "hit_rate_at_10": float(np.mean(hits)) if hits else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "protocol": "Leave-one-liked-item-out per user; recommend from previous liked item embeddings.",
    }


def evaluate_memory():
    items = load_item_profiles()
    users = load_user_profiles()
    user_likes = load_user_likes()

    return {
        "item_profiles": int(len(items)),
        "user_profiles": int(len(users)),
        "users_with_like_memory": int(len(user_likes)),
        "avg_item_rating": float(items["avg_rating"].mean()),
        "avg_item_review_count": float(items["review_count"].mean()),
        "avg_user_review_count": float(users["review_count"].mean()),
        "avg_user_review_length": float(users["avg_review_length"].mean()),
        "embedding_coverage": float(items["embedding"].notna().mean()),
        "persona_coverage": float(users["persona"].notna().mean()),
    }


def run_evaluation():
    metrics = {
        "memory": evaluate_memory(),
        "recommendations": evaluate_recommendations(),
        "task_a_user_modeling": evaluate_task_a(),
        "task_b_recommendation": evaluate_task_b(),
        "notes": [
            "Similarity metrics exclude the seed product itself.",
            "User-level metrics are built from real review-level user IDs in the Amazon review files.",
            "ROUGE-L is a lightweight local implementation to keep the repo reproducible without extra downloads.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
