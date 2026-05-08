import ast

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _as_embedding(value):
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, list):
        return np.array(value)

    if isinstance(value, str):
        return np.array(ast.literal_eval(value))

    raise TypeError(f"Unsupported embedding value: {type(value)!r}")


def recommend_similar_products(item_profiles, product_id, top_k=5):
    target = item_profiles[item_profiles["product_id"] == product_id]

    if target.empty:
        return None

    target_embedding = _as_embedding(target.iloc[0]["embedding"]).reshape(1, -1)
    similarities = []

    for _, row in item_profiles.iterrows():
        emb = _as_embedding(row["embedding"]).reshape(1, -1)
        sim = cosine_similarity(target_embedding, emb)[0][0]
        similarities.append(sim)

    recommendations = item_profiles.copy()
    recommendations["similarity"] = similarities

    columns = [
        "product_id",
        "avg_rating",
        "similarity",
    ]
    optional_columns = ["product_name", "category", "store", "price", "image_url"]
    columns.extend([column for column in optional_columns if column in recommendations.columns])

    return recommendations.sort_values("similarity", ascending=False).head(top_k)[columns]


def load_item_profiles(path="data/processed/item_profiles.csv"):
    return pd.read_csv(path)
