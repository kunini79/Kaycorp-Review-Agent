from pathlib import Path
import hashlib
import math
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_reviews(path=DEFAULT_CLEAN_REVIEWS_PATH):
    df = pd.read_csv(path)
    required_columns = {"user_id", "product_id", "rating", "review_text"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "rating"])

    return df


def build_item_profiles(df):
    aggregations = {
        "rating": ["mean", "count"],
        "review_text": lambda x: " ".join(x.head(3)),
    }

    optional_columns = [
        "product_name",
        "category",
        "store",
        "price",
        "rating_number",
        "features",
        "description",
        "image_url",
    ]
    for column in optional_columns:
        if column in df.columns:
            aggregations[column] = "first"

    item_profiles = df.groupby("product_id").agg(aggregations)

    flattened_columns = []
    for column in item_profiles.columns:
        if column == ("rating", "mean"):
            flattened_columns.append("avg_rating")
        elif column == ("rating", "count"):
            flattened_columns.append("review_count")
        elif column == ("review_text", "<lambda>"):
            flattened_columns.append("sample_reviews")
        else:
            flattened_columns.append(column[0])

    item_profiles.columns = flattened_columns

    return item_profiles.reset_index()


def build_persona(row):
    if row["avg_rating"] >= 4:
        mood = "generally positive"
    elif row["avg_rating"] >= 3:
        mood = "balanced"
    else:
        mood = "critical"

    if row["avg_review_length"] > 300:
        style = "detailed reviewer"
    else:
        style = "short-form reviewer"

    base_identity = ""
    if "persona_label" in row and pd.notna(row["persona_label"]):
        base_identity = f"User is a {row['persona_label']}.\n"

    return f"""
{base_identity}User is a {mood} customer and a {style}.
Average rating given: {row['avg_rating']:.1f}
Total reviews: {row['review_count']}
Example behavior:
{row['sample_reviews'][:300]}
""".strip()


def build_user_profiles(df):
    df = df.copy()
    df["review_length"] = df["review_text"].apply(len)

    aggregations = {
        "rating": ["mean", "count"],
        "review_length": "mean",
        "review_text": lambda x: " ".join(x.head(5)),
    }
    if "persona_label" in df.columns:
        aggregations["persona_label"] = "first"

    user_profiles = df.groupby("user_id").agg(aggregations)

    flattened_columns = []
    for column in user_profiles.columns:
        if column == ("rating", "mean"):
            flattened_columns.append("avg_rating")
        elif column == ("rating", "count"):
            flattened_columns.append("review_count")
        elif column == ("review_length", "mean"):
            flattened_columns.append("avg_review_length")
        elif column == ("review_text", "<lambda>"):
            flattened_columns.append("sample_reviews")
        else:
            flattened_columns.append(column[0])

    user_profiles.columns = flattened_columns

    user_profiles = user_profiles.reset_index()
    user_profiles["persona"] = user_profiles.apply(build_persona, axis=1)

    return user_profiles


def build_user_likes(df, minimum_rating=4):
    liked_items = df[df["rating"] >= minimum_rating]
    user_likes = liked_items.groupby("user_id")["product_id"].apply(list)

    return user_likes.reset_index(name="liked_product_ids")


def add_item_embeddings(item_profiles, model_name="all-MiniLM-L6-v2"):
    item_profiles = item_profiles.copy()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        item_profiles["embedding"] = item_profiles["sample_reviews"].apply(
            lambda x: model.encode(str(x)).tolist()
        )
    except ModuleNotFoundError:
        def hashed_embedding(text, dimensions=384):
            vector = [0.0] * dimensions
            tokens = re.findall(r"[a-z0-9]+", str(text).lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % dimensions
                vector[index] += 1.0

            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            return vector

        item_profiles["embedding"] = item_profiles["sample_reviews"].apply(hashed_embedding)

    return item_profiles


def save_profiles(user_profiles, item_profiles, user_likes, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_profiles.to_csv(output_dir / "user_profiles.csv", index=False)
    item_profiles.to_csv(output_dir / "item_profiles.csv", index=False)
    user_likes.to_csv(output_dir / "user_likes.csv", index=False)


def build_all_profiles(include_embeddings=True):
    df = load_reviews()
    item_profiles = build_item_profiles(df)
    user_profiles = build_user_profiles(df)
    user_likes = build_user_likes(df)

    if include_embeddings:
        item_profiles = add_item_embeddings(item_profiles)

    save_profiles(user_profiles, item_profiles, user_likes)

    return user_profiles, item_profiles, user_likes


if __name__ == "__main__":
    users, items, likes = build_all_profiles(include_embeddings=True)
    print(f"Saved {len(users)} user profiles")
    print(f"Saved {len(items)} item profiles")
    print(f"Saved {len(likes)} user like histories")
    if not users.empty:
        print()
        print(users["persona"].iloc[0])
