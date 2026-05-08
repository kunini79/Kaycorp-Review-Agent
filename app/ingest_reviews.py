import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "clean_reviews.csv"


def review_files():
    return sorted(
        path
        for path in RAW_DIR.glob("*Reviews*gz")
        if path.name.lower().startswith(("cell_", "health_", "software_"))
    )


def metadata_files():
    return sorted(RAW_DIR.glob("meta_*.jsonl.gz"))


def category_from_filename(path):
    name = path.name.replace("_Reviews", "").replace("_jsonl", "")
    name = name.replace(".jsonl.gz", "").replace(".gz", "")
    return name.replace("_", " ")


def read_review_candidates(max_per_file=20000, min_text_chars=20):
    rows = []

    for path in review_files():
        category = category_from_filename(path)
        kept = 0

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if kept >= max_per_file:
                    break

                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                user_id = str(row.get("user_id") or "").strip()
                product_id = str(row.get("parent_asin") or row.get("asin") or "").strip()
                rating = row.get("rating")

                if not user_id or not product_id or rating is None:
                    continue

                if len(text) < min_text_chars:
                    continue

                rows.append(
                    {
                        "user_id": user_id,
                        "product_id": product_id,
                        "asin": row.get("asin"),
                        "rating": float(rating),
                        "review_title": row.get("title") or "",
                        "review_text": text,
                        "category": category,
                        "timestamp": row.get("timestamp"),
                        "helpful_vote": row.get("helpful_vote", 0),
                        "verified_purchase": row.get("verified_purchase"),
                    }
                )
                kept += 1

    return pd.DataFrame(rows)


def keep_active_users(df, min_reviews=3, max_users=750):
    counts = Counter(df["user_id"])
    active_users = [user_id for user_id, count in counts.most_common() if count >= min_reviews]
    active_users = active_users[:max_users]

    filtered = df[df["user_id"].isin(active_users)].copy()
    if filtered.empty:
        return df

    return filtered


def build_metadata_lookup(product_ids, max_lines_per_file=300000):
    wanted = set(product_ids)
    lookup = {}

    for path in metadata_files():
        scanned = 0

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if scanned >= max_lines_per_file or wanted.issubset(lookup.keys()):
                    break

                row = json.loads(line)
                product_id = str(row.get("parent_asin") or "").strip()
                if product_id in wanted and product_id not in lookup:
                    images = row.get("images")
                    image_url = ""
                    if isinstance(images, list) and images:
                        first = images[0]
                        if isinstance(first, dict):
                            image_url = (
                                first.get("hi_res")
                                or first.get("large")
                                or first.get("thumb")
                                or ""
                            )

                    lookup[product_id] = {
                        "product_name": row.get("title") or product_id,
                        "store": row.get("store") or "",
                        "price": row.get("price"),
                        "features": " ".join(str(x) for x in (row.get("features") or [])[:3]),
                        "description": " ".join(
                            str(x) for x in (row.get("description") or [])[:2]
                        ),
                        "image_url": image_url,
                        "metadata_category": row.get("main_category") or "",
                    }

                scanned += 1

    return lookup


def enrich_with_metadata(df):
    lookup = build_metadata_lookup(df["product_id"].unique())

    def value_for(row, key, fallback=""):
        metadata = lookup.get(row["product_id"], {})
        return metadata.get(key) or fallback

    df = df.copy()
    df["product_name"] = df.apply(
        lambda row: value_for(row, "product_name", row["product_id"]), axis=1
    )
    df["store"] = df.apply(lambda row: value_for(row, "store"), axis=1)
    df["price"] = df.apply(lambda row: value_for(row, "price", None), axis=1)
    df["features"] = df.apply(lambda row: value_for(row, "features"), axis=1)
    df["description"] = df.apply(lambda row: value_for(row, "description"), axis=1)
    df["image_url"] = df.apply(lambda row: value_for(row, "image_url"), axis=1)
    df["category"] = df.apply(
        lambda row: value_for(row, "metadata_category", row["category"]), axis=1
    )
    df["is_synthetic_interaction"] = False

    return df


def ingest_reviews(max_per_file=20000, min_reviews_per_user=3, max_users=750):
    df = read_review_candidates(max_per_file=max_per_file)
    if df.empty:
        raise ValueError("No review-level rows found in data/raw.")

    df = keep_active_users(df, min_reviews=min_reviews_per_user, max_users=max_users)
    df = enrich_with_metadata(df)
    df = df.sort_values(["user_id", "timestamp"], na_position="last").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    reviews = ingest_reviews()
    print(f"Saved {len(reviews):,} real review interactions to {OUTPUT_PATH}")
    print(f"Users: {reviews['user_id'].nunique():,}")
    print(f"Products: {reviews['product_id'].nunique():,}")
