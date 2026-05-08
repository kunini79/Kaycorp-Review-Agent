import gzip
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "clean_reviews.csv"


USER_SEGMENTS = [
    {
        "user_id": "lagos_student_001",
        "persona_label": "budget-conscious student in Lagos",
        "preferred_terms": ["case", "charger", "phone", "earbuds", "software"],
        "rating_bias": 0.05,
        "verbosity": "short",
    },
    {
        "user_id": "abuja_parent_002",
        "persona_label": "practical parent in Abuja",
        "preferred_terms": ["health", "household", "protector", "cover", "accessory"],
        "rating_bias": 0.0,
        "verbosity": "medium",
    },
    {
        "user_id": "phc_techie_003",
        "persona_label": "tech enthusiast in Port Harcourt",
        "preferred_terms": ["phone", "software", "wireless", "usb", "bluetooth"],
        "rating_bias": 0.1,
        "verbosity": "detailed",
    },
    {
        "user_id": "kano_trader_004",
        "persona_label": "value-focused trader in Kano",
        "preferred_terms": ["pack", "replacement", "protector", "battery", "charger"],
        "rating_bias": -0.05,
        "verbosity": "short",
    },
    {
        "user_id": "ibadan_worker_005",
        "persona_label": "office worker in Ibadan",
        "preferred_terms": ["software", "keyboard", "mouse", "office", "case"],
        "rating_bias": 0.0,
        "verbosity": "medium",
    },
    {
        "user_id": "enugu_creator_006",
        "persona_label": "content creator in Enugu",
        "preferred_terms": ["camera", "tripod", "phone", "light", "wireless"],
        "rating_bias": 0.08,
        "verbosity": "detailed",
    },
    {
        "user_id": "lekki_gamer_007",
        "persona_label": "mobile gamer in Lekki",
        "preferred_terms": ["gaming", "controller", "phone", "screen", "headset"],
        "rating_bias": 0.02,
        "verbosity": "medium",
    },
    {
        "user_id": "kaduna_nurse_008",
        "persona_label": "health-conscious nurse in Kaduna",
        "preferred_terms": ["health", "household", "medical", "care", "clean"],
        "rating_bias": 0.06,
        "verbosity": "medium",
    },
    {
        "user_id": "yaba_developer_009",
        "persona_label": "software developer in Yaba",
        "preferred_terms": ["software", "usb", "laptop", "keyboard", "phone"],
        "rating_bias": -0.02,
        "verbosity": "detailed",
    },
    {
        "user_id": "ikeja_reseller_010",
        "persona_label": "electronics reseller in Ikeja",
        "preferred_terms": ["case", "charger", "accessory", "replacement", "pack"],
        "rating_bias": -0.04,
        "verbosity": "short",
    },
]


NIGERIAN_SNIPPETS = [
    "delivery was sharp",
    "for the price ehn, it made sense",
    "I cannot lie, it did the job",
    "actually not bad for everyday use",
    "this thing surprised me small",
]


def stable_seed(value):
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def first_image(images):
    if not isinstance(images, list) or not images:
        return ""

    first = images[0]
    if not isinstance(first, dict):
        return ""

    return first.get("hi_res") or first.get("large") or first.get("thumb") or ""


def compact_text(value):
    if isinstance(value, list):
        return " ".join(str(item) for item in value[:3] if item)

    if value is None:
        return ""

    return str(value)


def choose_users(product):
    title = str(product.get("title", "")).lower()
    category = str(product.get("main_category", "")).lower()
    text = f"{title} {category}"

    scored = []
    for user in USER_SEGMENTS:
        score = sum(1 for term in user["preferred_terms"] if term in text) * 3
        score += stable_seed(product.get("product_id", "") + user["user_id"]) % 5
        scored.append((score, user))

    scored.sort(key=lambda item: item[0], reverse=True)
    review_number = int(product.get("rating_number") or 1)
    interaction_count = min(5, max(2, review_number // 100 + 1))

    return [user for _, user in scored[:interaction_count]]


def synthetic_review(product, user, rating):
    product_name = product["product_name"]
    category = product.get("category", "product")
    snippet = NIGERIAN_SNIPPETS[stable_seed(product_name + user["user_id"]) % len(NIGERIAN_SNIPPETS)]

    if rating >= 4:
        opinion = "worked well and felt reliable"
    elif rating >= 3:
        opinion = "was okay, though not perfect"
    else:
        opinion = "could be better for the price"

    if user["verbosity"] == "detailed":
        return (
            f"As a {user['persona_label']}, I tried {product_name}. "
            f"The {category} option {opinion}; {snippet}. "
            f"I paid attention to build quality, usefulness, and whether it would hold up with regular Nigerian day-to-day use."
        )

    if user["verbosity"] == "medium":
        return (
            f"I used {product_name} and it {opinion}. "
            f"{snippet}. I would consider it if the price stays reasonable."
        )

    return f"{product_name} {opinion}; {snippet}."


def iter_metadata(limit_per_file=1700):
    raw_files = sorted(RAW_DIR.glob("meta_*.jsonl.gz"))

    for raw_file in raw_files:
        with gzip.open(raw_file, "rt", encoding="utf-8") as handle:
            emitted = 0
            for line in handle:
                if emitted >= limit_per_file:
                    break

                row = json.loads(line)
                product_id = row.get("parent_asin")
                title = row.get("title")
                rating = row.get("average_rating")

                if not product_id or not title or rating is None:
                    continue

                product = {
                    "product_id": str(product_id),
                    "product_name": str(title).strip(),
                    "rating": float(rating),
                    "rating_number": int(row.get("rating_number") or 0),
                    "category": row.get("main_category") or "",
                    "store": row.get("store") or "",
                    "price": row.get("price"),
                    "features": compact_text(row.get("features")),
                    "description": compact_text(row.get("description")),
                    "image_url": first_image(row.get("images")),
                }

                yield product
                emitted += 1


def repair_clean_reviews(limit_per_file=1700):
    interactions = []

    for product in iter_metadata(limit_per_file=limit_per_file):
        for user in choose_users(product):
            rng = random.Random(stable_seed(product["product_id"] + user["user_id"]))
            rating = product["rating"] + user["rating_bias"] + rng.uniform(-0.35, 0.35)
            rating = round(min(5.0, max(1.0, rating)), 1)

            interactions.append(
                {
                    "user_id": user["user_id"],
                    "persona_label": user["persona_label"],
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "rating": rating,
                    "review_text": synthetic_review(product, user, rating),
                    "category": product["category"],
                    "store": product["store"],
                    "price": product["price"],
                    "rating_number": product["rating_number"],
                    "features": product["features"],
                    "description": product["description"],
                    "image_url": product["image_url"],
                    "is_synthetic_interaction": True,
                }
            )

    df = pd.DataFrame(interactions)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    repaired = repair_clean_reviews()
    print(f"Saved {len(repaired):,} repaired interactions to {OUTPUT_PATH}")
    print(f"Users: {repaired['user_id'].nunique():,}")
    print(f"Products: {repaired['product_id'].nunique():,}")
