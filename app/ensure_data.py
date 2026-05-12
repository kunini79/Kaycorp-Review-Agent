from pathlib import Path

from app.ingest_reviews import review_files
from app.profile_builder import build_all_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED_FILES = [
    PROCESSED_DIR / "clean_reviews.csv",
    PROCESSED_DIR / "user_profiles.csv",
    PROCESSED_DIR / "item_profiles.csv",
    PROCESSED_DIR / "user_likes.csv",
]


def processed_data_ready():
    return all(path.exists() and path.stat().st_size > 0 for path in REQUIRED_FILES)


def ensure_data():
    if processed_data_ready():
        print("Processed data is ready.")
        return

    if not review_files():
        raise FileNotFoundError(
            "Processed data is missing and no review files were found. "
            "Place Amazon review files in data/raw/*.jsonl.gz, then run python main.py."
        )

    print("Processed data missing. Building from review files...")
    from app.ingest_reviews import ingest_reviews

    ingest_reviews()
    build_all_profiles(include_embeddings=True)
    print("Processed data build complete.")


if __name__ == "__main__":
    ensure_data()
