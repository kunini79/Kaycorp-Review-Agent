from app.ingest_reviews import ingest_reviews, review_files
from app.profile_builder import build_all_profiles


if __name__ == "__main__":
    if review_files():
        ingest_reviews()
    build_all_profiles(include_embeddings=True)
