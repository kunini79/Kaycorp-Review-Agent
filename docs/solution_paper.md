# Kaycorp Intelligent Review Agent Solution Paper

## 1. Executive Summary

This project builds a review intelligence system for customer modeling, recommendations, and review generation. The system uses real Amazon review data from three domains: All Beauty, Appliances, and Gift Cards. From this review history, it creates user memory, item memory, item embeddings, review generation workflows, and personalized recommendation endpoints.

The goal is to model people as behavioral agents rather than static IDs. Each user profile captures rating tendency, review count, review length, and sample writing behavior. Each item profile captures average rating, review count, review memory, product name, category, store, price, features, and metadata when available. The API then exposes two challenge-facing workflows:

- Task A: take user persona and product details as input, then generate a predicted star rating and a written review.
- Task B: take user persona and context as input, then return personalized recommendations.

The current implementation is intentionally reproducible and modular. It uses `sentence-transformers/all-MiniLM-L6-v2` for semantic retrieval, a transparent rating baseline for RMSE evaluation, and a deterministic persona-conditioned review generator with light Nigerian English contextualization.

## 2. Dataset

The system uses real Amazon review-level files:

- `All_Beauty.jsonl.gz`
- `Appliances.jsonl.gz`
- `Gift_Cards.jsonl.gz`

Each review file contains `user_id`, `asin`, `parent_asin`, `rating`, `title`, `text`, `timestamp`, `helpful_vote`, and `verified_purchase`. Product metadata files are used to enrich product IDs with readable titles, categories, brands or stores, descriptions, features, prices, and images.

The processed build currently contains:

- 6,960 real review interactions
- 750 real users
- 4,455 products
- 747 users with like memory
- 100% item embedding coverage
- 100% persona coverage

The ingestion step samples active users with enough history to support behavioral modeling. This improves the quality of the user profiles and allows leave-one-out evaluation for recommendation.

## 3. Data Pipeline

The pipeline begins with `app/ingest_reviews.py`. It reads review JSONL files, maps review fields into a common schema, filters very short text, keeps active users, and enriches products from metadata files. The output is `data/processed/clean_reviews.csv`.

The profile builder then creates three memory tables:

- `user_profiles.csv`: one row per user with average rating, review count, average review length, sample review memory, and natural-language persona.
- `item_profiles.csv`: one row per product with average rating, review count, sample review memory, metadata fields, and semantic embedding.
- `user_likes.csv`: each user's liked products, where liked means rating >= 4.

This gives the agent two complementary memory stores: user memory and item memory.

## 4. User Modeling Approach

For Task A, the agent needs to understand a user's tone, rating behavior, and contextual nuance. The user profile builder converts numeric behavior into a plain-language customer profile. For example, a user with high average rating is described as generally positive, while a user with lower average rating is described as more critical. Review length is used as a proxy for verbosity.

The profile includes:

- rating tendency,
- activity level,
- average review length,
- sample review language,
- a natural-language behavioral summary.

When the API receives a raw persona, the service embeds that persona and finds the nearest real behavioral user profile. This makes cold-start personas usable even when no exact `user_id` exists.

## 5. Rating Prediction

The current rating predictor is a transparent hybrid baseline:

```text
predicted_rating = 0.45 * item_mean + 0.40 * user_mean + 0.15 * global_mean
```

For raw persona input, the nearest behavioral user profile supplies the user-mean signal. If the exact product exists, its item average is used. If the product is cold-start, the system embeds the product details and uses the closest item memory to estimate an item-rating signal.

This model is simple, explainable, and reproducible. It is not the final ceiling of performance, but it gives the system a measurable rating simulation path and avoids returning arbitrary stars.

## 6. Review Generation

The review generator takes:

- user persona,
- product name or product details,
- predicted rating.

The predicted rating determines the sentiment band:

- 4-5: positive
- 3-4: neutral or balanced
- below 3: negative

The generator then writes a concise review conditioned on the persona. Nigerian contextualization is added through restrained phrases such as "for the price ehn", "delivery was sharp", and "I cannot lie". This is deliberately light-touch so the output sounds locally aware without becoming exaggerated.

The current review generator is deterministic for reproducibility. The phrase choice is derived from a stable hash of persona, product, and rating, so repeated evaluations produce the same output.

## 7. Recommendation Approach

For Task B, each item profile is embedded using `sentence-transformers/all-MiniLM-L6-v2`. The recommender supports three modes:

- product-to-product semantic similarity,
- user-history recommendations from liked items,
- cold-start persona/context recommendations from raw natural language.

The challenge-facing endpoint `/task-b/recommend` accepts persona, context, product history, optional category, and `top_k`. It embeds the persona and context, compares the query against item memory embeddings, filters seen products, and returns ranked products with similarity scores and product metadata.

This goes beyond a pure collaborative-filtering table because it can recommend from raw language context, not only from known user IDs.

## 8. Agentic Workflow

The agent follows a simple reasoning pattern:

1. Read persona and context.
2. Convert persona/context into a semantic query.
3. Retrieve candidate items from item memory.
4. Filter products already in history.
5. Rank by semantic similarity, rating, and review-count tie breakers.
6. Return recommendations with reasoning.

For multiturn behavior, `/task-b/conversation-turn` accepts updated context and reruns retrieval. This allows the user to refine preferences such as budget, category, durability, delivery speed, or domain.

## 9. API and Application

The FastAPI service exposes:

- `GET /health`
- `GET /products`
- `GET /users`
- `GET /recommendations/product/{product_id}`
- `GET /recommendations/user/{user_id}`
- `POST /reviews/generate`
- `POST /task-a/generate-review`
- `POST /task-b/recommend`
- `POST /task-b/conversation-turn`

The Streamlit app provides a visual demo with recommendation, review generation, and persona inspection tabs. The application is containerized with Docker and Docker Compose.

## 10. Evaluation

The evaluation script reports memory coverage and challenge-facing metrics.

Current metrics:

```json
{
  "memory": {
    "item_profiles": 22287,
    "user_profiles": 750,
    "users_with_like_memory": 749,
    "embedding_coverage": 1.0,
    "persona_coverage": 1.0
  },
  "task_a_user_modeling": {
    "samples": 150,
    "rating_rmse": 1.3754,
    "mean_rouge_l": 0.0848
  },
  "task_b_recommendation": {
    "users_evaluated": 300,
    "hit_rate_at_10": 0.03,
    "ndcg_at_10": 0.0246
  }
}
```

Task A uses leave-one-out rows for rating prediction. ROUGE-L compares the generated review with the held-out review text. Task B uses leave-one-liked-item-out per user. The system recommends from previous liked item embeddings and checks whether the held-out liked item appears in the top 10.

## 11. Experiments and Ablations

The project progressed through three stages.

First, a metadata-only prototype was built from Amazon `meta_*.jsonl.gz` files. This exposed a major limitation: metadata files contain products but not real users. All rows collapsed into `unknown_user`, which was not acceptable for a behavioral modeling challenge.

Second, a synthetic-user repair was created to make the demo usable while real review data was downloading. This produced synthetic Nigerian shopper personas and product interactions, but it was treated only as an emergency fallback.

Third, the real Amazon review files were added and the pipeline was rebuilt around actual reviewer IDs, ratings, timestamps, and review text. This is the current and valid competition path.

Ablation observations:

- Product metadata alone is insufficient for user modeling.
- Real user IDs are essential for persona and rating-behavior modeling.
- Semantic item memory improves cold-start querying because a raw persona can retrieve products without an exact user ID.
- Recommendation metrics are conservative because many products have only one review and the catalog is large.

## 12. Nigerian Contextualization

The competition gives additional marks for Nigerian behavior and language. The system includes Nigerian contextualization in three places:

- review generation phrase bank,
- cold-start persona examples,
- prompt-level emphasis on price sensitivity, delivery speed, and practical durability.

The design avoids heavy slang. Nigerian English is used as a light behavioral signal rather than a stereotype.

## 12.1 Cold-Start and Cross-Domain Handling

Cold-start is handled through raw persona and product-detail embeddings. A new user does not need a stored `user_id`; the `/task-b/recommend` endpoint can accept a natural-language persona and retrieve relevant products directly from item memory. For Task A, `/task-a/generate-review` accepts a raw persona plus product details, then finds the nearest behavioral user profile to estimate rating style.

Cross-domain behavior is supported by the three-domain dataset. The same recommendation workflow can search across Cell Phones and Accessories, Health and Household, and Software. The optional `category` field can constrain recommendations to a domain, while leaving it blank allows cross-domain retrieval from the full product memory.

## 13. Limitations

The current model is a strong reproducible baseline, but there are clear limitations.

The review generator is template-based, so ROUGE-L is low compared with a fine-tuned generative model. The rating predictor is an explainable mean-based hybrid rather than a learned ranking/rating model. Recommendation performance is also limited by sparse interactions and the large number of products with only one or two reviews.

The system supports multiturn refinement through repeated context, but it does not yet maintain a persistent conversation state. It can handle cold-start personas through semantic retrieval, but richer agent planning would improve contextual relevance.

## 14. Future Work

With more time, I would add:

- a matrix factorization or neural collaborative model for stronger rating prediction,
- a learned reranker for recommendations,
- BERTScore evaluation for review quality,
- persistent conversation sessions,
- a vector database for faster retrieval,
- a stronger LLM review generator with disclosure guardrails,
- Nigerian market-specific prompts and evaluation examples,
- a presentation deck with screenshots and live demo flow.

## 15. Reproducibility

The project is reproducible with:

```bash
pip install -r requirements.txt
python main.py
python -m app.evaluate
uvicorn app.api:app --reload
streamlit run streamlit_app.py
```

Docker support is included through `Dockerfile` and `docker-compose.yml`. The README documents the endpoints, evaluation command, and dataset expectations.

## 16. Implementation Details

The codebase is structured so each layer has a clear responsibility. `app/ingest_reviews.py` handles raw review files and metadata enrichment. `app/profile_builder.py` converts the cleaned interaction table into memory artifacts. `app/recommender.py` contains item-to-item similarity logic. `app/service.py` is the shared application layer used by both FastAPI and Streamlit. `app/api.py` exposes the challenge endpoints. `app/evaluate.py` runs the reproducible metric suite.

This separation matters for judging because each workflow can be inspected independently. The ingestion layer can be rerun when new data arrives. The profile layer can be improved without changing the API. The API can expose new tasks without rewriting the embedding or evaluation logic.

The processed data files are intentionally stored as CSV because they are easy for judges to inspect. Embeddings are serialized as lists in `item_profiles.csv`. This is not the fastest production format, but it is transparent and portable. For production use, these vectors should move into a vector database such as FAISS, Qdrant, Weaviate, or pgvector.

## 17. How The App Meets The Deliverables

Task 1 requires a containerized application that takes user persona and product details as input, then generates reviews and ratings. The endpoint `POST /task-a/generate-review` does this directly. It accepts raw persona text and product details. It returns a predicted rating, generated review, and nearest behavioral user used as an explanation signal.

Task 2 requires a containerized application that takes user persona as input and produces personalized recommendations. The endpoint `POST /task-b/recommend` does this directly. It accepts persona, context, history, category, and `top_k`. It returns ranked products with metadata and similarity scores.

The code repository requirement is addressed through a documented modular project with setup instructions, Docker support, API docs, Streamlit demo, evaluation script, and solution paper. Before final submission, the repository should be committed and pushed to GitHub.

## 18. Interpreting The Scores

The RMSE score should be interpreted as a baseline, not as a final optimized model. It is produced by a transparent hybrid mean model. This helps demonstrate the evaluation protocol and gives a measurable starting point. A learned model would likely improve RMSE.

The ROUGE-L score is also conservative. Reviews are highly personal and often contain details that are not predictable from product metadata alone. A generated review can be behaviorally plausible while still having low lexical overlap with the held-out review. For that reason, the project also emphasizes behavioral fidelity and human-readable persona inspection.

The Hit Rate@10 and NDCG@10 values are low but honest. The catalog contains more than 22,000 products, and many products have very few reviews. The evaluation asks the model to recover one held-out liked item from a very large catalog. In future work, a learned reranker and stronger user sequence model would be the first improvements.

## 19. Why This Is Agentic

The system is agentic in the practical sense that it performs multiple reasoning steps before output. It does not only look up a user ID. For Task A, it maps persona to a behavioral neighbor, estimates rating from user and item signals, and generates a review conditioned on that rating. For Task B, it turns persona and context into a semantic retrieval query, retrieves from item memory, filters history, and returns ranked recommendations with reasoning.

The multiturn endpoint is a minimal version of conversational retrieval. It allows the user to revise context and get a new ranking. For example, a user can start with "I need a phone case" and then refine with "must be cheap and good for Lagos delivery". The agent reruns retrieval with the updated context.

## 20. Submission Notes

The strongest final submission should include screenshots or a short screen recording of:

- FastAPI docs showing Task A and Task B endpoints,
- a Task A request and response,
- a Task B request and response,
- Streamlit recommendation tab,
- Streamlit review generation tab,
- `python -m app.evaluate` output.

The paper should be submitted as PDF if the submission form allows it. This Markdown version can be converted to PDF with any Markdown editor or document tool.
