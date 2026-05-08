import pandas as pd
import streamlit as st

from app.service import (
    generate_review_from_persona,
    get_similar_products,
    get_user_recommendations,
    load_item_profiles,
    load_user_profiles,
)


st.set_page_config(
    page_title="BCT Review Intelligence Agent",
    page_icon="",
    layout="wide",
)


@st.cache_data
def cached_items():
    return load_item_profiles()


@st.cache_data
def cached_users():
    return load_user_profiles()


items = cached_items()
users = cached_users()

items["display_name"] = items.apply(
    lambda row: f"{row.get('product_name', row['product_id'])} ({row['product_id']})",
    axis=1,
)
users["display_name"] = users.apply(
    lambda row: f"{row.get('persona_label', row['user_id'])} ({row['user_id']})",
    axis=1,
)

st.title("BCT Review Intelligence Agent")

summary_cols = st.columns(3)
summary_cols[0].metric("Products", f"{len(items):,}")
summary_cols[1].metric("Users", f"{len(users):,}")
summary_cols[2].metric("Avg Product Rating", f"{items['avg_rating'].mean():.2f}")

tab_recs, tab_reviews, tab_personas = st.tabs(
    ["Recommendations", "Review Generator", "Personas"]
)

with tab_recs:
    left, right = st.columns([1, 2])

    with left:
        mode = st.radio(
            "Recommendation mode",
            ["Product similarity", "User memory"],
            horizontal=True,
        )
        top_k = st.slider("Results", min_value=3, max_value=20, value=5)

        if mode == "Product similarity":
            selected_product = st.selectbox("Seed product", items["display_name"].tolist())
            product_id = items.loc[
                items["display_name"] == selected_product, "product_id"
            ].iloc[0]
            recommendations = get_similar_products(product_id, top_k=top_k)
        else:
            selected_user = st.selectbox("User", users["display_name"].tolist())
            user_id = users.loc[users["display_name"] == selected_user, "user_id"].iloc[0]
            recommendations = get_user_recommendations(user_id, top_k=top_k)

    with right:
        st.subheader("Live Recommendations")
        if recommendations:
            st.dataframe(pd.DataFrame(recommendations), width="stretch")
        else:
            st.info("No recommendations found for this selection.")

with tab_reviews:
    left, right = st.columns([1, 2])

    with left:
        selected_review_user = st.selectbox("Persona user", users["display_name"].tolist())
        selected_user_row = users.loc[users["display_name"] == selected_review_user].iloc[0]
        review_user_id = selected_user_row["user_id"]
        review_persona = selected_user_row["persona"]
        selected_product_id = st.selectbox(
            "Product memory",
            ["Manual product"] + items["display_name"].tolist(),
        )

        product_details = {}

        if selected_product_id == "Manual product":
            product_details = {
                "product_name": st.text_input("Product name", "Budget wireless earbuds"),
                "category": st.text_input("Category", "Cell Phones & Accessories"),
                "features": st.text_area(
                    "Product details",
                    "Affordable wireless earbuds with compact charging case.",
                    height=90,
                ),
            }
        else:
            product_row = items.loc[items["display_name"] == selected_product_id].iloc[0]
            product_details = {
                "product_id": product_row.get("product_id"),
                "product_name": product_row.get("product_name"),
                "category": product_row.get("category"),
                "store": product_row.get("store"),
                "features": product_row.get("features"),
                "description": product_row.get("description"),
            }

        generate = st.button("Generate Review", type="primary")

    with right:
        st.subheader("Task A Output")
        if generate:
            result = generate_review_from_persona(
                persona=review_persona,
                product_details=product_details,
            )
            if result:
                rating_cols = st.columns(3)
                rating_cols[0].metric(
                    "Predicted Rating",
                    f"{result['predicted_rating']:.1f} / 5",
                )
                rating_cols[1].metric("Behavioral User", review_user_id[:10])
                rating_cols[2].metric(
                    "Match Score",
                    f"{result['nearest_user_similarity']:.2f}",
                )
                st.text_area("Review", result["review"], height=260)
            else:
                st.error("Could not generate review for this user.")
        else:
            st.caption("Choose a persona and product, then generate a predicted rating and review.")

with tab_personas:
    st.subheader("AI-Readable User Personas")
    persona_columns = [
        "user_id",
        "persona_label",
        "avg_rating",
        "review_count",
        "avg_review_length",
        "persona",
    ]
    persona_columns = [column for column in persona_columns if column in users.columns]
    st.dataframe(users[persona_columns], width="stretch")
