from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.service import (
    generate_review_from_persona,
    generate_personalized_review,
    get_product,
    get_similar_products,
    get_user,
    get_user_recommendations,
    list_products,
    list_users,
    recommend_from_persona,
)


app = FastAPI(
    title="Kaycorp Intelligent Review Agent",
    description="Customer profiles, product recommendations, and review generation.",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    user_id: str = Field(..., examples=["lagos_student_001"])
    product_id: Optional[str] = Field(default=None, examples=["B013SK1JTY"])
    product_name: Optional[str] = Field(default=None, examples=["Wireless phone case"])
    avg_rating: Optional[float] = Field(default=None, ge=0, le=5)


class TaskARequest(BaseModel):
    persona: str = Field(
        ...,
        examples=[
            "A budget-conscious Nigerian student who writes short reviews and cares about price, delivery speed, and durability."
        ],
    )
    product_details: dict[str, Any] = Field(
        ...,
        examples=[
            {
                "product_id": "B08L6L3X1S",
                "product_name": "Clear phone case for blue smartphone",
                "category": "Cell Phones & Accessories",
                "features": "Transparent protective case, slim fit",
            }
        ],
    )


class TaskBRequest(BaseModel):
    persona: str = Field(
        ...,
        examples=[
            "A Lagos-based phone accessories buyer who prefers affordable, durable products with fast delivery."
        ],
    )
    context: str = Field(default="", examples=["Needs a phone accessory below a student budget."])
    product_history: list[str] = Field(default_factory=list, examples=[["B08L6L3X1S"]])
    category: Optional[str] = Field(default=None, examples=["Cell Phones"])
    top_k: int = Field(default=10, ge=1, le=50)


@app.get("/")
def root():
    return {
        "name": "Kaycorp Intelligent Review Agent",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products")
def products(limit: int = Query(default=25, ge=1, le=200)):
    return {"products": list_products(limit)}


@app.get("/products/{product_id}")
def product_detail(product_id: str):
    product = get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    return product


@app.get("/users")
def users(limit: int = Query(default=25, ge=1, le=200)):
    return {"users": list_users(limit)}


@app.get("/users/{user_id}")
def user_detail(user_id: str):
    user = get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.get("/recommendations/product/{product_id}")
def product_recommendations(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=50),
):
    recommendations = get_similar_products(product_id, top_k=top_k)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Product not found")

    return {
        "seed_product_id": product_id,
        "recommendations": recommendations,
    }


@app.get("/recommendations/user/{user_id}")
def user_recommendations(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=50),
):
    recommendations = get_user_recommendations(user_id, top_k=top_k)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="User has no recommendation memory")

    return {
        "user_id": user_id,
        "recommendations": recommendations,
    }


@app.post("/reviews/generate")
def create_review(request: ReviewRequest):
    result = generate_personalized_review(
        user_id=request.user_id,
        product_id=request.product_id,
        product_name=request.product_name,
        avg_rating=request.avg_rating,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="User not found")

    return result


@app.post("/task-a/generate-review")
def task_a_generate_review(request: TaskARequest):
    return generate_review_from_persona(
        persona=request.persona,
        product_details=request.product_details,
    )


@app.post("/task-b/recommend")
def task_b_recommend(request: TaskBRequest):
    return recommend_from_persona(
        persona=request.persona,
        context=request.context,
        top_k=request.top_k,
        product_history=request.product_history,
        category=request.category,
    )


@app.post("/task-b/conversation-turn")
def task_b_conversation_turn(request: TaskBRequest):
    result = recommend_from_persona(
        persona=request.persona,
        context=request.context,
        top_k=request.top_k,
        product_history=request.product_history,
        category=request.category,
    )
    result["agent_message"] = (
        "I used your latest context to refine the retrieval query, then reranked products "
        "against the item memory. Add another preference in the context field to continue."
    )
    return result
