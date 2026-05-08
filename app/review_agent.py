import hashlib


def generate_review(persona, product_name, avg_rating):
    if avg_rating >= 4:
        sentiment = "positive"
    elif avg_rating >= 3:
        sentiment = "neutral"
    else:
        sentiment = "negative"

    nigerian_phrases = [
        "for the price ehn",
        "actually not bad",
        "delivery was sharp",
        "this thing surprised me",
        "I cannot lie",
    ]

    # Use one light phrase so the tone feels natural, not exaggerated.
    seed = hashlib.sha256(f"{persona}|{product_name}|{avg_rating}".encode("utf-8")).hexdigest()
    local_touch = nigerian_phrases[int(seed[:8], 16) % len(nigerian_phrases)]

    return f"""
User Persona:
{persona}

Generated Review:
I recently tried {product_name} and my experience was mostly {sentiment}.
The quality was decent and it generally met expectations; {local_touch}.
I would consider recommending it depending on the user's preferences.
""".strip()
