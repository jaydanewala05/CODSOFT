"""
CodSoft AI Internship - Task 4
Product Recommendation System
Techniques: Content-Based + Collaborative Filtering (Hybrid)
Author: Jay Danewala
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SAMPLE DATASET  (no external file needed)
# ─────────────────────────────────────────────

PRODUCTS = pd.DataFrame({
    "product_id":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15],
    "name": [
        "Wireless Bluetooth Headphones",
        "Noise Cancelling Earbuds",
        "Gaming Headset RGB",
        "Smart Watch Fitness Tracker",
        "Smartwatch Pro Max",
        "Laptop Stand Adjustable",
        "Mechanical Keyboard RGB",
        "Wireless Mouse Ergonomic",
        "USB-C Hub 7-in-1",
        "Portable Charger 20000mAh",
        "Phone Case Shockproof",
        "Screen Protector Tempered Glass",
        "Bluetooth Speaker Waterproof",
        "Webcam HD 1080p",
        "Ring Light 10 inch",
    ],
    "category": [
        "Audio", "Audio", "Audio",
        "Wearables", "Wearables",
        "Accessories", "Accessories", "Accessories", "Accessories",
        "Charging", "Phone", "Phone",
        "Audio", "Camera", "Camera",
    ],
    "price": [
        2999, 1999, 3499, 4999, 7999,
        1299, 3999, 999, 1799, 1499,
        499, 299, 2499, 2999, 1299,
    ],
    "rating": [
        4.5, 4.2, 4.3, 4.6, 4.7,
        4.1, 4.4, 4.0, 4.2, 4.5,
        3.9, 3.8, 4.3, 4.1, 4.0,
    ],
    "tags": [
        "wireless bluetooth audio music headphones",
        "noise cancelling earbuds audio music",
        "gaming headset rgb audio",
        "smartwatch fitness health tracker wearable",
        "smartwatch premium health gps wearable",
        "laptop stand desk accessory ergonomic",
        "keyboard mechanical rgb gaming typing",
        "mouse wireless ergonomic office",
        "usb hub adapter connectivity accessories",
        "charger portable power bank charging",
        "phone case protection shockproof cover",
        "screen protector glass phone display",
        "speaker bluetooth audio outdoor waterproof",
        "webcam camera video streaming hd",
        "ring light camera photo video studio",
    ],
})

# User-product ratings matrix (rows=users, cols=product_ids)
RATINGS = pd.DataFrame({
    "user_id":    [1,1,1,1,  2,2,2,2,  3,3,3,3,  4,4,4,4,  5,5,5,5],
    "product_id": [1,3,6,7,  2,4,8,9,  1,2,13,10, 5,4,3,14, 6,7,8,15],
    "rating":     [5,4,3,4,  5,4,4,3,  4,5,4,3,   5,4,3,4,  4,5,4,3],
})


# ─────────────────────────────────────────────
# 1. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

def build_content_matrix():
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(PRODUCTS["tags"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return content_sim


def content_based_recommendations(product_id, top_n=5):
    content_sim = build_content_matrix()
    idx = PRODUCTS[PRODUCTS["product_id"] == product_id].index[0]
    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if PRODUCTS.iloc[s[0]]["product_id"] != product_id]
    top_indices = [s[0] for s in scores[:top_n]]
    result = PRODUCTS.iloc[top_indices][["product_id", "name", "category", "price", "rating"]].copy()
    result["score"] = [round(scores[i][1], 3) for i in range(top_n)]
    result["method"] = "Content-Based"
    return result


# ─────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING  (User-Based)
# ─────────────────────────────────────────────

def build_user_matrix():
    matrix = RATINGS.pivot_table(
        index="user_id", columns="product_id", values="rating"
    ).fillna(0)
    return matrix


def collaborative_recommendations(user_id, top_n=5):
    matrix = build_user_matrix()

    if user_id not in matrix.index:
        print(f"[INFO] User {user_id} not found. Showing popular products instead.")
        return popular_recommendations(top_n)

    user_sim = cosine_similarity(matrix)
    user_sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
    already_rated = set(RATINGS[RATINGS["user_id"] == user_id]["product_id"])

    scores = {}
    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = RATINGS[RATINGS["user_id"] == sim_user]
        for _, row in sim_user_ratings.iterrows():
            pid = row["product_id"]
            if pid not in already_rated:
                scores[pid] = scores.get(pid, 0) + sim_score * row["rating"]

    if not scores:
        return popular_recommendations(top_n)

    top_pids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    result = PRODUCTS[PRODUCTS["product_id"].isin(top_pids)][
        ["product_id", "name", "category", "price", "rating"]
    ].copy()
    result["score"] = result["product_id"].map(lambda p: round(scores.get(p, 0), 3))
    result["method"] = "Collaborative"
    result = result.sort_values("score", ascending=False)
    return result


# ─────────────────────────────────────────────
# 3. HYBRID RECOMMENDATION
# ─────────────────────────────────────────────

def hybrid_recommendations(user_id, product_id, top_n=5):
    print(f"\n{'='*55}")
    print(f"  Hybrid Recommendations for User {user_id}")
    print(f"  Based on product: '{PRODUCTS[PRODUCTS['product_id']==product_id]['name'].values[0]}'")
    print(f"{'='*55}")

    cb  = content_based_recommendations(product_id, top_n=10)
    cf  = collaborative_recommendations(user_id, top_n=10)

    cb["norm_score"] = MinMaxScaler().fit_transform(cb[["score"]])
    cf["norm_score"] = MinMaxScaler().fit_transform(cf[["score"]])

    combined = pd.concat([cb, cf])
    combined = combined.groupby("product_id").agg({
        "name": "first",
        "category": "first",
        "price": "first",
        "rating": "first",
        "norm_score": "sum",
    }).reset_index()
    combined = combined.sort_values("norm_score", ascending=False).head(top_n)
    combined["method"] = "Hybrid"

    print(f"\n{'#':<4} {'Product':<35} {'Category':<14} {'Price':>7}  {'Rating':>6}  {'Score':>6}")
    print("-" * 78)
    for i, (_, row) in enumerate(combined.iterrows(), 1):
        print(f"{i:<4} {row['name']:<35} {row['category']:<14} ₹{row['price']:>6}  "
              f"{row['rating']:>6}  {row['norm_score']:>6.3f}")
    print()
    return combined


# ─────────────────────────────────────────────
# 4. POPULAR PRODUCTS  (fallback)
# ─────────────────────────────────────────────

def popular_recommendations(top_n=5):
    result = PRODUCTS.nlargest(top_n, "rating")[
        ["product_id", "name", "category", "price", "rating"]
    ].copy()
    result["score"] = result["rating"] / 5.0
    result["method"] = "Popular"
    return result


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def show_all_products():
    print(f"\n{'='*60}")
    print("  Available Products")
    print(f"{'='*60}")
    print(f"{'ID':<5} {'Product':<35} {'Category':<14} {'Price':>7}  {'Rating'}")
    print("-" * 70)
    for _, row in PRODUCTS.iterrows():
        print(f"{row['product_id']:<5} {row['name']:<35} {row['category']:<14} "
              f"₹{row['price']:>6}  {row['rating']}")
    print()


def show_users():
    print(f"\n{'='*30}")
    print("  Available User IDs: 1 to 5")
    print(f"{'='*30}\n")


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  CodSoft AI — Product Recommendation System")
    print("  Techniques: Content-Based + Collaborative (Hybrid)")
    print("="*55)

    while True:
        print("\nOptions:")
        print("  1. Show all products")
        print("  2. Content-Based recommendations (by product)")
        print("  3. Collaborative recommendations (by user)")
        print("  4. Hybrid recommendations (best of both)")
        print("  5. Top rated products")
        print("  6. Quit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            show_all_products()

        elif choice == "2":
            show_all_products()
            try:
                pid = int(input("Enter Product ID: "))
                recs = content_based_recommendations(pid)
                print(f"\n--- Content-Based Recommendations ---")
                print(f"{'#':<4} {'Product':<35} {'Category':<14} {'Price':>7}  {'Score'}")
                print("-" * 68)
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    print(f"{i:<4} {row['name']:<35} {row['category']:<14} "
                          f"₹{row['price']:>6}  {row['score']}")
            except Exception as e:
                print(f"[ERR] {e}")

        elif choice == "3":
            show_users()
            try:
                uid = int(input("Enter User ID (1-5): "))
                recs = collaborative_recommendations(uid)
                print(f"\n--- Collaborative Recommendations for User {uid} ---")
                print(f"{'#':<4} {'Product':<35} {'Category':<14} {'Price':>7}  {'Score'}")
                print("-" * 68)
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    print(f"{i:<4} {row['name']:<35} {row['category']:<14} "
                          f"₹{row['price']:>6}  {row['score']}")
            except Exception as e:
                print(f"[ERR] {e}")

        elif choice == "4":
            show_all_products()
            show_users()
            try:
                uid = int(input("Enter User ID (1-5): "))
                pid = int(input("Enter a Product ID you like: "))
                hybrid_recommendations(uid, pid)
            except Exception as e:
                print(f"[ERR] {e}")

        elif choice == "5":
            recs = popular_recommendations()
            print(f"\n--- Top Rated Products ---")
            print(f"{'#':<4} {'Product':<35} {'Category':<14} {'Price':>7}  {'Rating'}")
            print("-" * 68)
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                print(f"{i:<4} {row['name']:<35} {row['category']:<14} "
                      f"₹{row['price']:>6}  {row['rating']}")

        elif choice == "6":
            print("Bye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()