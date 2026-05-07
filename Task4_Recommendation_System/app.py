"""
CodSoft AI Internship - Task 4
Product Recommendation System — Web UI
Run: python app.py  then open http://localhost:5000
Author: Jay Danewala
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
PRODUCTS = pd.DataFrame({
    "product_id":   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    "name": [
        "Wireless Bluetooth Headphones","Noise Cancelling Earbuds","Gaming Headset RGB",
        "Smart Watch Fitness Tracker","Smartwatch Pro Max","Laptop Stand Adjustable",
        "Mechanical Keyboard RGB","Wireless Mouse Ergonomic","USB-C Hub 7-in-1",
        "Portable Charger 20000mAh","Phone Case Shockproof","Screen Protector Tempered Glass",
        "Bluetooth Speaker Waterproof","Webcam HD 1080p","Ring Light 10 inch",
    ],
    "category": [
        "Audio","Audio","Audio","Wearables","Wearables",
        "Accessories","Accessories","Accessories","Accessories",
        "Charging","Phone","Phone","Audio","Camera","Camera",
    ],
    "price": [2999,1999,3499,4999,7999,1299,3999,999,1799,1499,499,299,2499,2999,1299],
    "rating": [4.5,4.2,4.3,4.6,4.7,4.1,4.4,4.0,4.2,4.5,3.9,3.8,4.3,4.1,4.0],
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
    # Real Unsplash product images
    "image": [
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&q=80",
        "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=400&q=80",
        "https://images.unsplash.com/photo-1599669454699-248893623440?w=400&q=80",
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&q=80",
        "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400&q=80",
        "https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=400&q=80",
        "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=400&q=80",
        "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400&q=80",
        "https://images.unsplash.com/photo-1625895197185-efcec01cffe0?w=400&q=80",
        "https://images.unsplash.com/photo-1609091839311-d5365f9ff1c5?w=400&q=80",
        "https://images.unsplash.com/photo-1601593346740-925612772716?w=400&q=80",
        "https://images.unsplash.com/photo-1585771724684-38269d6639fd?w=400&q=80",
        "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&q=80",
        "https://images.unsplash.com/photo-1587826080692-f439cd0b70da?w=400&q=80",
        "https://images.unsplash.com/photo-1620674156044-52b714665d46?w=400&q=80",
    ],
})

RATINGS = pd.DataFrame({
    "user_id":    [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5],
    "product_id": [1,3,6,7, 2,4,8,9, 1,2,13,10, 5,4,3,14, 6,7,8,15],
    "rating":     [5,4,3,4, 5,4,4,3, 4,5,4,3,   5,4,3,4,  4,5,4,3],
})

# ─────────────────────────────────────────────
# RECOMMENDATION LOGIC
# ─────────────────────────────────────────────
def build_content_matrix():
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(PRODUCTS["tags"])
    return cosine_similarity(matrix, matrix)

def content_based(product_id, top_n=6):
    sim = build_content_matrix()
    idx = PRODUCTS[PRODUCTS["product_id"] == product_id].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if PRODUCTS.iloc[s[0]]["product_id"] != product_id]
    top = [s[0] for s in scores[:top_n]]
    result = PRODUCTS.iloc[top].copy()
    result["score"] = [round(scores[i][1], 3) for i in range(top_n)]
    result["method"] = "Content-Based"
    return result

def collaborative(user_id, top_n=6):
    matrix = RATINGS.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    if user_id not in matrix.index:
        return popular(top_n)
    sim = pd.DataFrame(cosine_similarity(matrix), index=matrix.index, columns=matrix.index)
    similar = sim[user_id].sort_values(ascending=False).drop(user_id)
    already = set(RATINGS[RATINGS["user_id"] == user_id]["product_id"])
    scores = {}
    for su, ss in similar.items():
        for _, row in RATINGS[RATINGS["user_id"] == su].iterrows():
            pid = row["product_id"]
            if pid not in already:
                scores[pid] = scores.get(pid, 0) + ss * row["rating"]
    if not scores:
        return popular(top_n)
    top_pids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    result = PRODUCTS[PRODUCTS["product_id"].isin(top_pids)].copy()
    result["score"] = result["product_id"].map(lambda p: round(scores.get(p, 0), 3))
    result["method"] = "Collaborative"
    return result.sort_values("score", ascending=False)

def hybrid(user_id, product_id, top_n=6):
    cb = content_based(product_id, top_n=10)
    cf = collaborative(user_id, top_n=10)
    cb["norm"] = MinMaxScaler().fit_transform(cb[["score"]])
    cf["norm"] = MinMaxScaler().fit_transform(cf[["score"]])
    combined = pd.concat([cb, cf])
    combined = combined.groupby("product_id").agg({
        "name":"first","category":"first","price":"first",
        "rating":"first","image":"first","norm":"sum"
    }).reset_index()
    combined = combined.sort_values("norm", ascending=False).head(top_n)
    combined["score"] = combined["norm"].round(3)
    combined["method"] = "Hybrid"
    return combined

def popular(top_n=6):
    result = PRODUCTS.nlargest(top_n, "rating").copy()
    result["score"] = result["rating"] / 5.0
    result["method"] = "Popular"
    return result

def df_to_list(df):
    return df[["product_id","name","category","price","rating","image","score","method"]].to_dict("records")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    products = PRODUCTS[["product_id","name","category","price","rating","image"]].to_dict("records")
    return render_template("index.html", products=products)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    mode = data.get("mode")
    try:
        if mode == "content":
            result = content_based(int(data["product_id"]))
        elif mode == "collaborative":
            result = collaborative(int(data["user_id"]))
        elif mode == "hybrid":
            result = hybrid(int(data["user_id"]), int(data["product_id"]))
        elif mode == "popular":
            result = popular()
        else:
            return jsonify({"error": "Invalid mode"}), 400
        return jsonify(df_to_list(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n✅ Server running → open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)