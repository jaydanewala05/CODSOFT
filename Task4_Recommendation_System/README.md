# 🛍️ Task 4 — Product Recommendation System
### CodSoft AI Internship — Jay Danewala

---

## 📌 Overview

A hybrid product recommendation system combining two techniques:
- **Content-Based Filtering** — recommends products similar to what you like
- **Collaborative Filtering** — recommends based on what similar users liked
- **Hybrid** — combines both for the best results

---

## 🗂️ Project Structure

```
Task4_Recommendation_System/
├── recommendation.py    # Main app
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python recommendation.py
```

Menu options:
```
1. Show all products
2. Content-Based recommendations (by product)
3. Collaborative recommendations (by user)
4. Hybrid recommendations (best of both)
5. Top rated products
6. Quit
```

---

## 🧠 How It Works

### Content-Based Filtering
- Uses **TF-IDF** on product tags to find similar products
- Computes **cosine similarity** between product vectors
- Recommends products closest to the selected item

### Collaborative Filtering
- Builds a **user-product ratings matrix**
- Finds users similar to you using **cosine similarity**
- Recommends products liked by similar users

### Hybrid
- Normalizes scores from both methods using **MinMaxScaler**
- Combines and re-ranks for the best final recommendations

---

## 📦 Dependencies

- `pandas` — data handling
- `numpy` — array operations
- `scikit-learn` — TF-IDF, cosine similarity, normalization

---

## 👨‍💻 Author

**Jay Danewala**
- GitHub: [jaydanewala05](https://github.com/jaydanewala05)
- LinkedIn: [jay-danewala](https://linkedin.com/in/jay-danewala)

---

`#codsoft` `#internship` `#artificialintelligence` `#recommendationsystem` `#python` `#machinelearning`