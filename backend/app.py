from flask import Flask, request, jsonify
import mysql.connector
import pickle
import os
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import dill
import sys
import math

# Add script path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from scripts.federated import FederatedRecommender
from scripts.shap_wrapper import SurpriseWrapper, ContentWrapper, GraphWrapper, FederatedWrapper

app = Flask(__name__)

def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="ecommerce_recommendation"
    )

def load_models():
    models = {}
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    try:
        with open(os.path.join(model_dir, "collaborative_model.pkl"), "rb") as f:
            models["collaborative"] = SurpriseWrapper(pickle.load(f))
    except Exception as e:
        print(f"❌ Collaborative Model Not Found: {e}")

    try:
        with open(os.path.join(model_dir, "content_model.pkl"), "rb") as f:
            models["content_raw"] = pickle.load(f)
    except Exception as e:
        print(f"❌ Content Model Not Found: {e}")

    try:
        from gensim.models import KeyedVectors
        models["graph_raw"] = KeyedVectors.load(os.path.join(model_dir, "graph_model.kv"))
    except Exception as e:
        print(f"❌ Graph Model Not Found: {e}")

    try:
        models["federated_raw"] = tf.keras.models.load_model(os.path.join(model_dir, "session_model.h5"))
    except Exception as e:
        print(f"❌ Federated Model Not Found: {e}")

    for model_type in ["collaborative", "content", "graph", "federated"]:
        path = os.path.join(model_dir, f"shap_explainer_{model_type}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[f"shap_{model_type}"] = dill.load(f)
        else:
            print(f"❌ SHAP explainer for {model_type} not found.")

    return models

models = load_models()

def get_products():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, name, image_url FROM products")
    products = {
        int(pid): {
            "name": name,
            "image": image,
            "engagement_score": 0,
            "browsing_action": 0
        }
        for pid, name, image in cursor.fetchall()
    }
    cursor.close()
    conn.close()

    csv_path = os.path.join(os.path.dirname(__file__), "models", "social_browsing_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        agg = df.groupby("product_id").agg(
            engagement_score=("engagement_score", "mean"),
            browsing_action=("action", "sum")
        ).reset_index()

        for _, row in agg.iterrows():
            pid = int(row["product_id"])
            if pid in products:
                products[pid]["engagement_score"] = row["engagement_score"]
                products[pid]["browsing_action"] = row["browsing_action"]

    return products

def get_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, rating, comment FROM reviews")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    review_dict = {}
    for pid, rating, comment in rows:
        if pid not in review_dict:
            review_dict[pid] = []
        review_dict[pid].append({
            "rating": rating,
            "comment": comment
        })
    return review_dict

@app.route("/api/recommendations", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = int(data.get("user_id"))
    products = get_products()
    reviews = get_reviews()
    recommendations = []
    seen = set()
    shap_explanations = {}
    shap_breakdowns = {}

    # ✅ Collaborative
    if "collaborative" in models:
        try:
            predictions = [(pid, models["collaborative"].predict(np.array([[user_id, pid]]))[0])
                           for pid in products]
            top = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
            for pid, score in top:
                if pid not in seen:
                    recommendations.append((pid, score, "collaborative"))
                    seen.add(pid)
                    if "shap_collaborative" in models:
                        shap_vals = models["shap_collaborative"].shap_values(np.array([[user_id, pid]]))
                        shap_flat = np.array(shap_vals).flatten()
                        shap_explanations[pid] = float(shap_flat[0]) if len(shap_flat) else 0
                        shap_breakdowns[pid] = {
                            f"feature_{i}": float(v) for i, v in enumerate(shap_flat)
                        }
        except Exception as e:
            print(f"❌ Collaborative Error: {e}")

    # ✅ Content-Based
    if "content_raw" in models:
        try:
            sim = models["content_raw"]
            if isinstance(sim, dict):
                user_scores = sim.get(user_id, {})
                top = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                user_idx = user_id % sim.shape[0]
                top = sorted([(i, sim[user_idx][i]) for i in range(sim.shape[1]) if i in products],
                             key=lambda x: x[1], reverse=True)[:5]

            for pid, score in top:
                if pid not in seen:
                    recommendations.append((pid, score, "content"))
                    seen.add(pid)
                    if "shap_content" in models:
                        shap_vals = models["shap_content"].shap_values(np.array([pid]))
                        shap_flat = np.array(shap_vals).flatten()
                        shap_explanations[pid] = float(shap_flat[0]) if len(shap_flat) else 0
                        shap_breakdowns[pid] = {
                            f"feature_{i}": float(v) for i, v in enumerate(shap_flat)
                        }
        except Exception as e:
            print(f"❌ Content Error: {e}")

    # ✅ Graph-Based
    if "graph_raw" in models:
        try:
            user_node = f"user_{user_id}"
            graph_model = GraphWrapper(models["graph_raw"], user_node)
            product_ids = list(products.keys())
            scores = graph_model.predict(product_ids)
            top_indices = np.argsort(scores)[::-1][:5]
            for i in top_indices:
                pid = product_ids[i]
                if pid not in seen:
                    recommendations.append((pid, scores[i], "graph"))
                    seen.add(pid)
                    if "shap_graph" in models:
                        shap_vals = models["shap_graph"].shap_values(np.array([pid]))
                        shap_flat = np.array(shap_vals).flatten()
                        shap_explanations[pid] = float(shap_flat[0]) if len(shap_flat) else 0
                        shap_breakdowns[pid] = {
                            f"feature_{i}": float(v) for i, v in enumerate(shap_flat)
                        }
        except Exception as e:
            print(f"❌ Graph Error: {e}")

    # ✅ Federated
    if "federated" not in models:
        try:
            models["federated"] = FederatedRecommender(os.path.join(os.path.dirname(__file__), "models", "session_model.h5"))
        except Exception as e:
            print(f"❌ Failed to load Federated Model: {e}")

    if "federated" in models:
        try:
            fed_recs = models["federated"].predict(user_id)
            for pid in fed_recs:
                if pid not in seen:
                    recommendations.append((pid, 0.9, "federated"))
                    seen.add(pid)
                    if "shap_federated" in models:
                        shap_vals = models["shap_federated"].shap_values(np.array([pid]))
                        shap_flat = np.array(shap_vals).flatten()
                        shap_explanations[pid] = float(shap_flat[0]) if len(shap_flat) else 0
                        shap_breakdowns[pid] = {
                            f"feature_{i}": float(v) for i, v in enumerate(shap_flat)
                        }
        except Exception as e:
            print(f"❌ Federated Error: {e}")

    final = [
        {
            "product_id": pid,
            "name": products[pid]["name"],
            "image": products[pid]["image"],
            "engagement_score": float(products[pid]["engagement_score"]) if not math.isnan(products[pid]["engagement_score"]) else 0.0,
            "browsing_action": str(products[pid]["browsing_action"]) if products[pid]["browsing_action"] else "unknown",
            "shap_value": shap_explanations.get(pid, 0),
            "shap_breakdown": shap_breakdowns.get(pid, {}),
            "reviews": reviews.get(pid, [])
        }
        for pid, _, _ in recommendations[:5]
        if pid in products
    ]

    return jsonify({"recommendations": final})

@app.route("/")
def index():
    return "✅ E-commerce Recommendation API is running!"

if __name__ == "__main__":
    app.run(debug=True)
