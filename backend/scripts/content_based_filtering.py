import os
import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

try:
    print("🔵 Connecting to MySQL...")
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecommerce_recommendation"
    )
    cursor = conn.cursor()
    print("✅ Connected!")

    # 🟡 Fetch product descriptions
    query = "SELECT product_id, name, description FROM products"
    cursor.execute(query)
    results = cursor.fetchall()

    if not results:
        raise Exception("No data found in 'products' table!")

    print(f"✅ {len(results)} products loaded. Sample:", results[:2])

    # 🟡 Load into DataFrame
    df = pd.DataFrame(results, columns=["product_id", "name", "description"])

    # ⚠️ Clean null/empty descriptions
    df['description'] = df['description'].fillna("").astype(str)
    df = df[df['description'].str.strip() != ""]

    if df.empty:
        raise Exception("All product descriptions are empty after cleaning.")

    # 🟡 TF-IDF Vectorization
    print("🟡 Generating TF-IDF Vectors...")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['description'])

    print("✅ TF-IDF Matrix Shape:", tfidf_matrix.shape)

    # 🟢 Compute similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # ✅ Save model & data
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "content_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump({
            "cosine_sim": cosine_sim,
            "products": df,
            "tfidf": tfidf
        }, f)

    print(f"✅ Content-Based Model trained and saved at {model_path}!")

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")
except Exception as e:
    print(f"❌ Python Error: {e}")
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()
    print("🔴 MySQL Connection Closed.")
