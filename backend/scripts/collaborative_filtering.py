import mysql.connector
import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
import os

try:
    print("🔵 Connecting to MySQL...")
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="ecommerce_recommendation")
    cursor = conn.cursor()
    print("✅ Connected!")

    # 🟡 Fetch user ratings data
    query = "SELECT user_id, product_id, rating FROM reviews"
    cursor.execute(query)
    results = cursor.fetchall()

    if not results:
        print("❌ ERROR: No data found in 'reviews' table!")
        exit()

    print("✅ Sample Data Loaded:", results[:5])  # Print first 5 rows

    # 🟡 Convert data to DataFrame
    df = pd.DataFrame(results, columns=["user_id", "product_id", "rating"])

    # 🟡 Train SVD model
    print("🟡 Training Collaborative Filtering Model...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    # ✅ Ensure models directory exists and save the model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "collaborative_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Collaborative Filtering Model trained and saved at:", model_path)

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")
except Exception as e:
    print(f"❌ Python Error: {e}")
finally:
    cursor.close()
    conn.close()
    print("🔴 MySQL Connection Closed.")
