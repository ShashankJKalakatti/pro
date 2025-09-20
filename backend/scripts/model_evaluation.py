import mysql.connector
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

print("🔵 Connecting to MySQL...")

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecommerce_recommendation"
    )
    cursor = conn.cursor()
    print("✅ MySQL Connection Successful!")

    # 🟡 Step 1: Fetch Data
    print("🔵 Fetching review data...")
    cursor.execute("SELECT user_id, product_id, rating FROM reviews")
    data = cursor.fetchall()
    print("✅ Sample Data:", data[:5])

    # 🟡 Step 2: Convert to Pandas DataFrame
    print("🔵 Converting data to DataFrame...")
    df = pd.DataFrame(data, columns=["user_id", "product_id", "rating"])
    print(f"✅ DataFrame Created with {df.shape[0]} rows")
    print(df.head())

    # 🟡 Step 3: Convert Data Types
    print("🔵 Checking Data Types...")
    print(df.dtypes)

    df["user_id"] = df["user_id"].astype(int)
    df["product_id"] = df["product_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    print("✅ Data Types After Conversion:", df.dtypes)

    # 🟡 Step 4: Prepare Data for Surprise
    print("🔵 Preparing dataset for Surprise library...")
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader)

    # 🟡 Step 5: Load Model
    print("🔵 Loading Collaborative Filtering Model...")
    try:
        with open("backend/models/collaborative_model.pkl", "rb") as f:
            collab_model = pickle.load(f)
        print("✅ Model Loaded Successfully!")
    except FileNotFoundError:
        print("❌ ERROR: Model file not found! Ensure 'models/collaborative_model.pkl' exists.")
        exit()

    # 🟡 Step 6: Run Cross-Validation
    print("🟡 Running Cross-Validation on Model...")
    results = cross_validate(collab_model, dataset, cv=5)
    print("✅ Model Evaluation Completed!", results)

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")

finally:
    cursor.close()
    conn.close()
    print("🔴 MySQL Connection Closed.")
