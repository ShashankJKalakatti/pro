import os
import mysql.connector
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

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

    # 🟡 Fetch transactions sorted by time
    query = "SELECT user_id, product_id, purchase_date FROM transactions ORDER BY purchase_date"
    cursor.execute(query)
    results = cursor.fetchall()

    if not results:
        print("❌ ERROR: No data in 'transactions' table!")
        exit()

    print(f"✅ Loaded {len(results)} transactions.")
    df = pd.DataFrame(results, columns=["user_id", "product_id", "purchase_date"])

    # 🟡 Encode product IDs for Embedding layer
    df['product_id'] = df['product_id'].astype("category")
    df['user_id'] = df['user_id'].astype("category")  # Optional if needed later

    # 🔁 Encode product ID into integer index
    df['product_idx'] = df['product_id'].cat.codes
    product_to_index = {pid: idx for idx, pid in enumerate(df['product_id'].cat.categories)}
    index_to_product = {idx: pid for pid, idx in product_to_index.items()}

    # 🟡 Prepare sequences
    X, y = [], []
    for i in range(len(df) - 3):
        seq = df.iloc[i:i+3]['product_idx'].values
        target = df.iloc[i+3]['product_idx']
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    vocab_size = df['product_idx'].nunique()
    print("🧠 Vocabulary Size:", vocab_size)

    # 🟡 Build the LSTM Model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=3),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("🟡 Training Session-Based LSTM...")
    model.fit(X, y, epochs=10, batch_size=32)

    # 🟢 Save model and mapping
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "session_model.h5")
    mapping_path = os.path.join(model_dir, "product_mapping.pkl")

    model.save(model_path)
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "product_to_index": product_to_index,
            "index_to_product": index_to_product
        }, f)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Product mapping saved to {mapping_path}")

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")
except Exception as e:
    print(f"❌ Python Error: {e}")
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()
    print("🔴 MySQL Connection Closed.")
