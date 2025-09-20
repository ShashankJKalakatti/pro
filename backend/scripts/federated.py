import os
import flwr as fl
import tensorflow as tf
import numpy as np
import mysql.connector
import pickle

# ✅ Load product_id <-> index mapping from file
def load_product_mapping():
    mapping_path = os.path.join(os.path.dirname(__file__), "../models/product_mapping.pkl")
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return mapping["product_to_index"], mapping["index_to_product"]

# ✅ Fetch user-product-rating data and convert to index
def get_user_data(product_to_index):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="", database="ecommerce_recommendation"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, product_id, rating FROM interactions")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    if not data:
        print("⚠️ No user-product interaction data found.")
        return np.array([]), np.array([]), np.array([])

    users, products, ratings = zip(*data)
    products = [product_to_index.get(pid, 0) for pid in products]
    return np.array(users), np.array(products), np.array(ratings)

# ✅ Fetch user's last 2 product interactions
def get_recent_session(user_id, product_to_index):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="", database="ecommerce_recommendation"
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT product_id FROM transactions
        WHERE user_id = %s
        ORDER BY purchase_date DESC
        LIMIT 2
    """, (user_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(results) < 2:
        return None

    return [product_to_index.get(pid[0], 0) for pid in reversed(results)]  # Oldest → Newest

class FederatedRecommender(fl.client.NumPyClient):
    def __init__(self, model_path):
        print("🔵 Loading model from:", model_path)
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded.")
        self.product_to_index, self.index_to_product = load_product_mapping()

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print("🟢 Federated Learning Training Started...")

        users, products, ratings = get_user_data(self.product_to_index)
        if len(users) == 0:
            print("⚠️ Skipping training: no data.")
            return self.get_parameters(), 0, {}

        # Dummy padded input for training
        X_train = np.array([[p, p, p] for p in products])
        y_train = ratings

        self.model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
        print("✅ Federated Training Complete.")
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        print("🧪 Evaluating Federated Model...")

        _, products, ratings = get_user_data(self.product_to_index)
        if len(products) == 0:
            print("⚠️ Skipping evaluation.")
            return 0.0, 0, {}

        X_test = np.array([[p, p, p] for p in products])
        y_test = ratings
        loss, _ = self.model.evaluate(X_test, y_test, verbose=0)
        print("✅ Evaluation Done. Loss:", loss)
        return float(loss), len(X_test), {}

    def predict(self, user_id, top_k=5):
        print(f"🔮 Predicting top {top_k} products for user {user_id}...")
        recent_session = get_recent_session(user_id, self.product_to_index)
        product_ids = list(self.product_to_index.keys())

        if not recent_session or len(recent_session) < 2:
            print("⚠️ Not enough session data, using dummy input.")
            return product_ids[:top_k]

        input_sequences = []
        valid_products = []

        for pid in product_ids:
            if pid in self.product_to_index:
                candidate_idx = self.product_to_index[pid]
                input_seq = recent_session[-2:] + [candidate_idx]
                if len(input_seq) == 3:
                    input_sequences.append(input_seq)
                    valid_products.append(pid)

        if not input_sequences:
            print("⚠️ No valid inputs generated, returning top random.")
            return product_ids[:top_k]

        X_input = np.array(input_sequences)
        predictions = self.model.predict(X_input, verbose=0).flatten()
        top_indices = np.argsort(predictions)[::-1][:top_k]
        return [valid_products[i] for i in top_indices if i < len(valid_products)]


# ✅ Entry Point
if __name__ == "__main__":
    print("🚀 Starting Federated Client with Real Data...")
    model_path = os.path.join(os.path.dirname(__file__), "../models/session_model.h5")
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedRecommender(model_path))
