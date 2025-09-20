import mysql.connector
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import os

try:
    print("🔵 Connecting to MySQL...")
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecommerce_recommendation"
    )
    cursor = conn.cursor()
    print("✅ Connected to MySQL!")

    # 🟡 Fetch transactions
    query = "SELECT user_id, product_id FROM transactions"
    cursor.execute(query)
    results = cursor.fetchall()

    if not results:
        print("❌ ERROR: No data found in 'transactions' table!")
        exit()

    print(f"✅ Loaded {len(results)} records from 'transactions' table.")
    print("📊 Sample Rows:", results[:5])  # Show sample

    # 🟡 Convert to DataFrame
    df = pd.DataFrame(results, columns=["user_id", "product_id"])

    # 🟡 Create bipartite-like graph
    print("🟡 Creating User-Product Interaction Graph...")
    G = nx.Graph()
    G.add_edges_from([
        ("user_" + str(row["user_id"]), "product_" + str(row["product_id"]))
        for _, row in df.iterrows()
    ])

    if G.number_of_nodes() == 0:
        raise Exception("Graph is empty. Cannot proceed with Node2Vec training.")

    print(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 🟡 Train Node2Vec model
    print("🟡 Training Node2Vec Embeddings...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=2, quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print("✅ Node2Vec Training Complete!")

    # 🟢 Save embeddings
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "graph_model.kv")
    model.wv.save(save_path)
    print(f"✅ Model saved at {save_path}")

except mysql.connector.Error as err:
    print(f"❌ MySQL Error: {err}")
except Exception as e:
    print(f"❌ Python Error: {e}")
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals(): conn.close()
    print("🔴 MySQL Connection Closed.")
