import mysql.connector
import pandas as pd
import os

print("🔵 Integrating Social Media & Browsing Data...")

# ✅ Connect to MySQL
conn = mysql.connector.connect(host="localhost", user="root", password="", database="ecommerce_recommendation")
cursor = conn.cursor()

# ✅ Fetch browsing history
cursor.execute("SELECT user_id, product_id, action FROM browsing_history")
browsing_df = pd.DataFrame(cursor.fetchall(), columns=["user_id", "product_id", "action"])

# ✅ Fetch social media interactions
cursor.execute("SELECT user_id, product_id, engagement_score FROM social_media_data")
social_df = pd.DataFrame(cursor.fetchall(), columns=["user_id", "product_id", "engagement_score"])

# ✅ Fetch transactions
cursor.execute("SELECT user_id, product_id FROM transactions")
transactions_df = pd.DataFrame(cursor.fetchall(), columns=["user_id", "product_id"])

# ✅ Merge all datasets safely
combined_df = browsing_df.merge(social_df, on=["user_id", "product_id"], how="outer").merge(
    transactions_df, on=["user_id", "product_id"], how="outer"
).fillna(0)

# ✅ Normalize engagement score
if "engagement_score" in combined_df.columns:
    min_score, max_score = combined_df["engagement_score"].min(), combined_df["engagement_score"].max()
    if min_score != max_score:
        combined_df["engagement_score"] = (combined_df["engagement_score"] - min_score) / (max_score - min_score)

# ✅ Remove duplicate user-product pairs
combined_df = combined_df.drop_duplicates(subset=["user_id", "product_id"])

# ✅ Ensure the output directory exists
output_dir = "backend/models"
os.makedirs(output_dir, exist_ok=True)

# ✅ Save the processed data
combined_df.to_csv(os.path.join(output_dir, "social_browsing_data.csv"), index=False)
print("✅ Social & Browsing Data Integration Completed!")

cursor.close()
conn.close()
