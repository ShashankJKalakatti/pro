import sys
import os
import shap
import dill  # Better than pickle for saving complex objects
import pickle
import numpy as np
from surprise import SVD

# ✅ Add current script dir to sys.path to import wrapper
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from shap_wrapper import SurpriseWrapper

print("🔵 Loading SHAP Explainability...")

try:
    # ✅ Step 1: Load Collaborative Filtering Model
    model_path = os.path.join(os.path.dirname(__file__), "../models/collaborative_model.pkl")
    print(f"🔵 Loading Model from: {model_path}")

    with open(model_path, "rb") as f:
        collab_model = pickle.load(f)
    print("✅ Model Loaded Successfully!")

    # ✅ Step 2: Generate Sample User-Item Interactions
    print("🔵 Generating Sample Data...")
    sample_data = np.array([[u, u] for u in range(1, 6)])  # Format: [user_id, product_id]
    print("✅ Sample Data Created:", sample_data)

    # ✅ Step 3: Wrap Model for SHAP
    print("🔵 Wrapping Model for SHAP...")
    wrapped_model = SurpriseWrapper(collab_model)
    print("✅ SHAP Wrapper Created!")

    # ✅ Step 4: Create SHAP KernelExplainer
    print("🔵 Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(wrapped_model.predict, sample_data)
    print("✅ SHAP KernelExplainer Created!")

    # ✅ Save the SHAP Explainer using `dill`
    shap_explainer_path = os.path.join(os.path.dirname(__file__), "../models/shap_explainer.pkl")
    with open(shap_explainer_path, "wb") as f:
        dill.dump(explainer, f)
    print(f"✅ SHAP Explainer Saved at {shap_explainer_path}")

    # ✅ SHAPE GUARD — Ensure correct input shape
    print("🔵 Validating Input Shape for SHAP...")
    if len(sample_data.shape) != 2 or sample_data.shape[1] != 2:
        raise ValueError(f"Invalid input shape for SHAP explainer: {sample_data.shape}. Expected shape (n_samples, 2).")
    print("✅ Input shape validated:", sample_data.shape)

    # ✅ Step 5: Generate SHAP Values
    print("🔵 Generating SHAP Values...")
    shap_values = explainer.shap_values(sample_data)
    print("✅ SHAP Values Generated!")

    # ✅ Save SHAP Values
    shap_values_path = os.path.join(os.path.dirname(__file__), "../models/shap_values.pkl")
    with open(shap_values_path, "wb") as f:
        pickle.dump(shap_values, f)
    print(f"✅ SHAP Values Saved at {shap_values_path}")

    # ✅ Optional: Show SHAP plot (skip if running on a server)
    if "DISPLAY" in os.environ or os.name == "nt":
        try:
            shap.summary_plot(shap_values, sample_data)
        except Exception:
            print("⚠️ Could not display SHAP plot (GUI issue). Skipping.")

except FileNotFoundError:
    print("❌ ERROR: collaborative_model.pkl not found!")
except Exception as e:
    print(f"❌ ERROR: {e}")
