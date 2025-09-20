import os
import sys  # ✅ ADD THIS
import dill
import shap
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors

# ✅ Add scripts/ to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from shap_wrapper import SurpriseWrapper, ContentWrapper, GraphWrapper, FederatedWrapper

model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

print("📦 Loading models for SHAP explainability...")

# ✅ Collaborative
with open(os.path.join(model_dir, "collaborative_model.pkl"), "rb") as f:
    collab_model = SurpriseWrapper(pickle.load(f))

sample_input_collab = np.array([[1, 1], [2, 2], [3, 3]])
explainer_collab = shap.KernelExplainer(collab_model.predict, sample_input_collab)
with open(os.path.join(model_dir, "shap_explainer_collaborative.pkl"), "wb") as f:
    dill.dump(explainer_collab, f)
print("✅ SHAP explainer for collaborative saved.")

# ✅ Content-Based
with open(os.path.join(model_dir, "content_model.pkl"), "rb") as f:
    content_sim = pickle.load(f)
content_model = ContentWrapper(content_sim, user_idx=1)
sample_products = list(range(10))
explainer_content = shap.KernelExplainer(content_model.predict, np.array([sample_products]))
with open(os.path.join(model_dir, "shap_explainer_content.pkl"), "wb") as f:
    dill.dump(explainer_content, f)
print("✅ SHAP explainer for content saved.")

# ✅ Graph-Based
graph_model = KeyedVectors.load(os.path.join(model_dir, "graph_model.kv"))
graph_wrapper = GraphWrapper(graph_model, user_node="user_1")
explainer_graph = shap.KernelExplainer(graph_wrapper.predict, np.array([sample_products]))
with open(os.path.join(model_dir, "shap_explainer_graph.pkl"), "wb") as f:
    dill.dump(explainer_graph, f)
print("✅ SHAP explainer for graph saved.")

# ✅ Federated (Session-Based)
session_model = load_model(os.path.join(model_dir, "session_model.h5"))
session_tail = [1, 2]  # example session
fed_wrapper = FederatedWrapper(session_model, session_tail=session_tail)
explainer_federated = shap.KernelExplainer(fed_wrapper.predict, np.array([sample_products]))
with open(os.path.join(model_dir, "shap_explainer_federated.pkl"), "wb") as f:
    dill.dump(explainer_federated, f)
print("✅ SHAP explainer for federated saved.")
