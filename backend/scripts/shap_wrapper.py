import numpy as np

class SurpriseWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """
        X: np.ndarray of shape (n_samples, 2) with user_id and product_id
        """
        return np.array([
            self.model.predict(int(user_id), int(product_id)).est
            for user_id, product_id in X
        ])


class ContentWrapper:
    def __init__(self, sim_matrix, user_idx):
        self.sim_matrix = sim_matrix
        self.user_idx = user_idx

    def predict(self, product_ids):
        # ✅ Ensure product_ids is a flat list (SHAP passes 2D arrays)
        if isinstance(product_ids, np.ndarray):
            product_ids = product_ids.flatten()

        if isinstance(self.sim_matrix, dict):
            user_scores = self.sim_matrix.get(self.user_idx, {})
            return np.array([user_scores.get(int(pid), 0) for pid in product_ids])
        else:
            return np.array([
                self.sim_matrix[self.user_idx][pid] if pid < self.sim_matrix.shape[1] else 0
                for pid in product_ids
            ])


class GraphWrapper:
    def __init__(self, graph_model, user_node):
        self.graph_model = graph_model
        self.user_node = user_node

    def predict(self, product_ids):
        return np.array([
            self.graph_model.similarity(self.user_node, f"product_{pid}")
            if f"product_{pid}" in self.graph_model else 0
            for pid in product_ids
        ])


class FederatedWrapper:
    def __init__(self, model, session_tail):
        self.model = model
        self.session_tail = session_tail

    def predict(self, product_ids):
        # Ensure flat list of product_ids
        if isinstance(product_ids, np.ndarray):
            product_ids = product_ids.flatten()

        input_sequences = []
        for pid in product_ids:
            pid = int(pid)
            input_seq = self.session_tail[-2:] + [pid]
            input_seq = input_seq[:3]  # ensure exactly 3 elements
            input_sequences.append(input_seq)

        X_input = np.array(input_sequences)
        try:
            return self.model.predict(X_input).flatten()
        except Exception as e:
            print(f"⚠️ FederatedWrapper prediction failed: {e}")
            return np.zeros(len(product_ids))
