import flwr as fl

if __name__ == "__main__":
    print("🚀 Starting Flower Federated Learning Server...")
    fl.server.start_server(server_address="localhost:8080")
