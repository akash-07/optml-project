import os
from generate_synthetic_data import generate_jsons


if __name__ == "__main__":
    data_dir = os.path.join("..", "synthetic_data")
    filename = "synthetic_data"
    n_clients = 20
    generate_jsons(data_dir, filename, n_clients)
