import os, json
import numpy as np


def generate_data(name, mean_class1, var_class1, n_samples_class1, mean_class2, var_class2, n_samples_class2, n_dims):
    data = np.zeros((n_samples_class1 + n_samples_class2, n_dims))
    for i in range(n_dims):
        data[:, i] = np.concatenate([np.random.normal(mean_class1[i], var_class1[i], size=n_samples_class1), np.random.normal(mean_class2[i], var_class2[i], size=n_samples_class2)])
    labels = np.zeros((n_samples_class1 + n_samples_class2,))
    labels[n_samples_class1:] = 1

    return {name: {"x": data.tolist(), 1: labels.tolist()}}


def generate_jsons(data_dir, filename, n_clients, last_client_offset=3, mean_class1=(0, 0), var_class1=(0.25, 0.25), mean_class2=(1, 1), var_class2=(0.25, 0.25), n_samples_class1=50, n_samples_class2=50, n_dims=2):

    # JSON structure
    # {
    #   "users": [user_0, user_1, ..., user_n-1],
    #   "num_samples": [100, 120, ..., 80],
    #   "user_data": {
    #       user_0: {
    #           "x": data_value,
    #           "y": class_label
    #       },
    #       user_1: {
    #           "x": data_value,
    #           "y": class_label
    #       }
    #   }
    # }

    # Create structure with metadata
    json_data = {
        "users": np.arange(n_clients).astype(str).tolist(),
        "num_samples": [n_samples_class1 + n_samples_class2 for _ in range(n_clients)],
        "user_data": dict(),
    }

    # Create data for each client and add to json data structure
    for client in json_data["users"]:
        if client == n_clients - 1:  # Last client has a different mean for the distribution of the second class
            mean_class2 = (mean_class2[0] + last_client_offset, mean_class2[1] + last_client_offset)
        data = generate_data(client, mean_class1, var_class1, n_samples_class1, mean_class2, var_class2, n_samples_class2, n_dims)
        json_data["user_data"].update(data)

    # Create a json file and save the data in it
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, filename + '.json'), 'w') as f:
        json.dump(json_data, f,  indent=4)
