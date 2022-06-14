import tensorflow as tf
import os
import json
import numpy as np
from collections import defaultdict
from math import floor, ceil

#------------------------------------------------------------------------------
def read_dir(data_dir):
    clients = []
    num_samples = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            client_data = json.load(inf)
        clients.extend(client_data['users'])
        num_samples.extend(client_data['num_samples'])
        data.update(client_data['user_data'])

    return clients, num_samples, data


class FemnistData:
    def __init__(self, train_dir, test_dir):
        self.client_ids, self.num_samples, self.train_data = read_dir(
            train_dir)
        _, _, self.test_data = read_dir(test_dir)

    def get_client_ids(self):
        return self.client_ids

    def create_dataset_for_client(self, client_id):
        client_data = self.train_data[client_id]
        return tf.data.Dataset.from_tensor_slices((client_data['x'], client_data['y']))

    def create_train_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.train_data.values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def create_test_dataset_for_all_clients(self):
        xs = list()
        ys = list()
        for data in self.test_data.values():
            for x in data['x']:
                xs.append(x)
            for y in data['y']:
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        return tf.data.Dataset.from_tensor_slices((xs, ys))

#------------------------------------------------------------------------------
def femnist_batch_format_fn(x, y):
    # Modify shape (28, 28) to (28, 28, 1)
    return (tf.reshape(x, [-1, 28, 28, 1]),
        tf.expand_dims(y, axis=-1))

#------------------------------------------------------------------------------
def make_federated_data(dataset, preprocess_fn, client_ids, client_num_samples, client_capacities, batch_size, round_num):
    return [
      preprocess_fn(dataset.create_dataset_for_client(client_ids[i]), batch_size, client_capacities[i], client_num_samples[i], round_num)
      for i in range(len((client_ids)))
    ]

#------------------------------------------------------------------------------
def preprocess_femnist(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).map(femnist_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).map(
            femnist_batch_format_fn).prefetch(tf.data.AUTOTUNE).take(u)
