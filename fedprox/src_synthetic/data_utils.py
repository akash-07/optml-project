import tensorflow as tf
import os
import json
import numpy as np
from collections import defaultdict
from math import floor, ceil

#------------------------------------------------------------------------------
class SyntheticData:
    def __init__(self, train_dir, test_dir):
        with open(os.path.join(train_dir, 'train.json')) as f:
            train_d = json.load(f)
        
        self.client_ids, self.num_samples, self.train_data = train_d['users'], train_d['num_samples'], train_d['user_data']
        
        with open(os.path.join(test_dir, 'test.json')) as f:
            test_d = json.load(f)
        
        self.test_data = test_d['user_data']

    def get_client_ids(self):
        return self.client_ids

    def create_dataset_for_client(self, client_id):
        client_data = self.train_data[client_id]
        return tf.data.Dataset.from_tensor_slices((client_data['x'], client_data['y'])).map(lambda a,b: (tf.cast(a, tf.float64), tf.cast(b, tf.float64)))

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
        return tf.data.Dataset.from_tensor_slices((xs, ys)).map(lambda a,b: (tf.cast(a, tf.float64), tf.cast(b, tf.float64)))

#------------------------------------------------------------------------------
def make_federated_data(dataset, preprocess_fn, client_ids, client_num_samples, client_capacities, batch_size, round_num):
    return [
      preprocess_fn(dataset.create_dataset_for_client(client_ids[i]), batch_size, client_capacities[i], client_num_samples[i], round_num)
      for i in range(len((client_ids)))
    ]

#------------------------------------------------------------------------------
def preprocess_synthetic(dataset, b, u, n, r):
    u_p = floor(n/b)
    if(u <= u_p):
        return dataset.shuffle(n, seed=r).batch(b).take(u)
    else:
        x = ceil((b*u)/n)
        return dataset.repeat(x).shuffle(n, seed=r).batch(b).take(u)