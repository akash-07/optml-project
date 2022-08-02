import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
import os

nest_asyncio.apply()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_perceptron():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,))
    ])


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = get_perceptron()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 2], dtype=tf.float64),
            tf.TensorSpec(shape=[None,], dtype=tf.float64)),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])    

#------------------------------------------------------------------------------ 
# SERVER_STATE = {model weights}
@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables

# {model weights}@SERVER
@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)

#------------------------------------------------------------------------------
# Defining type signatures - 1
model_weights_type = server_init.type_signature.result
dummy_model = model_fn()
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
federated_server_state_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

#------------------------------------------------------------------------------
@tf.function
def client_update(model, dataset, full_training_dataset, server_weights, lr, accumulator, grad_store1, grad_store2):
    
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)
    
    # Update the local model using SGD.
    for batch in dataset:
        with tf.GradientTape(persistent=True) as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)

        # Apply the gradient using a client optimizer.
        updated_accumulator = tf.nest.map_structure(lambda a, g: -lr*g, accumulator, grads)
        updated_weights = tf.nest.map_structure(lambda w, a: w+a, client_weights, updated_accumulator)
        
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, updated_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), accumulator, updated_accumulator)
        
        # Simulate freezing of normal vector, train only biases
        client_weights[0].assign([[2.700213],[0.7134238]])
    
    # dictionary containing  {metric: [sum, count], ..}
    out_data = model.report_local_outputs()
    
    # Compute the gradient of local objective
    for batch in dataset.take(1):
        with tf.GradientTape(persistent=True) as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads_1 = tape.gradient(outputs.loss, client_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), grad_store1, grads_1)

    # Compute gradient of global objective
    num_batches = full_training_dataset.cardinality()
    tf.assert_equal(num_batches, tf.cast(1, tf.int64))
    for batch in full_training_dataset:
        with tf.GradientTape(persistent=True) as tape:
             # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads_2 = tape.gradient(outputs.loss, client_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y), grad_store2, grads_2)

    # Just consider the l2 norm of biases
    grad_diff = tf.abs(grad_store1[1]-grad_store2[1])

    return client_weights, grad_diff, out_data, out_data['loss'][1]

@tff.tf_computation(tf_dataset_type, tf_dataset_type, model_weights_type, tf.float32)
def client_update_fn(tf_dataset, full_training_dataset, server_weights_at_client, learning_rate):
    model = model_fn()
    
    # To be used when optimizer is SGD with momentum
    accumulator = tf.nest.map_structure(lambda l: tf.Variable(tf.zeros(l.shape, l.dtype)), server_weights_at_client)

    # For evaluting gradient difference
    grad_store1 = tf.nest.map_structure(lambda l: tf.Variable(tf.zeros(l.shape, l.dtype)), server_weights_at_client)
    grad_store2 = tf.nest.map_structure(lambda l: tf.Variable(tf.zeros(l.shape, l.dtype)), server_weights_at_client)
    
    client_weights, grad_diff, out_data, n = client_update(model, tf_dataset, full_training_dataset,
        server_weights_at_client, learning_rate, accumulator, grad_store1, grad_store2)

    return client_weights, grad_diff, out_data, n

#------------------------------------------------------------------------------ 
@tf.function
def server_update(model, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
    return model_weights

@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)

#------------------------------------------------------------------------------ 
# Defining type signatures - 2
client_learning_rates_type = tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False)
client_agg_weights_type = tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False)

#------------------------------------------------------------------------------
# Helper functions for data extraction and processing
@tff.tf_computation(client_update_fn.type_signature.result)
def extract_only_weights(tp_wts_mts):
  return tp_wts_mts[0]

@tff.tf_computation(client_update_fn.type_signature.result)
def extract_grad_diff(tp_wts_mts):
    return tp_wts_mts[1]

@tff.tf_computation(client_update_fn.type_signature.result)
def extract_training_metrics(tp_wts_mts):
    return tp_wts_mts[2]

# Receives a dictionary {metric: [sum_all_samples, total_samples]}
# Return dictionary of means for every metric
@tff.tf_computation(client_update_fn.type_signature.result[2])
def get_mean(metric_dict):
    d = {}
    for k,v in metric_dict.items():
        d[k] = v[0]/v[1] 
    return d

#------------------------------------------------------------------------------
@tff.federated_computation(
    federated_server_state_type, 
    federated_dataset_type, # List of client datasets
    federated_dataset_type, # Full training dataset
    client_learning_rates_type, 
    client_agg_weights_type) # List of p_i = n_i/n
def next_fn(
    server_state, 
    federated_dataset,
    full_training_dataset, 
    client_learning_rates,
    client_agg_weights):
    
    # Broadcast the server weights to the clients.
    server_weights_at_clients = tff.federated_broadcast(server_state)

    # Each client computes their updated weights.
    # Epochs and lr supplied by orchestrator (i.e us) 
    # instead of server for the purposes of simulation
    client_weights_and_metrics = tff.federated_map(
        client_update_fn, (federated_dataset, full_training_dataset, server_weights_at_clients, client_learning_rates))
    
    client_weights = tff.federated_map(extract_only_weights, client_weights_and_metrics)
    client_metrics = tff.federated_map(extract_training_metrics, client_weights_and_metrics)
    grad_diffs = tff.federated_map(extract_grad_diff, client_weights_and_metrics)
    
    # Weighted averaging of client models - sum p_i x w_i
    mean_client_weights = tff.federated_mean(client_weights, client_agg_weights)
    
    # compute mean of training metrics
    client_metrics_summed = tff.federated_sum(client_metrics)
    mean_metrics = tff.federated_map(get_mean, client_metrics_summed)

    # The server updates its model.
    server_state = tff.federated_map(server_update_fn, mean_client_weights)

    return server_state, mean_metrics, grad_diffs