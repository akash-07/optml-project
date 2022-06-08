import tensorflow_federated as tff
import tensorflow as tf
import collections

#------------------------------------------------------------------------------
def get_femnist_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (5, 5)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(62, activation='softmax')
    ])


def tff_femnist_model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = get_femnist_cnn()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

#------------------------------------------------------------------------------
def get_perceptron():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,))
    ])


def tff_perceptron_model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = get_perceptron()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 2], dtype=tf.float64),
            tf.TensorSpec(shape=[None,], dtype=tf.float64)),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])    