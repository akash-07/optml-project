import tensorflow_federated as tff
import tensorflow as tf

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