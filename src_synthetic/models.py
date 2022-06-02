import tensorflow_federated as tff
import tensorflow as tf

#------------------------------------------------------------------------------
class MyPerceptronKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable([-1.0, 1.0], dtype=tf.float64)
        self.b = tf.Variable(-1.0, dtype=tf.float64)

    def __call__(self, xs, **kwargs):
        return tf.add(tf.tensordot(xs, self.w, 1), self.b)

class PerceptronLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_predicted):
        return tf.reduce_sum(tf.maximum(tf.cast(0.0, tf.float64), -y_true*y_predicted))

# def perceptron_loss(y_predicted, y_true):
#     return tf.reduce_sum(tf.maximum(tf.zeros(y_true.shape), -y_true*y_predicted))

def tff_perceptron_model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = MyPerceptronKeras()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 2], dtype=tf.float64), tf.TensorSpec(shape=[None], dtype=tf.float64)),
        loss=PerceptronLoss()) 