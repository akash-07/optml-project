from models import *

def evaluate_femnist(server_model_weights, central_test_dataset):
  keras_model = get_femnist_cnn()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  keras_model.set_weights(server_model_weights)
  return keras_model.evaluate(central_test_dataset)

def evaluate_synthetic(server_model_weights, central_test_dataset):
  keras_model = get_perceptron()
  keras_model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)]  
  )
  keras_model.set_weights(server_model_weights)
  return keras_model.evaluate(central_test_dataset)