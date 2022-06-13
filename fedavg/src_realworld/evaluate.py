from models import *

def evaluate_femnist(server_model_weights, central_test_dataset):
  keras_model = get_femnist_cnn()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  keras_model.set_weights(server_model_weights)
  return keras_model.evaluate(central_test_dataset)