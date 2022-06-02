from models import *
from models import *
from models import *

def evaluate_synthetic(server_model_weights, central_test_dataset):
  model = MyPerceptronKeras()
  loss = PerceptronLoss()
  model.set_weights(server_model_weights)
  loss_val = tf.Variable(0.0, dtype=tf.float64)
  count = 0
  for batch in central_test_dataset:
    y_predicted = model(batch[0])
    y_true = batch[1]
    loss_val.assign_add(loss(y_true, y_predicted))
    count += 1
  return loss_val.numpy()/count