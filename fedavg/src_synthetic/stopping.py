def check_stopping_criteria_femnist(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.77 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False

def check_stopping_criteria_synthetic(loss_history, acc_history, cur_test_accuracy, round_num):
  if cur_test_accuracy >= 0.95 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False