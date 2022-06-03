def check_stopping_criteria_synthetic(loss_history, round_num):
  if loss_history[-1] <= 1e-5 or round_num >= 10000: 
    return True
  elif len(loss_history) < 10:
    return False
  
  return False