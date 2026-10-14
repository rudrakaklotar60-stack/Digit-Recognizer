def target(label):
    target = [0.0]*10
    target[label] = 1.0
    return target
    
def mse_loss (prediction, target):
    loss = sum((t-p)**2 for t, p in zip (prediction, target))
    return loss/10
    
def output_error(prediction, target):
    return [ (p- t) for p, t in zip(prediction, target) ]
    
def accuracy(prediction, label):
    prediction_digit = prediction.index(max(prediction ))
    return 1 if prediction_digit == label else 0