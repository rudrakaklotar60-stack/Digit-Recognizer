import math

def dot_product(inputs, weights):
   
    return sum(i * w for i, w in zip(inputs, weights))

def sigmoid(x):

    if x < -700: return 0.0
    if x > 700: return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output_val):
    return output_val * (1.0 - output_val)
    