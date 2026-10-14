import random
import pickle

def init_layer(inputs, outputs):
    return [[random.uniform(-0.1, 0.1) for _ in range(inputs)] for _ in range(outputs)]

# Layer 1: Input to Hidden 1
w1 = init_layer(784, 16)
b1 = [random.uniform(-0.1, 0.1) for _ in range(16)]

# Layer 2: Hidden 1 to Hidden 2 
w2 = init_layer(16, 16)
b2 = [random.uniform(-0.1, 0.1) for _ in range(16)]

# Layer 3: Hidden 2 to Output
w3 = init_layer(16, 10)
b3 = [random.uniform(-0.1, 0.1) for _ in range(10)]


with open('deep_brain_init.pkl', 'wb') as f:
    pickle.dump({
        'w1': w1, 'b1': b1,
        'w2': w2, 'b2': b2,
        'w3': w3, 'b3': b3
    }, f)

print("random weights intilised")
