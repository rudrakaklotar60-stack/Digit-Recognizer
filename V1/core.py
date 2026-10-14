from maths import dot_product, sigmoid

def forward_pass(image, brain):
    # Layer 1 (784 -> 16)
    h1 = [sigmoid(dot_product(image, w) + b) for w, b in zip(brain['w1'], brain['b1'])]
    
    # Layer 2 (16 -> 16)
    h2 = [sigmoid(dot_product(h1, w) + b) for w, b in zip(brain['w2'], brain['b2'])]
    
    # Layer 3 (16 -> 10)
    out = [sigmoid(dot_product(h2, w) + b) for w, b in zip(brain['w3'], brain['b3'])]
    
    return h1, h2, out
    