from error import output_error

def sig_der(x):
    return x*(1-x)
    
def train(image, h1, h2, pred, target, brain, lr):
    out_error = output_error(pred, target)
    out_del = [err*sig_der(p) for err, p in zip(out_error, pred)]
    
    h2_del = []
    for j in range (32):
        error_h2 = sum(brain['w3'][k][j]* out_del[k] for k in range (10))
        h2_del.append(error_h2*sig_der(h2[j]))
        
    h1_del = []
    for j in range (32):
        error_h1 = sum(brain['w2'][k][j]*h2_del[k] for k in range(32))
        h1_del.append(error_h1*sig_der(h1[j]))
        
    for k in range (10):
        for j in range (32):
            brain['w3'][k][j] -= lr*out_del[k]*h2[j]
        brain['b3'][k] -=  lr*out_del[k]
        
    for k in range (32):
        for j in range (32):
           brain['w2'][k][j] -=lr*h2_del[k]*h1[j]
        brain['b2'][k] -= lr*h2_del[k]
            
    for k in range (32):
        for j in range (784):
            brain['w1'][k][j] -=lr*h1_del[k]*image[j]
        brain ['b1'][k] -= lr*h1_del[k]
        