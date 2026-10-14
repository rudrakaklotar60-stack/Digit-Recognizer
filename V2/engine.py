import pickle
import random
from core import forward_pass
from error import target, mse_loss, accuracy
from training import train


with open('mnist_flattened.pkl', 'rb') as f:
    dataset = pickle.load(f)
    images, labels = dataset['images'], dataset['labels']


with open('init.pkl', 'rb') as f:
    brain = pickle.load(f)

lr = 0.001
epochs = int(input("Enter epochs : "))

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch} Starting ---")
    
 
    data = list(zip(images, labels))
    random.shuffle(data)
    
    total_loss = 0
    correct = 0
    
    for i, (img, lbl) in enumerate(data):
        h1, h2, pred = forward_pass(img, brain)
        target_ = target(lbl)
        
        train(img, h1, h2, pred, target_, brain, lr)
        
        total_loss += mse_loss(pred, target_)
        correct += accuracy(pred, lbl)
        
        if i % 1000 == 0 and i > 0:
            print(f"Img {i}/42000 | Loss: {total_loss/i} | Acc: {(correct/i)*100}%")

    # Save progress
    with open(f'trained_brain_e{epoch+13}.pkl', 'wb') as f:
        pickle.dump(brain, f)

print("\nMISSION ACCOMPLISHED")
