import pickle
import os
from PIL import Image


train_path = "path/to/training/data/set"

def process_and_save():
    all_images = []
    all_labels = []
    
    print("Starting...")
    
    for digit in range(10):
        folder = os.path.join(train_path, str(digit))
        
        for img_name in os.listdir(folder):
            #  Open and convert to Greyscale
            img = Image.open(os.path.join(folder, img_name)).convert('L')
            #  Flatten and Normalize (0.0 to 1.0)
            pixels = [p / 255.0 for p in list(img.getdata())]
            
            all_images.append(pixels)
            all_labels.append(digit)
            
    
    with open('mnist_flattened.pkl', 'wb') as f:
        pickle.dump({'images': all_images, 'labels': all_labels}, f)
        
    print("Done.")

process_and_save()
