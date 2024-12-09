import numpy as np
import os as s
import cv2 as cv
import matplotlib.pyplot as plt

print('loading data')
data = 'data/train'  # Path to the dataset
labels = []
images = []
image_size = 240

# Loop through the dataset folder
for imagefile in s.listdir(data):
    image_path = s.path.join(data, imagefile)
    image = cv.imread(image_path, cv.IMREAD_COLOR)  # Read image
    if image is None:
        print(f"Failed to load image: {imagefile}")
        continue  # Skip if image fails to load

    image = cv.resize(image, (image_size, image_size)) # Resize image to (120, 120)

    # Assign labels based on file name
    if imagefile.startswith("c"):
        label = 0
    elif imagefile.startswith("d"):
        label = 1
    else:
        label = 2  

    labels.append(label)
    images.append(image)

print('Labels count:', len(labels))
print('Images count:', len(images))

# Convert lists to NumPy arrays
labels = np.array(labels)
images = np.array(images)

# Convert labels to one-hot encoding
num_classes = 2  # Number of classes
labels_one_hot = np.eye(num_classes)[labels]

# Shuffle  data while preserving  pairing between images and labels
print("Shuffling data...")
indices = np.random.permutation(len(images))  # Generate random indices
images = images[indices]  # Shuffle images
labels_one_hot = labels_one_hot[indices]  # Shuffle corresponding one-hot encoded labels

# Print shuffled data details
print('Sample label (one-hot encoded):', labels_one_hot[0])
print('Array images dimension:', images.shape)
print('Image dimension:', images[0].shape)
print('Labels one-hot shape:', labels_one_hot.shape)



#now cnn
stride = 1
padding = 0


batch_sz = 60
num_batches = len(images) // batch_sz
batch_img = []
batch_lab = []

for h in range(num_batches):
    start = h * batch_sz
    end = (h + 1) * batch_sz
    batch = images[start:end]
    batch_l = labels_one_hot[start:end]
    batch_img.append(batch)
    batch_lab.append(batch_l)
batch_img = np.array(batch_img)
batch_lab = np.array(batch_lab)





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np
import os as s
import cv2 as cv
import matplotlib.pyplot as plt



class AlexNet(nn.Module):
        def __init__(self, num_classes = 2):
            super(AlexNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),
                #nn.BatchNorm2d(96),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            
            self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
                #nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
                #nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size = 3, stride = 2))
            
            self.fc = nn.Sequential(
                #nn.Dropout(0.5),
                nn.Linear(1024, 4096),
                nn.ReLU())
            self.fc1 = nn.Sequential(
                #nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU())
            self.fc2= nn.Sequential(
                nn.Linear(4096, num_classes))

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.fc1(out)
            out = self.fc2(out)
            return out
# Create model instance
#model2 = AlexNet()
# Loss function and optimizer
model2 = AlexNet()
# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
opt = optimizer.SGD(model2.parameters(),  lr=0.0005)  # Fixed optimizer variable name

def train(epoch):
    for i in range(epoch):
        epoch_loss = 0  # To track total loss for the epoch
        print(f"\nEpoch {i + 1}/{epoch}")
        
        for ba in range(num_batches):
            opt.zero_grad()  # Reset gradients
            
            # Prepare the batch data
            batch_im = torch.tensor(batch_img[ba], dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: (batch_sz, 3, 240, 240)
            batch_ll = torch.tensor(batch_lab[ba], dtype=torch.long).argmax(dim=1)  # Convert one-hot labels to integers
            
            # Forward pass
            output = model2(batch_im)  # Model's predictions
            
            # Compute loss for the batch
            loss = loss_function(output, batch_ll)
            
            # Backpropagation
            loss.backward()
            opt.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            #print(f"Batch {ba + 1}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Print average loss for the epoch
        print(f"Epoch {i + 1} Average Loss: {epoch_loss / num_batches:.4f}")

results = train(100)

