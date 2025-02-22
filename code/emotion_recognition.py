# pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn

import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define paths
zip_path = "archive.zip"
extract_path = "fer2013_data/"

# Extract if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Extraction complete")

train_dir = "fer2013_data/train/"
test_dir = "fer2013_data/test/"

print("Train categories:", os.listdir(train_dir))
print("Test categories:", os.listdir(test_dir))

import torch
RANDOM_STATE = 42
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
NUM_CLASSES = 7
torch.manual_seed(RANDOM_STATE)

# Data augmentation for better accuracy
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load images from directory
train_generator = datagen.flow_from_directory(
    "fer2013_data/train/",
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change from 'grayscale' to 'rgb'
    class_mode="categorical"
)

test_generator = datagen.flow_from_directory(
    "fer2013_data/test/",
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change from 'grayscale' to 'rgb'
    class_mode="categorical"
)

#----------------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

RANDOM_STATE = 42
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 7  # 7 emotion classes

train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),  # ensure images are grayscale
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define paths for training and testing data directories
train_dir = "fer2013_data/train"  # contains subfolders for each emotion
test_dir  = "fer2013_data/test"   # contains subfolders for each emotion



# Get emotion categories (subfolder names)
categories = sorted(os.listdir(train_dir))

# Number of images per category to display
num_images = 5

# Set up plot with multiple rows
fig, axes = plt.subplots(len(categories), num_images, figsize=(num_images * 3, len(categories) * 3))

# Loop through categories and select images
for row, category in enumerate(categories):
    category_path = os.path.join(train_dir, category)
    image_names = os.listdir(category_path)[:num_images]  # Select first 5 images

    for col, image_name in enumerate(image_names):
        image_path = os.path.join(category_path, image_name)
        img = Image.open(image_path).convert("L")  # Ensure RGB mode

        axes[row, col].imshow(img,cmap="gray")
        axes[row, col].set_title(category if col == 0 else "")  # Only set title for first column
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()

# Load datasets using ImageFolder (folder names define class labels)
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super(CNN, self).__init__()
        # Convolutional Blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.fc_layers(x)
        return x

# Instantiate the model (running on CPU)
model = CNN(channels=1, num_classes=NUM_CLASSES)
model.to("cuda")
summary(model, (1, 48, 48))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()  # Use class weights if needed for imbalanced classes

# Training loop: one epoch
def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / len(dataloader), 100.0 * correct / total

# Evaluation loop
def evaluate(model, dataloader, loss_fn):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total

best_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer)
    test_acc = evaluate(model, test_dataloader, loss_fn)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch [{epoch+1}/{EPOCHS}]: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

y_true = []
y_pred = []

# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.inference_mode():
    for inputs, labels in test_dataloader:
        inputs  = inputs.to(device)
        labels  = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Move tensors to CPU before converting to numpy
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute classification metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Macro F1 Score:", f1_score(y_true, y_pred, average='macro'))
print("Test Accuracy:", accuracy_score(y_true, y_pred))

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

