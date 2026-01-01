import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
DATA_DIR = './dataset'

def get_model():
    """
    Loads MobileNetV3-Small and modifies it for:
    1. Grayscale input (1 channel instead of 3)
    2. 4 Output classes (Logo, Oracle, Fake, Background)
    """
    # Load pre-trained MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # 1. Modify First Layer: Change from 3 channels (RGB) to 1 channel (Grayscale)
    original_layer = model.features[0][0]
    new_layer = nn.Conv2d(1, original_layer.out_channels, 
                          kernel_size=original_layer.kernel_size, 
                          stride=original_layer.stride, 
                          padding=original_layer.padding, 
                          bias=False)
    
    # Sum the weights to preserve pre-trained features
    with torch.no_grad():
        new_layer.weight[:] = torch.sum(original_layer.weight, dim=1, keepdim=True)
    
    model.features[0][0] = new_layer

    # 2. Modify Classifier Layer: Change output to 4 classes
    # 0: Logo, 1: Oracle, 2: Fake, 3: Background
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 4) 
    
    return model

def main():
    print("--- STARTING TRAINING ---")

    # --- 1. DATA AUGMENTATION ---
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        
        # Zoom Augmentation (Keep this!)
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        
        # REMOVED: transforms.RandomRotation(degrees=10) <--- DELETE THIS LINE
        # We manually added 90/180/270 rotations, so we don't need random spinning.
        
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3), # Reduced slightly
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' not found. Please check folder structure.")
        return

    # Load Data
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    
    # USE 100% OF DATA
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Classes detected: {full_dataset.classes}")
    print(f"Total training images: {len(full_dataset)}")

    # --- 2. SETUP MODEL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = get_model().to(device)
    
    # --- UPGRADE: CLASS WEIGHTS ---
    # This solves the bias. We tell the math:
    # "Pay 20% MORE attention to Logo (1.2)"
    # "Pay 50% LESS attention to Background (0.5)" (Because background is easy)
    # Order: [Logo, Oracle, Fake, Background]
    class_weights = torch.tensor([1.2, 1, 1, 0.5]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

    # --- 4. SAVE MODEL ---
    torch.save(model.state_dict(), "robocon_model.pth")
    print("\nSUCCESS: Model saved as 'robocon_model.pth'")

if __name__ == "__main__":
    main()