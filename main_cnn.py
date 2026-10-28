import torch
import torch.nn as nn
from torchvision import models, transforms
from data_loader import get_data_loaders
from evaluate import evaluate_model
import os

def create_resnet18_model(num_classes, device, freeze_backbone=True):
    """
    Create a ResNet18 model, optionally freezing the backbone.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def main():
    # Dataset paths
    train_dir = r'dataset/Training'
    test_dir = r'dataset/Testing'

    # Hyperparameters
    batch_size = 32
    image_size = 128
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # AMP only if GPU
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP enabled for GPU.")
    else:
        print("AMP disabled on CPU.")

    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_loader, test_loader, class_names = get_data_loaders(
        train_dir, test_dir, batch_size, image_size,
        train_transform=train_transform,
        test_transform=test_transform
    )
    print(f"Classes: {class_names}")

    # Create model
    model = create_resnet18_model(len(class_names), device, freeze_backbone=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_acc = 0.0
    os.makedirs('models', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = 100 * correct_train / total_train

        # Validation
        model.eval()
        running_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / total_val
        val_acc = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_resnet18_model.pth')
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

    # Evaluate
    evaluate_model(model, test_loader, class_names, device)

if __name__ == '__main__':
    main()
