import torch
from data_loader import get_data_loaders
from model import create_vit_model
from train import train_model
from evaluate import evaluate_model

def main():
    train_dir = r"C:\Users\admin\OneDrive\Documents\BrainTumor_x\dataset\Training"
    test_dir = r"C:\Users\admin\OneDrive\Documents\BrainTumor_x\dataset\Testing"    
    batch_size = 16
    image_size = 224
    epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    train_loader, test_loader, class_names = get_data_loaders(train_dir, test_dir, batch_size, image_size)
    print(f"Classes: {class_names}")

    print("Initializing model...")
    model = create_vit_model(num_classes=len(class_names), device=device)

    print("Starting training...")
    model, train_losses, train_acc, val_losses, val_acc = train_model(
    model, train_loader, test_loader, device, epochs=10)

    print("Evaluating model...")
    evaluate_model(model, test_loader, class_names, device)

if __name__ == "__main__":
    main()
    
    