import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(csv_path='metrics/training_metrics.csv'):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Plot accuracy graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Plot loss graph
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')  # Save the graphs
    plt.close()
    print("Graphs saved as 'training_curves.png'")

# Example usage: Run this after training
if __name__ == "__main__":
    plot_training_curves()