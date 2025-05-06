from models.lenet import LeNet
from utils.data_loader import get_data_loaders
from train import train_model
from test import test_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load data
    train_images, train_labels, test_images, test_labels = get_data_loaders()

    # Initialize model
    model = LeNet()

    # Train model
    trained_model, loss_history, accuracy_history, recall_history, rmse_history = train_model(
        model, train_images, train_labels, epochs=10, batch_size=64, learning_rate=0.01
    )

    # Plot training curves
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(recall_history)
    plt.title('Training Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    plt.subplot(2, 2, 4)
    plt.plot(rmse_history)
    plt.title('Training RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.tight_layout()
    plt.show()

    # Test model
    test_accuracy, test_recall, test_rmse = test_model(trained_model, test_images, test_labels)

    print("\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")