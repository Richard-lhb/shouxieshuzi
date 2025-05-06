import numpy as np
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error
from utils.activation_functions import softmax

def test_model(model, test_images, test_labels):
    # Forward pass
    outputs = model.forward(test_images)
    probabilities = softmax(outputs)
    predictions = np.argmax(probabilities, axis=0)

    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, predictions)
    test_recall = recall_score(test_labels, predictions, average='macro')
    test_rmse = np.sqrt(mean_squared_error(test_labels, predictions))

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')

    return test_accuracy, test_recall, test_rmse