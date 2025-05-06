import numpy as np
import time
from models.lenet import LeNet
from utils.activation_functions import sigmoid
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def train_model(model, train_images, train_labels, epochs=30, batch_size=64, learning_rate=0.01):
    num_batches = len(train_images) // batch_size
    loss_history = []
    accuracy_history = []
    recall_history = []
    rmse_history = []

    for epoch in range(epochs):
        total_loss = 0
        all_predictions = []
        all_labels = []

        # Learning rate decay
        current_lr = learning_rate * (0.95 ** epoch)

        for batch in range(num_batches):
            start_time = time.time()
            start = batch * batch_size
            end = start + batch_size
            batch_images = train_images[start:end]
            batch_labels = train_labels[start:end]

            if len(batch_images) == 0:
                continue

            # Forward pass
            outputs = model.forward(batch_images)
            probabilities = softmax(outputs)
            one_hot_labels = np.eye(10)[batch_labels].T

            # Calculate loss (cross-entropy)
            log_probs = outputs - np.log(np.sum(np.exp(outputs), axis=0, keepdims=True))
            loss = -np.sum(one_hot_labels * log_probs) / batch_size
            total_loss += loss

            # Backward pass
            d_output = (probabilities - one_hot_labels) / batch_size
            gradients = model.backward(d_output)

            # Update weights
            model.update_weights(*gradients, current_lr)

            # Calculate predictions
            predictions = np.argmax(probabilities, axis=0)
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)

            end_time = time.time()
            # 打印训练进度和时间信息
            if batch % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{num_batches}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f}s')

        # Epoch statistics
        average_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions, average='macro')
        rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))

        loss_history.append(average_loss)
        accuracy_history.append(accuracy)
        recall_history.append(recall)
        rmse_history.append(rmse)

        print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}, Acc: {accuracy:.4f}, Recall: {recall:.4f}, RMSE: {rmse:.4f}, LR: {current_lr:.6f}')

    return model, loss_history, accuracy_history, recall_history, rmse_history