import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from model import CustomLSTM, STGCNModel, CTRGCNModel
from training import get_adjacency_matrix, reshape_for_stgcn


def evaluate(model, X_test, y_test, gestures, model_type, model_path):
    # Load the specific weights provided from the main loop
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=device))

    save_dir = f'./model/{model_type}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

        # Convert to class indices for metrics
        y_pred = test_outputs.argmax(dim=1)
        y_true = y_test.argmax(dim=1)

        accuracy = (y_pred == y_true).float().mean()
        print(f'\nFinal Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

        # Convert to numpy for sklearn/seaborn
        predicted_labels = y_pred.cpu().numpy()
        true_labels = y_true.cpu().numpy()

        # 1. Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
        plt.title(f'Confusion Matrix - {model_type}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"{save_dir}confusion_matrix.png")
        plt.show()

        # 2. Classification Report
        report_text = classification_report(true_labels, predicted_labels, target_names=gestures, zero_division=0)
        class_report_dict = classification_report(true_labels, predicted_labels, target_names=gestures,
                                                  output_dict=True, zero_division=0)
        print("\nClassification Report:")
        print(report_text)

        with open(f"{save_dir}classification_report.txt", "w") as f:
            f.write(f"Final Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}\n")
            f.write(report_text)

        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame(class_report_dict).iloc[:-1, :].T, annot=True, cmap='Blues')
        plt.title(f'Classification Report - {model_type}')
        plt.savefig(f"{save_dir}classification_heatmap.png")
        plt.show()


if __name__ == "__main__":

    # MODEL_TYPE = "LSTM"
    # MODEL_TYPE = "STGCN"
    MODEL_TYPE = "CTRGCN"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = f'./model/{MODEL_TYPE}/'
    MODEL_PATH = f'{model_dir}best_model.pth'

    X_raw = np.load('X_TRAIN_01.npy')
    y_raw = np.load('y_TRAIN_01.npy')
    gestures = np.load("gestures_01.npy")

    if MODEL_TYPE == "LSTM":
        X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
        model = CustomLSTM(input_size=258, hidden_size=64, num_classes=len(gestures)).to(device)
    else:
        X_tensor = torch.tensor(reshape_for_stgcn(X_raw), dtype=torch.float32).to(device)
        if MODEL_TYPE == "STGCN":
            model = STGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "CTRGCN":
            model = CTRGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        else:
            print("Model Not Found !")
            exit()

    y_tensor = torch.tensor(y_raw, dtype=torch.long).to(device)
    y_one_hot = torch.nn.functional.one_hot(y_tensor).float().to(device)
    _, X_test, _, y_test = train_test_split(X_tensor, y_one_hot, test_size=0.2)

    evaluate(model, X_test, y_test, gestures, MODEL_TYPE, MODEL_PATH)
