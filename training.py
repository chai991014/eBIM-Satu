import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mediapipe as mp
from model import CustomLSTM, STGCNModel, CTRGCNModel, SkateFormerModel


# --- STGCN Helpers ---
def get_adjacency_matrix():
    A = np.eye(75)
    mp_holistic = mp.solutions.holistic
    for conn in mp_holistic.POSE_CONNECTIONS:
        A[conn[0], conn[1]] = 1;
        A[conn[1], conn[0]] = 1
    for conn in mp_holistic.HAND_CONNECTIONS:
        A[conn[0] + 33, conn[1] + 33] = 1;
        A[conn[1] + 33, conn[0] + 33] = 1
        A[conn[0] + 54, conn[1] + 54] = 1;
        A[conn[1] + 54, conn[0] + 54] = 1
    return A


def reshape_for_stgcn(X):
    N, T, _ = X.shape
    X_new = np.zeros((N, T, 75, 3))
    for i in range(N):
        for t in range(T):
            frame = X[i, t]
            pose = frame[0:132].reshape(33, 4)[:, :3]
            lh = frame[132:195].reshape(21, 3)
            rh = frame[195:258].reshape(21, 3)
            X_new[i, t] = np.concatenate([pose, lh, rh], axis=0)
    return X_new.transpose(0, 3, 1, 2)


def train(model, model_type, train_loader, test_loader, device, num_epochs=200):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    model_dir = f'./model/{model_type}/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    top_3_models = []

    patience = 15  # Stop if test loss doesn't improve for 15 epochs
    best_test_loss = float('inf')
    early_stop_counter = 0
    best_overall_score = (-1, float('-inf'))
    final_best_path = f'{model_dir}best_model.pth'

    model_prefix = f'{model_dir}epoch_'

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (batch_X, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            correct_train += (outputs.argmax(dim=1) == batch_y.argmax(dim=1)).sum().item()
            total_train += batch_y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_train / total_train

        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        with torch.no_grad():
            for b_x, b_y in test_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                with torch.cuda.amp.autocast():
                    out = model(b_x)
                    running_test_loss += criterion(out, b_y).item() * b_x.size(0)
                    correct_test += (out.argmax(1) == b_y.argmax(1)).sum().item()

        test_loss = running_test_loss / len(test_loader.dataset)
        test_acc = correct_test / len(test_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)

        # 1. Primary Metric: Test Accuracy | 2. Secondary Metric: Test Loss (tie-breaker)
        current_score = (test_acc, -test_loss)  # Lower loss is better, hence negative for sorting

        if current_score > best_overall_score:
            best_overall_score = current_score
            torch.save(model.state_dict(), final_best_path)

        if len(top_3_models) < 3 or current_score > top_3_models[0][0]:
            current_epoch = epoch + 1
            current_model_path = f"{model_prefix}{current_epoch}.pth"

            torch.save(model.state_dict(), current_model_path)
            # Store accuracy, loss (raw), epoch, and path
            top_3_models.append((current_score, current_epoch, current_model_path))

            # Sort by the score tuple (Accuracy first, then lower loss)
            top_3_models.sort(key=lambda x: x[0])

            if len(top_3_models) > 3:
                _, _, worst_path = top_3_models.pop(0)
                if os.path.exists(worst_path):
                    os.remove(worst_path)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0  # Reset if we find a new best loss
        else:
            early_stop_counter += 1  # Increment if loss doesn't improve

        if early_stop_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch + 1}!')
            break

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f} Test Acc: {test_acc.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f} Test Acc: {test_acc:.4f}')

    end_time = time.time()
    total_time = end_time - start_time

    # Plotting code remains same as LSTM.py
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training & Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Training & Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"{model_dir}learning_curves.png"
    plt.savefig(save_path)

    plt.show()

    print("\nTraining Complete.")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print("Top 3 Models saved (by epoch):")
    for acc, ep, path in sorted(top_3_models, key=lambda x: x[1]):
        print(f"- {path} (Accuracy: {acc[0]:.4f})")


if __name__ == "__main__":

    # MODEL_TYPE = "LSTM"
    # MODEL_TYPE = "STGCN"
    # MODEL_TYPE = "CTRGCN"
    MODEL_TYPE = "SKATEFORMER"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw = np.load('X_TRAIN_01.npy')
    y_raw = np.load('y_TRAIN_01.npy')
    gestures = np.load("gestures_01.npy")

    if MODEL_TYPE == "LSTM":
        X_tensor = torch.tensor(X_raw, dtype=torch.float32)
        model = CustomLSTM(input_size=258, hidden_size=64, num_classes=len(gestures)).to(device)
    else:
        X_tensor = torch.tensor(reshape_for_stgcn(X_raw), dtype=torch.float32)
        if MODEL_TYPE == "STGCN":
            model = STGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "CTRGCN":
            model = CTRGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "SKATEFORMER":
            model = SkateFormerModel(num_classes=len(gestures)).to(device)
        else:
            print("Model Not Found !")
            exit()

    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    y_one_hot = F.one_hot(y_tensor).float()
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_one_hot, test_size=0.05)

    batch_size = 16  # Adjust this (16 or 32) based on your memory
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    train(model, MODEL_TYPE, train_loader, test_loader, device)
