import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mediapipe as mp
from model import CustomLSTM, STGCNModel, CTRGCNModel, SkateFormerModel
from utils import setup_logger


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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, delta=0.001, start_epoch=50, path='checkpoint.pth'):
        self.patience = patience
        self.start_epoch = start_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if epoch < self.start_epoch:
            if val_loss < self.val_loss_min:
                self.save_checkpoint(val_loss, model, epoch)
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch + 1


def train(model, model_type, train_loader, val_loader, device, num_epochs=200):

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)

    # Label smoothing=0.1 helps the model learn that signs are related
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # OneCycleLR: Provides a 'warmup' phase and a 'cool down' phase.
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=num_epochs)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    model_dir = f'./model/{model_type}/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file_path = setup_logger(model_dir, "training_session")
    csv_log_path = os.path.join(model_dir, "metrics_log.csv")
    with open(csv_log_path, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    final_best_path = f'{model_dir}best_model.pth'

    early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA, start_epoch=WARM_UP, path=final_best_path)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Start training for {model_type}...")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (batch_X, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Scheduler for OneCycleLR
            scheduler.step()

            train_loss += loss.item()
            correct_train += (outputs.argmax(dim=1) == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_test = 0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                with torch.cuda.amp.autocast():
                    out = model(b_x)
                    val_loss += criterion(out, b_y).item() * b_x.size(0)
                    correct_test += (out.argmax(1) == b_y).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_test / len(val_loader.dataset)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        metrics = [epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc]

        with open(csv_log_path, "a") as f:
            f.write(",".join([f"{m:.4f}" for m in metrics]) + "\n")

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {metrics[1]:.4f} | Train Acc: {metrics[2]:.4f} | Val Loss: {metrics[3]:.4f} | Val Acc: {metrics[4]:.4f}")

        # Scheduler for MultiStepLR
        # scheduler.step()

        early_stopping(epoch_val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch + 1}")
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Plotting code remains same as LSTM.py
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_type} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title(f'{model_type} Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"{model_dir}learning_curves.png"
    plt.savefig(save_path)

    plt.show()

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY: {model_type}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Epoch Saved: {early_stopping.best_epoch}")
    print(f"Model Path: {early_stopping.path}")
    print(f"CSV Metrics: {csv_log_path}")
    print(f"Text Log: {log_file_path}")
    print("=" * 60)


if __name__ == "__main__":

    # MODEL_TYPE = "LSTM"
    # MODEL_TYPE = "STGCN"
    # MODEL_TYPE = "CTRGCN"
    MODEL_TYPE = "SKATEFORMER"

    # SETTING
    LR = 1e-3
    WEIGHT_DECAY = 0.05
    PATIENCE = 20
    DELTA = 0.001
    WARM_UP = 50
    EPOCH = 200
    BATCH_SIZE = 32

    # SkateFormer SETTING
    # LR = 5e-4

    # STGCN SGD SETTING (with MultiStepLR)
    # LR = 0.1
    # WEIGHT_DECAY = 1e-4
    # PATIENCE = 40

    data_dir = "datasets_npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_raw = np.load(f'{data_dir}/X_train.npy')
    y_train_raw = np.load(f'{data_dir}/y_train.npy')
    X_val_raw = np.load(f'{data_dir}/X_val.npy')
    y_val_raw = np.load(f'{data_dir}/y_val.npy')
    gestures = np.load(f'{data_dir}/gestures.npy')

    if MODEL_TYPE == "LSTM":
        X_train = torch.tensor(X_train_raw, dtype=torch.float32)
        X_val = torch.tensor(X_val_raw, dtype=torch.float32)
        model = CustomLSTM(input_size=258, hidden_size=128, num_classes=len(gestures)).to(device)
    else:
        X_train = torch.tensor(reshape_for_stgcn(X_train_raw), dtype=torch.float32)
        X_val = torch.tensor(reshape_for_stgcn(X_val_raw), dtype=torch.float32)
        if MODEL_TYPE == "STGCN":
            model = STGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "CTRGCN":
            model = CTRGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "SKATEFORMER":
            model = SkateFormerModel(num_classes=len(gestures)).to(device)
        else:
            print("Model Not Found !")
            exit()

    y_train = torch.tensor(y_train_raw, dtype=torch.long)
    y_val = torch.tensor(y_val_raw, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8)

    train(model, MODEL_TYPE, train_loader, val_loader, device, num_epochs=EPOCH)
