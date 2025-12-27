import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import CustomLSTM, STGCNModel, CTRGCNModel, SkateFormerModel
from training import get_adjacency_matrix, reshape_for_stgcn


def evaluate(model, test_loader, gestures, model_type, model_path):
    # Load the specific weights provided from the main loop
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=device))

    save_dir = f'./model/{model_type}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_preds = []
    all_trues = []
    val_loss = 0.0
    correct_test = 0

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(b_x)
                val_loss += criterion(outputs, b_y).item() * b_x.size(0)
                correct_test += (outputs.argmax(1) == b_y).sum().item()
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_trues.append(b_y.cpu().numpy())

        test_loss = val_loss / len(test_loader.dataset)
        accuracy = correct_test / len(test_loader.dataset)

        print(f'\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

        # Convert to numpy for sklearn/seaborn
        predicted_labels = np.concatenate(all_preds)
        true_labels = np.concatenate(all_trues)

        class_report_dict = classification_report(true_labels, predicted_labels, target_names=gestures, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(class_report_dict).transpose()
        report_df.to_csv(f'{save_dir}classification_report.csv')

        # 3. PRINT ONLY TOP/BOTTOM SUMMARY TO CONSOLE
        print(f"\nOverall Accuracy: {class_report_dict['accuracy']:.4f}")
        print(f"Top 5 Classes (by F1):")
        print(report_df.sort_values(by='f1-score', ascending=False).head(5))

        # 1. Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=gestures, columns=gestures)
        conf_matrix_df.to_csv(f'{save_dir}confusion_matrix.csv')

        plt.figure(figsize=(30, 25))
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
        plt.title(f'Confusion Matrix - {model_type}', fontsize=20)
        plt.xlabel('Predicted', fontsize=15)
        plt.ylabel('True', fontsize=15)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{save_dir}confusion_matrix_high_res.png', dpi=300)
        plt.show()
        plt.close()


def summary():
    model_dir = './model'
    if os.path.exists(model_dir):
        # Get all entries in the directory that are folders
        model_types = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    else:
        model_types = []
    summary_data = []

    for mt in model_types:
        file_path = f'./model/{mt}/classification_report.csv'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)

            # Extracting the key metrics from the transposed report rows
            # 'accuracy' row contains the overall score in all metric columns
            acc = df.loc['accuracy', 'f1-score'] if 'accuracy' in df.index else "N/A"
            macro_precision = df.loc['macro avg', 'precision'] if 'macro avg' in df.index else "N/A"
            macro_recall = df.loc['macro avg', 'recall'] if 'macro avg' in df.index else "N/A"
            macro_f1 = df.loc['macro avg', 'f1-score'] if 'macro avg' in df.index else "N/A"

            summary_data.append({
                'Model Type': mt,
                'Accuracy': acc,
                'Macro Precision': macro_precision,
                'Macro Recall': macro_recall,
                'Macro F1': macro_f1
            })
            print(f"Successfully processed {mt}")
        else:
            print(f"Warning: {file_path} not found. Skipping...")

    if summary_data:
        # Create and save the final comparison table
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('./model/summary.csv', index=False)
        print("\nSummary saved to: ./model/summary.csv")
        print(summary_df)
    else:
        print("No report files were found to aggregate.")


if __name__ == "__main__":

    # MODEL_TYPE = "LSTM"
    # MODEL_TYPE = "STGCN"
    # MODEL_TYPE = "CTRGCN"
    MODEL_TYPE = "SKATEFORMER"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "datasets_npy"
    model_dir = f'./model/{MODEL_TYPE}/'
    MODEL_PATH = f'{model_dir}best_model.pth'

    X_test_raw = np.load(f'{data_dir}/X_test.npy')
    y_test_raw = np.load(f'{data_dir}/y_test.npy')
    gestures = np.load(f'{data_dir}/gestures.npy')

    if MODEL_TYPE == "LSTM":
        X_test = torch.tensor(X_test_raw, dtype=torch.float32).to(device)
        model = CustomLSTM(input_size=258, hidden_size=128, num_classes=len(gestures)).to(device)
    else:
        X_test = torch.tensor(reshape_for_stgcn(X_test_raw), dtype=torch.float32).to(device)
        if MODEL_TYPE == "STGCN":
            model = STGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "CTRGCN":
            model = CTRGCNModel(num_classes=len(gestures), adjacency_matrix=get_adjacency_matrix()).to(device)
        elif MODEL_TYPE == "SKATEFORMER":
            model = SkateFormerModel(num_classes=len(gestures)).to(device)
        else:
            print("Model Not Found !")
            exit()

    y_test = torch.tensor(y_test_raw, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

    evaluate(model, test_loader, gestures, MODEL_TYPE, MODEL_PATH)

    # summary()
