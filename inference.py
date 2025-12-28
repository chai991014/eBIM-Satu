import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import mediapipe as mp
from model import CustomLSTM, STGCNModel, CTRGCNModel, SkateFormerModel
from training import get_adjacency_matrix, reshape_for_stgcn
from preprocess import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# ==========================================
# 1. Configuration & Model Loading
# ==========================================

# MODEL_TYPE = "LSTM"
# MODEL_TYPE = "STGCN"
# MODEL_TYPE = "CTRGCN"
MODEL_TYPE = "SKATEFORMER"

MODEL_PATH = f'./model/{MODEL_TYPE}/best_model.pth'
gloss_PATH = 'datasets_npy/gloss.npy'
# VIDEO_FILE = 'BIM_Dataset_V3/ambil/ambil_07_05_01.mp4'
VIDEO_FILE = 0  # Real-time webcam inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gloss = np.load(gloss_PATH)

# Initialize architecture
if MODEL_TYPE == "LSTM":
    model = CustomLSTM(input_size=258, hidden_size=128, num_classes=len(gloss)).to(device)
elif MODEL_TYPE == "STGCN":
    model = STGCNModel(num_classes=len(gloss), adjacency_matrix=get_adjacency_matrix()).to(device)
elif MODEL_TYPE == "CTRGCN":
    model = CTRGCNModel(num_classes=len(gloss), adjacency_matrix=get_adjacency_matrix()).to(device)
elif MODEL_TYPE == "SKATEFORMER":
    model = SkateFormerModel(num_classes=len(gloss)).to(device)
else:
    print("Model Not Found !")
    exit()

# Load weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Successfully loaded {MODEL_TYPE} model")
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()

# ==========================================
# 2. Real-time Inference Loop
# ==========================================

sequence = []
sentence = []
predictions = []
threshold = 0.5  # Confidence threshold for displaying results

# Replace with 0 for webcam or provide a video path
cap = cv2.VideoCapture(VIDEO_FILE)

# Set mediapipe model
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Extraction and Sequence logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Maintain last 30 frames

        # Prediction Logic
        if len(sequence) == 30:
            # Prepare data based on model type
            if MODEL_TYPE in ["STGCN", "CTRGCN", "SKATEFORMER"]:
                # Convert list to array and add batch dim: (1, 30, 258)
                input_data = np.expand_dims(sequence, axis=0)
                # Transform to (1, 3, 30, 75)
                input_processed = reshape_for_stgcn(input_data)
            else:
                input_processed = np.expand_dims(sequence, axis=0)

            # Convert to tensor and move to device
            input_tensor = torch.tensor(input_processed, dtype=torch.float32).to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    res = model(input_tensor)
                # Get probabilities via Softmax
                prob = F.softmax(res, dim=1)
                max_val, max_idx = torch.max(prob, dim=1)

            # Output results
            confidence = max_val.item()
            predicted_idx = max_idx.item()

            if confidence > threshold:
                label = gloss[predicted_idx] if predicted_idx < len(gloss) else "Unknown"

                # Visual feedback on image
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f'{label} ({confidence:.2f})', (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Sign Language Recognition', image)

        # Break gracefully
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
