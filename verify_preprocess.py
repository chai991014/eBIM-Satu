import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def draw_skeletal_lines_from_array(frame, landmarks):
    """
    Draws skeletal lines and landmarks from the flattened 258-feature array
    using the exact styling (colors and sizes) from the training notebook.
    """
    height, width, _ = frame.shape

    # Colors from the notebook
    CONNECTION_COLOR = (80, 110, 10)  # Greenish lines
    LANDMARK_COLOR = (0, 0, 255)  # Red dots

    # Helper to get (x, y) pixels from flattened indices
    def get_coords(idx, type='pose'):
        if type == 'pose':
            base = idx * 4
        elif type == 'lh':
            base = 132 + (idx * 3)
        else:
            base = 195 + (idx * 3)

        x = int(landmarks[base] * width)
        y = int((landmarks[base + 1]) * height)
        return x, y

    # 1. Draw Pose Connections & Landmarks (Radius 1, Thickness 1)
    for connection in mp_holistic.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(frame, get_coords(start_idx, 'pose'), get_coords(end_idx, 'pose'),
                 CONNECTION_COLOR, 1)

    for i in range(33):
        cv2.circle(frame, get_coords(i, 'pose'), 1, LANDMARK_COLOR, -1)

    # 2. Draw Hand Connections & Landmarks (Radius 2, Thickness 1)
    for hand_type in ['lh', 'rh']:
        for connection in mp_holistic.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame, get_coords(start_idx, hand_type), get_coords(end_idx, hand_type),
                     CONNECTION_COLOR, 1)

        for i in range(21):
            cv2.circle(frame, get_coords(i, hand_type), 2, LANDMARK_COLOR, -1)


def verify_landmarks(video_path, landmark_folder, output_path):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Logic: Sync 30 .npy files across the total video length
    target_indices = np.linspace(0, total_video_frames - 1, 30).astype(int)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (frame_width, frame_height))

    print(f"Verifying 30 .npy files against {total_video_frames} video frames...")

    for i, frame_pos in enumerate(target_indices):
        # Jump to the specific frame that was used for this .npy file
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()

        if not ret:
            break

        # Path for files 0.npy through 29.npy
        npy_path = os.path.join(landmark_folder, f"{i}.npy")

        if os.path.exists(npy_path):
            try:
                landmarks = np.load(npy_path)

                # Verify the baseline 258 features
                if landmarks.shape[0] == 258:
                    # draw_landmarks_from_array(frame, landmarks)
                    draw_skeletal_lines_from_array(frame, landmarks)
                    status_color = (0, 255, 0)
                    msg = f"OK: {i}.npy"
                else:
                    status_color = (0, 0, 255)
                    msg = "INVALID SHAPE"
            except Exception as e:
                status_color = (0, 0, 255)
                msg = "LOAD ERROR"
        else:
            status_color = (0, 0, 255)
            msg = f"MISSING {i}.npy"

        # Overlays
        cv2.putText(frame, f"NPY Index: {i} | Frame: {frame_pos}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        out.write(frame)
        cv2.imshow('Landmark Verification', frame)

        # Press 'q' to stop
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break

    print(f"Verification finished. Saved to: {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    gesture = "ambil"
    code = "07_05_01"
    style = "original"
    # style = "rotcw10"
    # style = "rotacw10"
    # style = "scaleup"
    # style = "scaledown"
    VIDEO_FILE = f'BIM_Dataset_V3/{gesture}/{gesture}_{code}.mp4'
    LANDMARK_DIR = f'datasets/{gesture}/landmarks_{style}_{gesture}_{code}.mp4'

    # verify_landmarks(VIDEO_FILE, LANDMARK_DIR, f"landmarks_{style}_{gesture}_{code}.mp4")
