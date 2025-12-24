import cv2
import numpy as np
import os
import mediapipe as mp
from preprocess import rotate_skeleton, scale_skeleton, apply_skeletal_shrink, adjust_shoulder_width, adjust_arm_length, adjust_torso_height, mix_skeletal_parts

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
    POSE_LABEL_COLOR = (255, 255, 255)  # White
    HAND_LABEL_COLOR = (0, 255, 255)  # Yellow

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
    for i in range(33):
        coords = get_coords(i, 'pose')
        cv2.circle(frame, get_coords(i, 'pose'), 1, LANDMARK_COLOR, -1)
        # DEBUG: Put Pose index (0-32)
        cv2.putText(frame, str(i), (coords[0] + 2, coords[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, POSE_LABEL_COLOR, 1)

    for connection in mp_holistic.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(frame, get_coords(start_idx, 'pose'), get_coords(end_idx, 'pose'),
                 CONNECTION_COLOR, 1)

    # 2. Draw Hand Connections & Landmarks (Radius 2, Thickness 1)
    for hand_type in ['lh', 'rh']:
        for i in range(21):
            coords = get_coords(i, hand_type)
            cv2.circle(frame, coords, 2, LANDMARK_COLOR, -1)
            # DEBUG: Put Absolute Index (e.g., 132, 135...)
            abs_idx = 132 + (i * 3) if hand_type == 'lh' else 195 + (i * 3)
            cv2.putText(frame, str(abs_idx), (coords[0] + 2, coords[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, HAND_LABEL_COLOR, 1)

        for connection in mp_holistic.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame, get_coords(start_idx, hand_type), get_coords(end_idx, hand_type),
                     CONNECTION_COLOR, 1)


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


def verify_augmentation_video(video_path, landmark_folder, output_path, aug_style=None, partner_folder=None):
    """
    Loads 30 frames, applies augmentations, and saves a verification video.
    """
    # 1. Setup Video Capture and sync frames
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = np.linspace(0, total_video_frames - 1, 30).astype(int)

    # 2. Load and Augment the Sequence (30, 258)
    npy_files = sorted([f for f in os.listdir(landmark_folder) if f.endswith('.npy')],
                       key=lambda f: int(''.join(filter(str.isdigit, f))))
    sequence = np.array([np.load(os.path.join(landmark_folder, f)) for f in npy_files])

    aug_sequence = np.copy(sequence)
    # Apply heavy augmentations to make the difference visible on screen
    if partner_folder:
        aug_style = None

    if aug_style == "rotate":
        aug_sequence = rotate_skeleton(aug_sequence)
    elif aug_style == "scale":
        aug_sequence = scale_skeleton(aug_sequence)
    elif aug_style == "child":
        aug_sequence = apply_skeletal_shrink(aug_sequence)
    elif aug_style == "shoulder":
        aug_sequence = adjust_shoulder_width(aug_sequence)
    elif aug_style == "arm":
        aug_sequence = adjust_arm_length(aug_sequence)
    elif aug_style == "height":
        aug_sequence = adjust_torso_height(aug_sequence)
    elif partner_folder:
        None
    else:
        exit()

    if partner_folder:
        partner_files = sorted([f for f in os.listdir(partner_folder) if f.endswith('.npy')],
                               key=lambda f: int(''.join(filter(str.isdigit, f))))
        partner_seq = np.array([np.load(os.path.join(partner_folder, f)) for f in partner_files])
        aug_sequence = mix_skeletal_parts(aug_sequence, partner_seq)

    # 3. Create Video (Black Background)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (int(cap.get(3)), int(cap.get(4))))

    for i, frame_pos in enumerate(target_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret: break

        # Draw the augmented skeleton on the real video frame
        draw_skeletal_lines_from_array(frame, aug_sequence[i])

        # Add labels to identify what we are seeing
        cv2.putText(frame, "REAL VIDEO + AUGMENTED SKELETON", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow('Overlay Verification', frame)
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Verification video saved: {output_path}")


if __name__ == "__main__":

    gesture = "abang"
    code = "1_1_1"
    code2 = "2_1_1"
    # AUG_STYLE = "rotate"
    # AUG_STYLE = "scale"
    # AUG_STYLE = "child"
    # AUG_STYLE = "shoulder"
    # AUG_STYLE = "arm"
    # AUG_STYLE = "height"
    VIDEO_FILE = f'BIM_Dataset_V3/{gesture}/{gesture}_{code}.mp4'
    LANDMARK_DIR = f'datasets/{gesture}/landmarks_{gesture}_{code}.mp4'
    PARTNER_DIR = f'datasets/{gesture}/landmarks_{gesture}_{code2}.mp4'  # Optional for Part Mixing

    # verify_landmarks(VIDEO_FILE, LANDMARK_DIR, f"landmarks_{gesture}_{code}.mp4")

    if code2:
        verify_augmentation_video(VIDEO_FILE, LANDMARK_DIR, f"landmarks_{gesture}_{code}_MIX_{code2}.mp4", partner_folder=PARTNER_DIR)
    else:
        verify_augmentation_video(VIDEO_FILE, LANDMARK_DIR, f"landmarks_{gesture}_{code}_AUG_{AUG_STYLE}.mp4", aug_style=AUG_STYLE)
