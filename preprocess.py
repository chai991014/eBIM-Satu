import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion from BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is no longer writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion RGB to BGR
    return image, results


def draw_styled_landmarks(image,results):

    # Draw pose connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
                              )
    # Draw left hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
                              )
    # Draw right hand connection
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


def apply_skeletal_shrink(data, shrink_range=(0.20, 0.35)):
    """
    Shrinks the entire skeleton (Pose + Hands) towards the center to perfectly simulate child proportions.
    """
    aug_data = np.copy(data)
    # Randomly decide how much to 'shrink' for this specific sample
    shrink_factor = np.random.uniform(shrink_range[0], shrink_range[1])

    for f in range(aug_data.shape[0]):
        # 1. Calculate the 'Pivot Y' (Shoulder height) for this frame
        # Shoulder indices: 11 (left), 12 (right). Y is the 2nd value (idx*4 + 1)
        sh_y_avg = (aug_data[f, 11 * 4 + 1] + aug_data[f, 12 * 4 + 1]) / 2

        # 2. Unified Shrink Logic: Move all X towards 0.5, all Y towards sh_y_avg

        # Part A: Pose Landmarks (33 joints, 4 values each: x,y,z,v)
        for i in range(33):
            x_idx, y_idx = i * 4, i * 4 + 1
            aug_data[f, x_idx] = aug_data[f, x_idx] + (0.5 - aug_data[f, x_idx]) * shrink_factor
            aug_data[f, y_idx] = aug_data[f, y_idx] - (aug_data[f, y_idx] - sh_y_avg) * shrink_factor

        # Part B: Hand Landmarks (42 joints total, 3 values each: x,y,z)
        # Left Hand (132-194) and Right Hand (195-257)
        for i in range(132, 258, 3):
            x_idx, y_idx = i, i + 1
            aug_data[f, x_idx] = aug_data[f, x_idx] + (0.5 - aug_data[f, x_idx]) * shrink_factor
            aug_data[f, y_idx] = aug_data[f, y_idx] - (aug_data[f, y_idx] - sh_y_avg) * shrink_factor

    return aug_data


def adjust_shoulder_width(data, intensity=0.04):
    """Simulates broader or narrower shoulders."""
    aug_data = np.copy(data)
    # Randomly decide width change for this sample
    # shift_x = np.random.uniform(-intensity, intensity)
    shift_x = - intensity
    for f in range(aug_data.shape[0]):
        # Left Shoulder (11) moves left/right
        aug_data[f, 11 * 4] -= shift_x
        # Right Shoulder (12) moves opposite direction
        aug_data[f, 12 * 4] += shift_x
    return aug_data


def adjust_arm_length(data, intensity=0.20):
    """Simulates longer or shorter humerus/forearm."""
    aug_data = np.copy(data)
    # Extension factor: >1 is longer, <1 is shorter
    ext_factor = np.random.uniform(1.0 - intensity, 1.0 + intensity)

    for f in range(data.shape[0]):
        # 1. Update Pose Elbows (13, 14) relative to Shoulders
        for s, e in [(11, 13), (12, 14)]:
            aug_data[f, e * 4] = data[f, s * 4] + (data[f, e * 4] - data[f, s * 4]) * ext_factor
            aug_data[f, e * 4 + 1] = data[f, s * 4 + 1] + (data[f, e * 4 + 1] - data[f, s * 4 + 1]) * ext_factor

        # 2. Update ALL Pose hand-end landmarks (15-22) relative to the NEW Elbows
        # This fixes the floating 18, 20, 22 you see in your debug video.
        left_hand_pose = [15, 17, 19, 21]
        right_hand_pose = [16, 18, 20, 22]

        # Left Side
        new_el_l_x, new_el_l_y = aug_data[f, 13 * 4], aug_data[f, 13 * 4 + 1]
        for p_idx in left_hand_pose:
            orig_vec_x = data[f, p_idx * 4] - data[f, 13 * 4]
            orig_vec_y = data[f, p_idx * 4 + 1] - data[f, 13 * 4 + 1]
            aug_data[f, p_idx * 4] = new_el_l_x + (orig_vec_x * ext_factor)
            aug_data[f, p_idx * 4 + 1] = new_el_l_y + (orig_vec_y * ext_factor)

        # Right Side
        new_el_r_x, new_el_r_y = aug_data[f, 14 * 4], aug_data[f, 14 * 4 + 1]
        for p_idx in right_hand_pose:
            orig_vx = data[f, p_idx * 4] - data[f, 14 * 4]
            orig_vy = data[f, p_idx * 4 + 1] - data[f, 14 * 4 + 1]
            aug_data[f, p_idx * 4] = new_el_r_x + (orig_vx * ext_factor)
            aug_data[f, p_idx * 4 + 1] = new_el_r_y + (orig_vy * ext_factor)

        # 3. SNAP ACTUAL HAND DATA TO THE NEW POSE WRIST
        # Now that Pose 15 and 16 are moved correctly, the hand landmarks will follow.
        l_shift_x = aug_data[f, 15 * 4] - data[f, 132]
        l_shift_y = aug_data[f, 15 * 4 + 1] - data[f, 133]
        for j in range(21):
            aug_data[f, 132 + (j * 3)] = data[f, 132 + (j * 3)] + l_shift_x
            aug_data[f, 132 + (j * 3) + 1] = data[f, 133 + (j * 3)] + l_shift_y

        r_shift_x = aug_data[f, 16 * 4] - data[f, 195]
        r_shift_y = aug_data[f, 16 * 4 + 1] - data[f, 196]
        for j in range(21):
            aug_data[f, 195 + (j * 3)] = data[f, 195 + (j * 3)] + r_shift_x
            aug_data[f, 195 + (j * 3) + 1] = data[f, 196 + (j * 3)] + r_shift_y

    return aug_data


def adjust_torso_height(data, intensity=0.05):
    """Simulates a longer or shorter upper body."""
    aug_data = np.copy(data)
    # shift_y = np.random.uniform(-intensity, intensity)
    shift_y = - intensity
    for f in range(aug_data.shape[0]):
        # Move Shoulders, Elbows, and Wrists as a block relative to the hips
        for i in range(11, 23):
            aug_data[f, i * 4 + 1] += shift_y

        # Snap Hands
        l_gap_x, l_gap_y = aug_data[f, 60] - aug_data[f, 132], aug_data[f, 61] - aug_data[f, 133]
        r_gap_x, r_gap_y = aug_data[f, 64] - aug_data[f, 195], aug_data[f, 65] - aug_data[f, 196]
        for j in range(21):
            aug_data[f, 132 + (j * 3)] += l_gap_x;
            aug_data[f, 132 + (j * 3) + 1] += l_gap_y
            aug_data[f, 195 + (j * 3)] += r_gap_x;
            aug_data[f, 195 + (j * 3) + 1] += r_gap_y
    return aug_data


def rotate_skeleton(data, angle_range=(-10, 10)):
    """
    Rotates the x,y coordinates around the center (0.5, 0.5).
    """
    angle = np.radians(np.random.uniform(angle_range[0], angle_range[1]))
    cos_val, sin_val = np.cos(angle), np.sin(angle)

    aug_data = np.copy(data)
    # Reshape to (frames, num_keypoints, coordinates)
    # Your keypoints have x, y, z (and visibility for pose)
    for f in range(aug_data.shape[0]):
        # Pose (33 joints, first 132 values: x,y,z,v)
        for i in range(33):
            x, y = aug_data[f, i * 4] - 0.5, aug_data[f, i * 4 + 1] - 0.5
            aug_data[f, i * 4] = x * cos_val - y * sin_val + 0.5
            aug_data[f, i * 4 + 1] = x * sin_val + y * cos_val + 0.5

        # Left Hand (21 joints, 132-194: x,y,z) & Right Hand (195-257: x,y,z)
        for i in range(132, 258, 3):
            x, y = aug_data[f, i] - 0.5, aug_data[f, i + 1] - 0.5
            aug_data[f, i] = x * cos_val - y * sin_val + 0.5
            aug_data[f, i + 1] = x * sin_val + y * cos_val + 0.5
    return aug_data


def scale_skeleton(data, scale_range=(0.8, 1.2)):
    """
    Scales the skeleton to simulate signer being closer or further.
    """
    # scale = np.random.uniform(scale_range[0], scale_range[1])
    scale = scale_range[1]
    aug_data = np.copy(data)

    # Scale x and y relative to center 0.5
    # Pose x,y
    for i in range(33):
        aug_data[:, i * 4] = (aug_data[:, i * 4] - 0.5) * scale + 0.5
        aug_data[:, i * 4 + 1] = (aug_data[:, i * 4 + 1] - 0.5) * scale + 0.5

    # Hands x,y
    for i in range(132, 258, 3):
        aug_data[:, i] = (aug_data[:, i] - 0.5) * scale + 0.5
        aug_data[:, i + 1] = (aug_data[:, i + 1] - 0.5) * scale + 0.5

    return aug_data


def mix_skeletal_parts(seq_a, seq_b):
    """
    Swaps hand landmarks of seq_b into seq_a.
    Pose: 0-131 | Hands: 132-257
    """
    mixed_seq = np.copy(seq_a)
    mixed_seq[:, 132:] = seq_b[:, 132:]
    return mixed_seq


def process_video_frame(gloss, video_directory, train_dataset_path):
    """
    REMOVED: Internal AUG loop and image-level rotation/scaling.
    Now only processes the 'original' video once to save time.
    """
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for ges in gloss:
            # Specify the video path
            data_path = os.path.join(video_directory, ges)
            data_video = os.listdir(data_path)

            for vid in data_video:

                if not os.path.exists(os.path.join(train_dataset_path)):
                    os.makedirs(train_dataset_path)

                video_path = os.path.join(video_directory, ges, vid)
                print(video_path)
                # We only save the 'original' landmarks now. Augmentation happens in compose_train_data.
                landmark_path = os.path.join(train_dataset_path, ges, 'landmarks_' + vid)

                if os.path.exists(landmark_path):
                    continue  # Skip if already processed

                video = cv2.VideoCapture(video_path)
                all_video_frames = []
                while True:
                    ret, frame = video.read()

                    if not ret:
                        # Break the loop if there are no more frames
                        break
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Extract keypoints for every frame to preserve timing
                    keypoints = extract_keypoints(results)
                    all_video_frames.append(keypoints)

                video.release()

                # Resampling Logic: Scale any video length to exactly 30 frames
                total_captured = len(all_video_frames)
                if total_captured > 0:
                    if not os.path.exists(landmark_path):
                        os.makedirs(landmark_path)

                    # Pick 30 indices evenly spaced across the entire video
                    indices = np.linspace(0, total_captured - 1, 30).astype(int)

                    for i, idx in enumerate(indices):
                        res = all_video_frames[idx]
                        npy_path = os.path.join(landmark_path, str(i))
                        np.save(npy_path, res)


def compose_train_data(gloss, train_dataset_path):
    """
    Includes Automatic Oversampling, Undersampling, Part Mixing, and Shoulder Noise.
    """
    label_map = {label: num for num, label in enumerate(gloss)}
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for gs in gloss:
        gloss_samples = []
        class_path = os.path.join(train_dataset_path, gs)

        # Load all available 'original' samples for this class
        folder_names = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]

        for fname in folder_names:
            load_path = os.path.join(class_path, fname)
            npy_files = sorted([f for f in os.listdir(load_path) if f.endswith('.npy')],
                               key=lambda f: int(''.join(filter(str.isdigit, f))))
            video = [np.load(os.path.join(load_path, npy)) for npy in npy_files]
            gloss_samples.append(np.array(video))

        num_orig = len(gloss_samples)
        np.random.shuffle(gloss_samples)

        if num_orig < 40:
            idx_train = int(num_orig * 0.5)
            idx_val = idx_train + max(2, int(num_orig * 0.2))
        else:
            idx_train = int(num_orig * 0.7)
            idx_val = idx_train + int(num_orig * 0.1)

        pool_train = gloss_samples[:idx_train]
        pool_val = gloss_samples[idx_train:idx_val]
        pool_test = gloss_samples[idx_val:]

        def augment_to_target(pool, target=100):
            out = [np.copy(p) for p in pool]
            if not pool:
                return out

            bias_correction_count = int(len(pool) * 0.2)
            effective_target = max(target, len(pool) + bias_correction_count)

            while len(out) < effective_target:
                idx = np.random.randint(0, len(pool))
                aug_sample = np.copy(pool[idx])
                # --- 1. ANATOMICAL DIVERSITY (Body Proportions) ---
                # Applied first to set the 'base' skeleton shape
                if np.random.random() < 0.7:
                    aug_sample = adjust_shoulder_width(aug_sample)
                if np.random.random() < 0.7:
                    aug_sample = adjust_arm_length(aug_sample)
                if np.random.random() < 0.5:  # Reduced slightly to prevent over-distortion
                    aug_sample = adjust_torso_height(aug_sample)

                # --- 2. AGE ADAPTATION (Global Scale) ---
                if np.random.random() < 0.2:
                    aug_sample = apply_skeletal_shrink(aug_sample)

                # --- 3. MOVEMENT STYLE (Part Mixing) ---
                # Swaps movements between two skeletons in the same pool
                if len(pool) < 30 and np.random.random() < 0.5:
                    partner_idx = np.random.randint(0, len(pool))
                    aug_sample = mix_skeletal_parts(aug_sample, pool[partner_idx])

                # --- 4. ENVIRONMENTAL NOISE (Camera Zoom/Tilt) ---
                # Applied last, so it rotates the final assembled skeleton
                if np.random.random() < 0.6:
                    aug_sample = rotate_skeleton(aug_sample)
                if np.random.random() < 0.6:
                    aug_sample = scale_skeleton(aug_sample)

                out.append(aug_sample)
            return out

        aug_train = augment_to_target(pool_train)
        X_train.extend(aug_train)
        y_train.extend([label_map[gs]] * len(aug_train))

        X_val.extend(pool_val)
        y_val.extend([label_map[gs]] * len(pool_val))

        X_test.extend(pool_test)
        y_test.extend([label_map[gs]] * len(pool_test))

        print(f"Class {gs}: Split created (Train:{len(aug_train)}, Val:{len(pool_val)}, Test:{len(pool_test)})")

    save_dir = "datasets_npy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'X_train.npy'), np.array(X_train, dtype=np.float32))
    np.save(os.path.join(save_dir, 'y_train.npy'), np.array(y_train, dtype=np.int32))

    np.save(os.path.join(save_dir, 'X_val.npy'), np.array(X_val, dtype=np.float32))
    np.save(os.path.join(save_dir, 'y_val.npy'), np.array(y_val, dtype=np.int32))

    np.save(os.path.join(save_dir, 'X_test.npy'), np.array(X_test, dtype=np.float32))
    np.save(os.path.join(save_dir, 'y_test.npy'), np.array(y_test, dtype=np.int32))

    np.save(os.path.join(save_dir, 'gloss.npy'), np.array(gloss))

    print(f"Dataset successfully saved to {save_dir}/")
    print(f"X_train shape: {np.array(X_train).shape} | Total Samples: {len(X_train)}")
    print(f"X_val shape: {np.array(X_val).shape} | Total Samples: {len(X_val)}")
    print(f"X_test shape: {np.array(X_test).shape} | Total Samples: {len(X_test)}")


if __name__ == "__main__":

    # Specify the path you upload the videos in the Google Drive
    video_directory = 'BIM_Dataset_V3'

    # Get all file names in the directory
    gloss_files = os.listdir(video_directory)

    # Specify the gloss
    gloss = np.array(gloss_files)
    # print(gloss)

    # Specify your path to store landmarks files
    train_dataset_path = 'datasets'

    # _stderr = sys.stderr
    # with open(os.devnull, 'w') as f:
    #     sys.stderr = f
    #     process_video_frame(gloss, video_directory, train_dataset_path)
    #     sys.stderr = _stderr

    compose_train_data(gloss, train_dataset_path)
