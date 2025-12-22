import cv2
import numpy as np
import os
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


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)        # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)   # Draw left connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right connections


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


def augment_image(image, angle=0, scale=1.0):
    """
    Rotates and scales an image around its center.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # getRotationMatrix2D handles both rotation and scaling
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply the transformation
    augmented_image = cv2.warpAffine(image, M, (w, h))
    return augmented_image


def detect_hand_landmarks(image_path):
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make detections
        image, results = mediapipe_detection(image, holistic)

        # Draw landmarks to the frame
        draw_styled_landmarks(image, results)

        # Convert the image back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image with hand landmarks
        cv2.imshow('Landmark Detection Verification', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video_frame(gestures, train_dataset_path):
    for ges in gestures:

        # Specify the video path
        data_path = os.path.join(video_directory, ges)
        data_video = os.listdir(data_path)

        for vid in data_video:

            if not os.path.exists(os.path.join(train_dataset_path)):
                os.makedirs(train_dataset_path)

            landmark_path = os.path.join(train_dataset_path, ges, 'landmarks' + vid)
            video_path = os.path.join(video_directory, ges, vid)
            print(video_path)

            for aug in AUG:

                # Update folder name to distinguish between original, rotated, and scaled
                # e.g., 'landmarks_rot10_video_name'
                folder_name = 'landmarks_' + aug['name'] + '_' + vid
                landmark_path = os.path.join(train_dataset_path, ges, folder_name)

                # Locate the video dataset (Must re-open video for each augmentation pass)
                video = cv2.VideoCapture(video_path)
                all_video_frames = []

                count = 0
                frame_count = 0

                # Set mediapipe model
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    while True:
                        ret, frame = video.read()

                        if not ret:
                            # Break the loop if there are no more frames
                            break

                        if aug['name'] != 'original':
                            frame = augment_image(frame, angle=aug['angle'], scale=aug['scale'])

                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks to the frame
                        draw_styled_landmarks(image, results)

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


def compose_train_data(gestures, train_dataset_path):
    label_map = {label: num for num, label in enumerate(gestures)}

    gesture_sequence, labels = [], []

    for gs in gestures:
        gesture = []

        for fname in os.listdir(os.path.join(train_dataset_path, gs)):
            path = os.path.join(train_dataset_path, gs, fname)
            if os.path.isdir(path):
                gesture.append(fname)

        for no in gesture:
            load_path = os.path.join(train_dataset_path, gs, no)
            npy_files = sorted([f for f in os.listdir(load_path) if f.endswith('.npy')])
            npy_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            video = []
            for npy in npy_files:
                video.append(np.load(os.path.join(load_path, npy)))
                print(os.path.join(load_path, npy))

            gesture_sequence.append(video)
            labels.append(label_map[gs])

        # Calculate the maximum sequence length for this gesture
        max_len = max(len(seq) for seq in gesture_sequence)

        # Pad shorter sequences with zeros to match the maximum length
        gesture_sequence = [
            seq + [[0] * seq[0].shape[0]] * (max_len - len(seq))  # Pad with zeros
            for seq in gesture_sequence
        ]

        print(np.array(gesture_sequence).shape, np.array(labels).shape)

    X = np.array(gesture_sequence)
    y = np.array(labels)

    np.save('X_TRAIN_02.npy', X)
    np.save('y_TRAIN_02.npy', y)

    np.save('gestures_02.npy', gestures)


if __name__ == "__main__":

    # Specify the path you upload the videos in the Google Drive
    video_directory = 'BIM_Dataset_V3'

    # Get all file names in the directory
    gestures_files = os.listdir(video_directory)

    # Specify the gestures
    gestures = np.array(gestures_files)
    # print(gestures)

    # Specify your path to store landmarks files
    train_dataset_path = 'datasets'

    AUG = [{'name': 'original', 'angle': 0, 'scale': 1.0},
           {'name': 'rotcw10', 'angle': -10, 'scale': 1.0},
           {'name': 'rotacw10', 'angle': 10, 'scale': 1.0},
           {'name': 'scaleup', 'angle': 0, 'scale': 1.2},
           {'name': 'scaledown', 'angle': 0, 'scale': 0.8}]

    process_video_frame(gestures, train_dataset_path)

    compose_train_data(gestures, train_dataset_path)
