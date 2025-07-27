import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime

    # Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

# Function to calculate ergonomic risk score
def calculate_risk_score(landmarks):
    # Extract key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Calculate shoulder and hip alignment
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    hip_diff = abs(left_hip.y - right_hip.y)
    spine_alignment = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)

    # Calculate knee angles
    left_knee_angle = calculate_angle([left_hip.x, left_hip.y], [left_knee.x, left_knee.y], [left_ankle.x, left_ankle.y])
    right_knee_angle = calculate_angle([right_hip.x, right_hip.y], [right_knee.x, right_knee.y], [right_ankle.x, right_ankle.y])

    # Risk score calculation
    risk_score = 0
    if shoulder_diff > 0.1 or hip_diff > 0.1:
        risk_score += 1  # Poor alignment
    if spine_alignment > 0.2:
        risk_score += 1  # Poor spine posture
    if left_knee_angle < 160 or right_knee_angle < 160:
        risk_score += 1  # Poor knee posture

    # Determine risk level
    if risk_score >= 2:
        return "High Risk", risk_score
    elif risk_score == 1:
        return "Medium Risk", risk_score
    else:
        return "Low Risk", risk_score

# Function to log data
def log_data(person_id, risk_level, risk_score, timestamp):
    data = {
        "Person ID": person_id,
        "Risk Level": risk_level,
        "Risk Score": risk_score,
        "Timestamp": timestamp
    }
    df = pd.DataFrame([data])
    df.to_csv("posture_data.csv", mode="a", header=not pd.io.common.file_exists("posture_data.csv"), index=False)

# Main function
def main():
    cap = cv2.VideoCapture(0)
    person_id = 0  # Track multiple people

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the frame
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate risk score
            risk_level, risk_score = calculate_risk_score(results.pose_landmarks.landmark)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Display risk level and score
            cv2.putText(image, f"Person {person_id}: {risk_level}, Score: {risk_score}", (10, 30 + person_id * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Log data
            log_data(person_id, risk_level, risk_score, timestamp)

            person_id += 1  # Increment person ID for next person

        # Reset person ID for next frame
        person_id = 0

        # Display the frame
        cv2.imshow('Advanced Posture Training', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()