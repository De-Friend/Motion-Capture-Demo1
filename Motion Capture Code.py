import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def classify_body_posture(landmarks):
    shoulder_y_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    shoulder_y_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    shoulder_x_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    shoulder_x_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    ear_y = (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2
    ear_x = (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2

    angle_neck = np.degrees(np.arctan2(ear_y - ((shoulder_y_left + shoulder_y_right) / 2), 1))
    angle_hip = np.degrees(np.arctan2(((shoulder_y_left + shoulder_y_right) / 2) - hip_y, 1))

    forward_head_threshold = 0.05  # Adjust this threshold as necessary
    swayback_threshold = 0.05      # Adjust this threshold as necessary

    if abs(shoulder_y_left - shoulder_y_right) > 0.05:  # Adjust threshold as needed
        return "Shoulder Imbalance"
    elif abs(ear_x - ((shoulder_x_left + shoulder_x_right) / 2)) > forward_head_threshold:
        return "Forward-Head Posture"
    elif abs(angle_neck) < 10 and abs(angle_hip) < 10 and hip_y > ((shoulder_y_left + shoulder_y_right) / 2):
        return "Flatback"
    elif abs(hip_x - ((shoulder_x_left + shoulder_x_right) / 2)) > swayback_threshold and hip_y < ((shoulder_y_left + shoulder_y_right) / 2):
        return "Swayback Posture"
    else:
        return "Normal"

def detect_bells_palsy(landmarks):
    left_eye_upper_y = landmarks[159].y
    left_eye_lower_y = landmarks[145].y
    right_eye_upper_y = landmarks[386].y
    right_eye_lower_y = landmarks[374].y

    left_eye_diff = abs(left_eye_upper_y - left_eye_lower_y)
    right_eye_diff = abs(right_eye_upper_y - right_eye_lower_y)

    left_mouth_corner = landmarks[61]
    right_mouth_corner = landmarks[291]

    left_mouth_y = left_mouth_corner.y
    right_mouth_y = right_mouth_corner.y

    eye_asymmetry = abs(left_eye_diff - right_eye_diff)
    mouth_asymmetry = abs(left_mouth_y - right_mouth_y)

    eye_threshold = 0.02  # Adjust this threshold as necessary
    mouth_threshold = 0.02  # Adjust this threshold as necessary

    if eye_asymmetry > eye_threshold or mouth_asymmetry > mouth_threshold:
        return "Bell's Palsy"
    else:
        return "Normal"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_frame)
    face_result = face_mesh.process(rgb_frame)

    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_result.pose_landmarks.landmark
        posture = classify_body_posture(landmarks)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Posture: {posture}', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

            landmarks = face_landmarks.landmark
            status = detect_bells_palsy(landmarks)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Status: {status}', (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Posture and Bell\'s Palsy Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
