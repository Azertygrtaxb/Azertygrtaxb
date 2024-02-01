import cv2
import mediapipe as mp

# Initialisation de MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Spécifications de dessin en vert pour les points de repère et les connexions
green_color = (0, 255, 0)
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=green_color)
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=green_color)

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Dessiner les points de repère et les connexions en vert
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_drawing_spec)

    cv2.imshow('MediaPipe Pose', img)
    if cv2.waitKey(5) & 0xFF == 27:  # Appuyer sur Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
