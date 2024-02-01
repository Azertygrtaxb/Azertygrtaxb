import cv2
import mediapipe as mp

# Initialisation de MediaPipe pour les mains et les visages
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Spécifications de dessin
drawing_spec_points = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 200, 0))
drawing_spec_lines = mp_drawing.DrawingSpec(thickness=2, color=(100, 200, 100))

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Traitement pour les mains
    results_hands = hands.process(img_rgb)
    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Traitement pour le visage
    results_face = face_mesh.process(img_rgb)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                img, face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec_points, drawing_spec_lines)

    # Affichage du résultat
    cv2.imshow('MediaPipe Hands and Face Mesh', img)
    if cv2.waitKey(5) & 0xFF == 27:  # Appuyer sur Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()
