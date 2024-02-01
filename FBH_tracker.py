
#### PROGRAMME POUR UNE SEULE POSE ####

import cv2
import mediapipe as mp

# Initialisation de MediaPipe pour les mains, les visages et la pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_drawing = mp.solutions.drawing_utils

# Spécifications de dessin
drawing_spec_points = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(100, 200, 100))
drawing_spec_lines = mp_drawing.DrawingSpec(thickness=4, color=(100, 200, 100))
green_color = (0, 255, 0)
blue_color = (0, 0, 255)
pose_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=green_color)
hands_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=blue_color)

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Traitement pour les mains
    results_hands = hands.process(img_rgb)
    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS,  landmark_drawing_spec=hands_drawing_spec,
            connection_drawing_spec=hands_drawing_spec)

    # Traitement pour le visage
    results_face = face_mesh.process(img_rgb)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                img, face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec_points, drawing_spec_lines)

    # Traitement pour la pose
    results_pose = pose.process(img_rgb)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_drawing_spec,
            connection_drawing_spec=pose_drawing_spec)

    # Affichage du résultat
    cv2.imshow('MediaPipe Hands, Face Mesh, and Pose', img)
    if cv2.waitKey(5) & 0xFF == 27:  # Appuyer sur Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()



#### PROGRAMME QUI FONCTIONNE MAIS PAS POSSIBLE DE METTRE PLUSIEURS POSES ####


# import cv2
# import mediapipe as mp

# # Initialisation de MediaPipe pour les mains et les visages
# mp_hands = mp.solutions.hands
# mp_face_mesh = mp.solutions.face_mesh
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.6)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils

# # Spécifications de dessin pour les mains et les visages
# drawing_spec_points = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(100, 200, 100))
# drawing_spec_lines = mp_drawing.DrawingSpec(thickness=2, color=(100, 200, 100))

# # Charger le modèle de détection de pose d'OpenCV
# net = cv2.dnn.readNetFromCaffe("C:/Users/axelo/OneDrive/Desktop/Tracker/pose_deploy_linevec.prototxt", "C:/Users/axelo/OneDrive/Desktop/Tracker/pose_iter_440000.caffemodel")


# # Fonction pour dessiner la pose
# def draw_pose(img, output, threshold=0.1):
#     H, W = img.shape[:2]
#     points = []

#     for i in range(18):  # Nombre de points de repère
#         heatMap = output[0, i, :, :]
#         _, conf, _, point = cv2.minMaxLoc(heatMap)
#         x = (W * point[0]) / output.shape[3]
#         y = (H * point[1]) / output.shape[2]

#         if conf > threshold:
#             points.append((int(x), int(y)))
#         else:
#             points.append(None)

#     POSE_PAIRS = [
#         [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], 
#         [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]
#     ]

#     for pair in POSE_PAIRS:
#         partA = pair[0]
#         partB = pair[1]

#         if points[partA] and points[partB]:
#             cv2.line(img, points[partA], points[partB], (0, 255, 0), 2)
#             cv2.circle(img, points[partA], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#             cv2.circle(img, points[partB], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# # Initialisation de la caméra
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Traitement pour les mains avec MediaPipe
#     results_hands = hands.process(img_rgb)
#     if results_hands.multi_hand_landmarks:
#         for handLms in results_hands.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

#     # Traitement pour le visage avec MediaPipe
#     results_face = face_mesh.process(img_rgb)
#     if results_face.multi_face_landmarks:
#         for face_landmarks in results_face.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 img, face_landmarks, 
#                 mp_face_mesh.FACEMESH_CONTOURS,
#                 drawing_spec_points, drawing_spec_lines)

#     # Préparation de l'image pour la détection de pose d'OpenCV
#     blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
#     net.setInput(blob) 





#### PROGRAMME POUR PLUSIEURS PERSONNES MAIS PBM CPU : PAS ASSEZ PUISSANT ####


# import cv2
# import mediapipe as mp

# # Initialisation de MediaPipe pour les mains et les visages
# mp_hands = mp.solutions.hands
# mp_face_mesh = mp.solutions.face_mesh
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.6)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils

# # Spécifications de dessin pour les mains et les visages
# drawing_spec_points = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(100, 200, 100))
# drawing_spec_lines = mp_drawing.DrawingSpec(thickness=2, color=(100, 200, 100))

# # Charger le modèle de détection de pose d'OpenCV
# net = cv2.dnn.readNetFromCaffe(
#     "C:/Users/axelo/OneDrive/Desktop/Tracker/pose_deploy_linevec.prototxt",
#     "C:/Users/axelo/OneDrive/Desktop/Tracker/pose_iter_440000.caffemodel"
# )

# # Fonction pour dessiner la pose
# def draw_pose(img, output, threshold=0.1):
#     H, W = img.shape[:2]
#     points = []

#     for i in range(18):  # Nombre de points de repère
#         heatMap = output[0, i, :, :]
#         _, conf, _, point = cv2.minMaxLoc(heatMap)
#         x = (W * point[0]) / output.shape[3]
#         y = (H * point[1]) / output.shape[2]

#         if conf > threshold:
#             points.append((int(x), int(y)))
#         else:
#             points.append(None)

#     POSE_PAIRS = [
#         [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], 
#         [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]
#     ]

#     for pair in POSE_PAIRS:
#         partA = pair[0]
#         partB = pair[1]

#         if points[partA] and points[partB]:
#             cv2.line(img, points[partA], points[partB], (0, 255, 0), 2)
#             cv2.circle(img, points[partA], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#             cv2.circle(img, points[partB], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# # Initialisation de la caméra
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Traitement pour les mains avec MediaPipe
#     results_hands = hands.process(img_rgb)
#     if results_hands.multi_hand_landmarks:
#         for handLms in results_hands.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

#     # Traitement pour le visage avec MediaPipe
#     results_face = face_mesh.process(img_rgb)
#     if results_face.multi_face_landmarks:
#         for face_landmarks in results_face.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 img, face_landmarks, 
#                 mp_face_mesh.FACEMESH_CONTOURS,
#                 drawing_spec_points, drawing_spec_lines)

#     # Préparation de l'image pour la détection de pose d'OpenCV
#     blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
#     net.setInput(blob)
#     output = net.forward()  # Exécution de la détection de pose

#     # Dessiner les résultats de la détection de pose d'OpenCV sur l'image
#     draw_pose(img, output)

#     # Affichage du résultat
#     cv2.imshow('MediaPipe Hands, Face Mesh, and OpenCV Pose', img)
#     if cv2.waitKey(5) & 0xFF == 27:  # Appuyer sur Echap pour quitter
#         break

# cap.release()
# cv2.destroyAllWindows()
