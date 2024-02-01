import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

window_size = 5
x_vals = deque(maxlen=window_size)
y_vals = deque(maxlen=window_size)

prev_x, prev_y = None, None
last_action_time = 0
action_cooldown = 0.5  # Réduire le délai pour une réponse plus rapide

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            x = (index_tip.x + middle_tip.x) / 2
            y = (index_tip.y + middle_tip.y) / 2

            x_vals.append(x)
            y_vals.append(y)

            avg_x = sum(x_vals) / len(x_vals)
            avg_y = sum(y_vals) / len(y_vals)

            index_dist = calculate_distance(index_tip, index_mcp)
            middle_dist = calculate_distance(middle_tip, middle_mcp)

            if index_dist > 0.1 and middle_dist > 0.1:  # Ajustez le seuil si nécessaire
                current_time = time.time()
                if current_time - last_action_time > action_cooldown:
                    if prev_x is not None and prev_y is not None:
                        if avg_x > prev_x + 0.05:  # Sensibilité accrue
                            print("Mouvement détecté : Droite")
                            pyautogui.press('right')
                            last_action_time = current_time
                        elif avg_x < prev_x - 0.05:
                            print("Mouvement détecté : Gauche")
                            pyautogui.press('left')
                            last_action_time = current_time

                        if avg_y > prev_y + 0.05:
                            print("Mouvement détecté : Haut")
                            pyautogui.press('up')
                            last_action_time = current_time
                        elif avg_y < prev_y - 0.05:
                            print("Mouvement détecté : Bas")
                            pyautogui.press('down')
                            last_action_time = current_time

            prev_x, prev_y = avg_x, avg_y

    cv2.imshow('Gesture Control', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
