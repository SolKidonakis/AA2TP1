import cv2
import mediapipe as mp
import numpy as np

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

gestures = {"piedra": 0, "papel": 1, "tijeras": 2}
dataset = []
labels = []

cap = cv2.VideoCapture(0)

print("Presiona 'p' para piedra, 'a' para papel, 't' para tijeras, o 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de manos
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener las coordenadas de los landmarks
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Grabando dataset', frame)

    # Capturar la etiqueta
    key = cv2.waitKey(1)
    if key == ord('p'):
        dataset.append(landmarks)
        labels.append(gestures["piedra"])
    elif key == ord('a'):
        dataset.append(landmarks)
        labels.append(gestures["papel"])
    elif key == ord('t'):
        dataset.append(landmarks)
        labels.append(gestures["tijeras"])
    elif key == ord('q'):
        break

# Guardar los datos en archivos .npy
np.save("rps_dataset.npy", np.array(dataset))
np.save("rps_labels.npy", np.array(labels))

cap.release()
cv2.destroyAllWindows()
