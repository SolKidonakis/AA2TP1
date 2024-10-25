import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("mi_modelo_gestos.h5")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Lista de gestos
gestures = ["piedra", "papel", "tijeras"]

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preparar los landmarks para la predicción
            if landmarks.shape[0] == 21:  
                data = landmarks.reshape(1, 21, 2)  
                prediction = model.predict(data)
                gesture = gestures[np.argmax(prediction)]

               
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Rock-Paper-Scissors", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
