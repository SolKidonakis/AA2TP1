import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


X = np.load('rps_dataset.npy')  
Y = np.load('rps_labels.npy')  


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(21, 2)),  
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Guardar el modelo en un archivo .h5
model.save('mi_modelo_gestos.h5')

print("Modelo guardado en 'mi_modelo_gestos.h5'.")
