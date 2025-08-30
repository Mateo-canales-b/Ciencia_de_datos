# ----- Importaciones y configuración -----
import numpy as np                          # Librería numérica para crear datos y hacer operaciones básicas
import tensorflow as tf                     # TensorFlow: framework de ML
from pathlib import Path                    # Manejo simple de rutas de archivos

np.random.seed(42)                          # Fijamos semilla de NumPy para reproducibilidad
tf.random.set_seed(42)                      # Fijamos semilla de TensorFlow para reproducibilidad

print("TensorFlow:", tf.__version__)        # Muestra la versión de TF, útil para diagnosticar
gpus = tf.config.list_physical_devices('GPU')  # Consulta GPUs visibles (MPS en Apple Silicon)
print("GPUs visibles:", gpus)               # Si ves al menos una, TF debería usar Metal (MPS) en el M1

# ----- Comprobación rápida de GPU (opcional) -----
if gpus:
    # Pequeño test de multiplicación de matrices en GPU para confirmar que funciona
    with tf.device('/GPU:0'):
        a = tf.random.normal((1024, 1024))  # Tensor 1024x1024
        b = tf.random.normal((1024, 1024))  # Otro tensor 1024x1024
        c = tf.matmul(a, b)                 # Multiplicación de matrices (debería usar GPU MPS si está disponible)
    print("Prueba de matmul en GPU completada.")
else:
    print("No se detectó GPU: se usará CPU.")

# ----- Generación de un dataset sintético (regresión y = 2x + 1 + ruido) -----
X = np.linspace(0, 10, 200).astype('float32').reshape(-1, 1)  # 200 puntos entre 0 y 10, forma (200,1)
ruido = 0.5 * np.random.randn(200).astype('float32')          # Ruido gaussiano para simular datos reales
y = (2.0 * X[:, 0] + 1.0 + ruido).astype('float32')           # Relación verdadera con ruido -> vector (200,)

# División train/test (80/20)
n_train = int(0.8 * len(X))               # 80% para entrenamiento
X_train, y_train = X[:n_train], y[:n_train]  # Primer segmento como train
X_test,  y_test  = X[n_train:], y[n_train:]  # Resto como test

# ----- Definición del modelo Keras -----
modelo = tf.keras.Sequential([             # Modelo secuencial: capas apiladas
    tf.keras.layers.Input(shape=(1,)),     # Capa de entrada: una característica (x)
    tf.keras.layers.Dense(8, activation='relu'),  # Capa oculta con 8 neuronas y ReLU
    tf.keras.layers.Dense(1)               # Capa de salida: 1 neurona (predicción continua)
])

# ----- Compilación del modelo -----
modelo.compile(
    optimizer='adam',                      # Optimizador Adam (aprendizaje adaptativo)
    loss='mse',                            # Función de pérdida: error cuadrático medio
    metrics=['mae']                        # Métrica adicional: error absoluto medio
)

# ----- Entrenamiento -----
hist = modelo.fit(
    X_train, y_train,                      # Datos de entrenamiento
    validation_split=0.2,                  # 20% del train para validación interna
    epochs=200,                            # Pasadas por los datos (ajusta según velocidad)
    batch_size=32,                         # Tamaño de lote
    verbose=0                              # 0 = silencioso; 1 = barra; 2 = por época
)
print("Entrenamiento completado.")
print("Última pérdida (train):", float(hist.history['loss'][-1]))         # Muestra la pérdida final en train
print("Última pérdida (val):",   float(hist.history['val_loss'][-1]))     # Muestra la pérdida final en validación

# ----- Evaluación en test -----
test_loss, test_mae = modelo.evaluate(X_test, y_test, verbose=0)  # Evalúa en conjunto de prueba
print("MSE en test:", float(test_loss))                           # Error cuadrático medio en test
print("MAE en test:", float(test_mae))                            # Error absoluto medio en test

# ----- Predicciones de ejemplo -----
x_nuevos = np.array([[0.0], [5.0], [10.0]], dtype='float32')   # Tres entradas de ejemplo
pred = modelo.predict(x_nuevos, verbose=0)                     # Predicción del modelo entrenado
for x_val, y_hat in zip(x_nuevos.flatten(), pred.flatten()):
    print(f"x={x_val:.1f} -> y_pred≈ {y_hat:.3f}")             # Muestra predicciones aproximadas (esperable ~ 1, 11, 21)

# ----- Guardar y cargar el modelo -----
ruta = Path("linear_model_m1.keras")                           # Ruta de guardado (formato Keras moderno)
modelo.save(ruta)                                              # Guarda pesos + arquitectura
print(f"Modelo guardado en: {ruta.resolve()}")                 # Confirma dónde quedó

modelo_recargado = tf.keras.models.load_model(ruta)            # Carga el modelo desde disco
pred2 = modelo_recargado.predict(x_nuevos, verbose=0)          # Vuelve a predecir con el modelo cargado
print("Predicciones tras recargar:", pred2.flatten().round(3).tolist())  # Comprueba consistencia