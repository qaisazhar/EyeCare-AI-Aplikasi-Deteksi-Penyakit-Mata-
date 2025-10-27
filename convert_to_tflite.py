import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('model/model_katarak.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model/model_katarak.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model converted to TFLite successfully')
