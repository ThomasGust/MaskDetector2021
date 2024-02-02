import tensorflow as tf
from tensorflow.keras.layers import Dense
from utils2 import prepare_image, bgr_to_rgb, select_face, greyscale
import cv2
import numpy as np

def get_model():
    IMG_SHAPE = (128, 128, 3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # fine_tune_at = 120

    # # Freeze all the layers before the `fine_tune_at` layer
    # for layer in base_model.layers[:fine_tune_at]:
    #   layer.trainable =  False

    base_model.trainable = False

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.models.Sequential([
        Dense(1, activation='sigmoid')
    ])
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    base_learning_rate = 0.0005
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


model = get_model()
model.load_weights("MaskedDetectorNew")

image = cv2.imread("WIN_20211119_20_46_45_Pro.jpg")
image = select_face(image)
#image = cv2.resize(image, (128, 128))
image = bgr_to_rgb(image=image)
pred_img = greyscale(image)
#pred_img = prepare_image(image)
probs = model.predict(np.expand_dims(pred_img, 0))[0][0]
print(probs)