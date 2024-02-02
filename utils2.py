from PIL import Image
import cv2
import numpy as np
import plotly.express as px
import tensorflow as tf

from tensorflow.keras.layers import Dense


def get_blue_channel(img): return img[:, :, 0]


def get_green_channel(img): return img[:, :, 1]


def get_red_channel(img): return img[:, :, 2]


def bgr_to_rgb(image): return image[..., ::-1]


def rgb_to_bgr(image): return bgr_to_rgb(image)  # This is not a mistake


def plot_image(image, title=""):  # BGR
    px.imshow(bgr_to_rgb(image), height=500, title=title).show()


LABEL2TEXT = {
    1: "Masked",
    0: "Improperly Masked/No Mask"
}
LABEL2TEXTNEW = {
    "Mask": "Good",
    "Mask_Mouth_Chin": "Bad",
    "Mask_Nose_Mouth": "Bad",
    "Mask_Chin": "Bad"
}
LABELS = ["Mask", "Mask_Mouth_Chin", "Mask_Nose_Mouth", "Mask_Chin"]


def plot_data_point(ind, images, labels):
    image = images[ind]
    label = labels[ind]
    plot_image(image, title=f"Image {ind}, LABEL: " + str(label))


def plot_augmentations(test_image, augmentations):
    augmented = [test_image] + [a(test_image) for a in augmentations]
    augmented = [bgr_to_rgb(i) for i in augmented]
    augmented = np.array(augmented)

    labels = ['Original'] + [a.__name__.title() for a in augmentations]

    fig = px.imshow(augmented, facet_col=0)
    for i, col in enumerate(labels):
        fig.layout.annotations[i]["text"] = col
    fig.show()


def greyscale(image):
    image = tf.image.rgb_to_grayscale(image).numpy()
    image = np.repeat(image, 3, -1)
    return image


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    return image


### Getting the Dlib Shape predictor!
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def draw_rect_on_image(image, good=True):
    faces = faceCascade.detectMultiScale(
        image,
        # minNeighbors = 5,
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        if good:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def plot_face(image, good=True):
    faces = faceCascade.detectMultiScale(
        image
    )

    if len(faces) < 1:
        plot_image(image)

    img_copy = image.copy()

    for x, y, w, h in faces:
        if good:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plot_image(img_copy)


def select_face(image):
    faces = faceCascade.detectMultiScale(image)
    if len(faces) < 1:
        return tf.image.resize(image, (128, 128))
    else:
        x, y, w, h = faces[0]
        cropped = tf.image.crop_to_bounding_box(
            image, y, x, h, w
        )
        return tf.image.resize(cropped, (128, 128))


def sequence_of_augmentations(img, seq):
    for f in seq:
        img = f(img)
    return img


# ======= LOAD DATA ===============


def get_model():
    IMG_SHAPE = (128, 128, 3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

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


def get_model_new():
    IMG_SHAPE = (128, 128, 3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

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
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    base_learning_rate = 0.0005
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def plot_sequence_of_images(seq, labels=None):
    seq_rgb = np.array([bgr_to_rgb(im) for im in seq])
    fig = px.imshow(seq_rgb, facet_col=0)

    if labels is not None:
        for i, col in enumerate(labels):
            fig.layout.annotations[i]['text'] = f"{col}"
    else:
        for i in range(len(seq)):
            fig.layout.annotations[i]['text'] = ''
    fig.show()


def run_model_on_single_image(model, img, thresh=0.5):
    print(img.shape())
    prob = model.predict(np.expand_dims(img, 0))[0][0]
    print("Making model predictions went fine.")
    pred = int(prob > thresh)
    #plot_image(img, title=f'Probability Mask = {prob} \n Prediction: {LABEL2TEXT[pred]}')
    return prob, pred


def save_np_array(array, filename, handler="PIL"):
    if handler == "PIL":
        im = Image.fromarray(array)
        im.save(filename)
    else:
        cv2.imwrite(filename=filename, img=array)


def prepare_image(img):
    return greyscale(select_face(img))


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def get_download_link(filename, text):

    href = f'<a href="{filename}">{text}</a>'
    return href
