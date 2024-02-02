import cv2
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import tensorflow as tf
from pyngrok import ngrok
import urllib.request
import base64
import random
import string


def plot_image(image, title=''):  # BGR
    px.imshow(bgr_to_rgb(image), height=500, title=title).show()


def plot_data_point(ind, images, labels):
    image = images[ind]
    label = labels[ind]
    plot_image(image, title=f"Image {ind}, LABEL: " + label)


def plot_color_histogram(ind, images, labels):
    img = images[ind]
    label = labels[ind]
    img = img[..., ::-1]
    fig = make_subplots(1, 2)
    fig.add_trace(go.Image(z=img), 1, 1)
    for channel, color in enumerate(['red', 'green', 'blue']):
        fig.add_trace(go.Histogram(x=img[..., channel].ravel(), opacity=0.5,
                                   marker_color=color, name='%s channel' % color), 1, 2)
    fig.update_layout(title=f"Image {ind}, LABEL: " + label, height=500)

    fig.show()


def plot_channels(ind, images, labels, black_and_white=True):
    img = images[ind]
    label = labels[ind]

    if black_and_white:
        r = get_red_channel(img)
        g = get_green_channel(img)
        b = get_blue_channel(img)

        r = np.stack((r,) * 3, axis=-1)
        g = np.stack((g,) * 3, axis=-1)
        b = np.stack((b,) * 3, axis=-1)
    else:
        r, g, b = np.zeros((128, 128, 3)), np.zeros((128, 128, 3)), np.zeros((128, 128, 3))
        r[:, :, 0] = get_red_channel(img)
        g[:, :, 1] = get_green_channel(img)
        b[:, :, 2] = get_blue_channel(img)

    new_images = np.array([bgr_to_rgb(img), r, g, b])

    fig = px.imshow(np.array(new_images), height=500, facet_col=0)
    for i, col in enumerate(["Original", "Red Channel", "Green Channel", "Blue Channel"]):
        fig.layout.annotations[i]['text'] = f"{col}"
    fig.show()


def get_blue_channel(img):
    return img[:, :, 0]


def get_green_channel(img):
    return img[:, :, 1]


def get_red_channel(img):
    return img[:, :, 2]


def compare_blueness(blueness, labels, images):
    df = pd.DataFrame().assign(label=labels, Blueness=[blueness(a) for a in images])
    fig = make_subplots(1, 2, column_widths=[0.3, 0.7])
    fig.add_trace(
        px.violin(df, y='Blueness', x='label').data[0]
        , row=1, col=1
    )
    for i in range(4):
        fig.add_trace(
            px.histogram(df, x='Blueness', color='label', barmode='group').data[i]
            , row=1, col=2
        )
    fig.update_layout(height=600, width=1000)
    fig.show()


def compare_colors(ind, fns, images):
    img = images[ind]
    labels = ["Redness", "Greenness", "Blueness"]
    scores = [f(img) for f in fns]
    df = pd.DataFrame().assign(Labels=labels, Scores=scores)
    fig = make_subplots(1, 2)
    fig.add_trace(
        px.imshow(bgr_to_rgb(img)).data[0],
        row=1, col=1
    )

    fig.add_trace(
        px.bar(df, x='Labels', y='Scores').data[0],
        row=1, col=2
    )
    fig.update_layout(height=600, width=800, title_text="R vs G vs B")
    fig.show()


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    return image


def measure_blueness1(img):
    return np.sum(get_blue_channel(img))


def measure_color1(img, color):
    if color == "red":
        return np.sum(get_red_channel(img))
    elif color == "green":
        return np.sum(get_green_channel(img))
    else:
        return np.sum(get_blue_channel(img))


# Instructor: These values are pretty good
best_lower_red = 130
best_upper_red = 155
best_lower_green = 180
best_upper_green = 215
best_lower_blue = 195
best_upper_blue = 255


def measure_blueness3(img, lower_red, upper_red, lower_green, upper_green, lower_blue, upper_blue):
    return np.sum(
        (lower_blue < get_blue_channel(img)) &
        (get_blue_channel(img) <= upper_blue) &
        (lower_green < get_green_channel(img)) &
        (get_green_channel(img) <= upper_green) &
        (lower_red < get_red_channel(img)) &
        (get_red_channel(img) <= upper_red)
    )


def threshold_classifier(threshold, img):
    blueness = measure_blueness3(img, best_lower_red,
                                 best_upper_red,
                                 best_lower_green,
                                 best_upper_green,
                                 best_lower_blue,
                                 best_upper_blue)

    ### BEGIN CODE HERE (fill in the condition in the if statement)
    if None:
        ### END CODE HERE
        return "Good"
    else:
        return "Bad"


def dotnpy_to_image_dataset(images_path, labels_path, save_dir):
    images = np.load(images_path)
    labels = np.load(labels_path)
    for label in labels:
        if os.path.isdir:
            pass
        else:
            os.mkdir(f"{save_dir}/{label}")
    for i, img in enumerate(images):
        image = bgr_to_rgb(img)
        image = Image.fromarray(image)
        label = labels[i]
        image.save(fp=f"data/mask_images/{label}/image_{i + 1}.png")
        print(f"Image {i + 1} has been successfully saved")


def get_image_dataset(data_dir, val_split=0.2, seed=1234, im_height=128, im_width=128,
                      batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(128, 128),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(im_height, im_width),
        batch_size=batch_size)

    return train_ds, val_ds


def greyscale_to_x_channels(greyscale_np_img_array, channels):
    greyscale_np_img_array = greyscale_np_img_array[:, :, 0]
    stacked_img = np.stack((greyscale_np_img_array,) * channels, axis=-1)
    return stacked_img


def bgr_to_rgb(bgr_image):
    return bgr_image[..., ::-1]


def rgb_to_bgr(image):
    return bgr_to_rgb(image)


def swap_channels(image):
    pass


def transform_data(dataset):
    pass


def launch_website(filename):
    print("Click this link to try your web app:")
    #public_url = ngrok.connect(port='80')
    #rand_anum_str = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(6))
    #os.system(f"ssh {rand_anum_str}:80:localhost: serveo.net")
    os.system(f"cmd /k streamlit run  --server.port 80 {filename}.py")

    return "hehehehehehe"


def get_durl(from_url):
    r = urllib.request.urlopen(from_url)
    print(r.read())
    return "bakofd"


def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')


def get_np_array_from_durl_file(filename):
    path = os.path.join(get_download_path(), filename)
    with open(path, "r") as f:
        url = f.read()
    binary = base64.b64decode(url.split(',')[1])

    with open("temp.png", "wb") as f:
        f.write(binary)
    img = Image.open("temp.png")
    array = np.array(img)
    os.remove(path)
    os.remove("temp.png")
    return array


def get_np_array_from_durl(durl):
    binary = base64.b64decode(durl.split(',')[1])
    with open("temp.png", "wb") as f:
        f.write(binary)


if __name__ == "__main__":
    get_np_array_from_durl_file(filename="test.txt")
