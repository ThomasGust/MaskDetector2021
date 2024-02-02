import streamlit as st
import streamlit.components.v1 as components
from utils import get_np_array_from_durl_file
import os
import time
from utils import get_download_path
import random
from utils2 import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')


st.title("Mask Detector Web App!")

menu = ["Use Model", "See the Data"]

choice = st.sidebar.selectbox("Menu", menu)
st.sidebar.write("Functionality for multiple pages is added through this sidebar!")


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
if choice == "Use Model":
    image = None
    st.subheader("Use the Model")
    st.write("App allows for images to given through the camera, or through a file upload!")
    use_type = ["Upload", "Camera"]
    u_choice = st.radio("Select Input Type", use_type)

    if u_choice == "Camera":
        take_a_photo = st.empty()
        t = take_a_photo.button("Take a photo!")
        st.write(
            "The way the camera works is pretty weird (no server side js so the model will download a file called test.txt, if its being finicky, just try uploading an image from your computer.")
        c_height = 600
        if t:
            if os.path.isfile("temp.png"):
                os.remove("temp.png")
            if os.path.isfile("Temp_Pred.png"):
                os.remove("Temp_Pred.png")
            print("Talking a photo!")
            c1 = components.html(f"""
                
                <script>
                async function takePhoto(){{
                      
                      const div = document.createElement('div');
                      const capture = document.createElement('button');
                      capture.textContent = 'Capture';
                      div.appendChild(capture);
                
                      const video = document.createElement('video');
                      video.style.display = 'block';
                      const stream = await navigator.mediaDevices.getUserMedia({{video: true}});
                
                      document.body.appendChild(div);
                      div.appendChild(video);
                      video.srcObject = stream;
                      await video.play();
                
                
                      await new Promise((resolve) => capture.onclick = resolve);
                
                      const canvas = document.createElement('canvas');
                      canvas.width = video.videoWidth;
                      canvas.height = video.videoHeight;
                      canvas.getContext('2d').drawImage(video, 0, 0);
                      stream.getVideoTracks()[0].stop();
                      div.remove();
                      
                      savedImgDataURL = canvas.toDataURL("image/jpeg;base64");
                      //scannedImage = canvas.getImageData(0, 0, canvas.width, canvas.height);
                
                      parent.document.getElementsByTagName("datahold")[0].id = savedImgDataURL;
                      parent.document.getElementsByTagName("datahold")[0].innerHTML = savedImgDataURL;
                      var blob = new Blob([savedImgDataURL], {{type: "image/jpeg;base64"}});
                      parent.saveAs(blob, "test.txt");
                      
                      
                      //console.log(parent.document.getElementsByTagName("datahold")[0].innerHTML);
                      
                      //console.log(parent.document.innerHTML);
                      return savedImgDataURL;
                }}
                takePhoto();
                frame = parent.document.querySelectorAll("iframe")[0];
                frame.parentNode.removeChild(frame);
                </script>
                """, height=c_height)
            isFlag = False
            while isFlag is False:
                isFlag = os.path.isfile(os.path.join(get_download_path(), f"test.txt"))
                time.sleep(1)

            image = get_np_array_from_durl_file(filename=f"test.txt")
            image = cv2.resize(image, (128, 128))
            image = image[:, :, :-1]

            save_np_array(array=image, filename="temp.png")
            pred_img = prepare_image(image)
            save_np_array(array=pred_img, filename="Temp_Pred.png", handler="cv2")
            probability, prediction = run_model_on_single_image(model=model, img=pred_img, thresh=0.5)
            c_height = 0
            c2 = components.html("""

                                        <script>
                                        frame = parent.document.querySelectorAll("iframe")[0];
                                        frame.parentNode.removeChild(frame);
                                        </script>
                                        """, height=0)
            st.subheader("Model generates the following predictions based on a machine learning technique called "
                         "neural networks:")
            st.image("temp.png",
                     caption=f"Inputed Image, Probability Mask: {probability}, Prediction: {LABEL2TEXT[prediction]}",
                     width=512)
            st.image("Temp_Pred.png", caption="This is the processed image the model used to make its decision.",
                     width=512)
    elif u_choice == "Upload":

        f = st.file_uploader("Upload an Image", type=["png", "jpeg", "jpg"])
        if f is not None:
            if os.path.isfile("temp.png"):
                os.remove("temp.png")
            if os.path.isfile("Temp_Pred.png"):
                os.remove("Temp_Pred.png")
            print("That went fine 1")
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            print("That went fine 2")
            image = cv2.imdecode(file_bytes, 1)
            print("That went fine 3")
            image = cv2.resize(image, (128, 128))
            print("That went fine 4")
            image = bgr_to_rgb(image=image)
            print("That went fine 5")
            pred_img = prepare_image(image)
            print("That went fine 6")
            save_np_array(array=image, filename="temp.png")
            print("That went fine 7")
            save_np_array(array=pred_img, filename="Temp_Pred.png", handler="cv2")
            print("That went fine 8")
            # probability, prediction = run_model_on_single_image(model=model, img=pred_img, thresh=0.5) #THIS IS THE LINE THAT BREAKS EVERYTHING
            probability = model.predict(np.expand_dims(pred_img, 0))[0][0]
            #probability = 0.6
            print("That went fine 8.5")
            prediction = int(probability > 0.5)
            print("That went fine 9")
            st.image("temp.png", caption=f"Probability Mask: {probability}, Prediction: {LABEL2TEXT[prediction]}",
                     width=512)
            print("That went fine 10")
            st.image("Temp_Pred.png", caption="This is the processed image the model used to make its prediction",
                     width=512)
            print("That went fine 11")
elif choice == "See the Data":
    st.subheader("Visualize The Data")
    st.write("Credit goes to Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi,"
             "for the dataset we used (MaskedFace-Net).")
    im_paths = []
    show_im_paths = []

    for filename in os.listdir("Data/mask_images/Mask"):
        if filename.endswith(".png"):
            im_paths.append([os.path.join("Data/mask_images/Mask", filename), "Good"])
    for filename in os.listdir("Data/mask_images/Mask_Chin"):
        if filename.endswith(".png"):
            im_paths.append([os.path.join("Data/mask_images/Mask_Chin", filename), "Bad"])
    for filename in os.listdir("Data/mask_images/Mask_Mouth_Chin"):
        if filename.endswith(".png"):
            im_paths.append([os.path.join("Data/mask_images/Mask_Mouth_Chin", filename), "Bad"])
    for filename in os.listdir("Data/mask_images/Mask_Nose_Mouth"):
        if filename.endswith(".png"):
            im_paths.append([os.path.join("Data/mask_images/Mask_Nose_Mouth", filename), "Bad"])

    ims_show = int(st.slider("How many sample images do you want to display?", min_value=5, max_value=100))

    for im in range(ims_show):
        flag = True
        while flag:
            pa = random.choice(im_paths)
            if pa not in show_im_paths:
                show_im_paths.append(pa)
                flag = False
    ims = []
    cs = []
    for i in show_im_paths:
        ims.append(i[0])
        cs.append(i[1])

    st.image(ims, cs)

    st.markdown(get_download_link(filename="MaskImages.zip", text="Download the data!"), unsafe_allow_html=True)
    st.write("The images are grouped into 4 different categories (or classes), and we teach our neural network to "
             "distinguish between these classes based on a dataset of a bunch of images (which is sampled from above). "
             "Three of the classes are considered 'Bad' (for improper mask use) and 1 is considered 'Good' for proper mask use.")
    del_err = components.html("""
    <script>
        parent.document.getElementsByClassName("st-ae st-af st-ag st-ah st-ai st-aj st-ak st-ej st-am st-f6 st-ao st-ap st-aq st-ar st-as st-at st-ek st-av st-aw st-ax st-ay st-az st-b9 st-b1 st-b2 st-b3 st-b4 st-b5 st-el")[0].remove()
    </script>
    """)
    st.subheader("Here comes the fun part! Data augmentation!")
    st.write("Although we would like to think that the model will only receive perfect images, "
             "this is far from the case, and we need to make our model more versatile. (The dataset is also "
             "not perfect, as it only contains images of people with blue surgical masks).")
    st.write("One way we can achieve this, is through data augmentation which applies a series of operations to a set "
             "of, "
             "images to increase the variety of data in the dataset. (To combat the color problem we fed the model a "
             "greyscaled version of the augmented dataset).")
    zoom_range = float(st.number_input("Enter a zoom range for the Augmenter (measured in %):", value=80)) / 100
    rotation_range = float(
        st.number_input("Enter a rotation range for the Augmenter (measured in degrees):", value=60))
    horizontal_flip = bool(st.selectbox(label="Horizontal flip (True or false)", options=["True", "False"], index=0))
    vertical_flip = bool(st.selectbox(label="Vertical flip (True or false)", options=["True", "False"], index=0))
    width_shift_range = float(
        st.number_input("Enter a width shift range for the Augmenter (measured as a fraction of the images size)",
                        value=40))
    height_shift_range = float(
        st.number_input("Enter a height shift range for the Augmenter (measured as a fraction of the images size)",
                        value=40))
    shear_range = float(
        st.number_input("Enter the shear range value for the Augmenter (measured in degrees)", value=10))
    # 80, 60, True, True, 40, 40, 10

    data_gen = ImageDataGenerator(zoom_range=zoom_range
                                  , rotation_range=rotation_range
                                  , horizontal_flip=horizontal_flip
                                  , vertical_flip=vertical_flip
                                  , width_shift_range=width_shift_range
                                  , height_shift_range=height_shift_range
                                  , shear_range=shear_range
                                  , fill_mode='constant'
                                  )
    images = np.load("Data/x.npy")
    labels = np.load("Data/y.npy")

    images = tf.image.rgb_to_grayscale(images).numpy()
    images = np.repeat(images, 3, -1).astype(np.uint8)
    data_transformed = data_gen.flow(x=images, y=labels)

    st.write("Here are some processed sample images based on the values you have inputed above.:")
    aims_show = int(st.slider("How many augmented samples would you like to see", min_value=1, max_value=32))

    batch, labels = data_transformed.next()
    ic_pairs = []

    for index in range(aims_show):
        flag = True
        while flag:
            rand = random.randint(0, 31)
            fname = f"Data/temp_images/temp_augmented_image_{rand}.png"
            cv2.imwrite(filename=fname,
                        img=batch[rand])
            a = [fname, LABEL2TEXTNEW[labels[[rand]][0]]]
            if a not in ic_pairs:
                ic_pairs.append(a)
                flag = False
    aims = []
    acs = []
    for ic_pair in ic_pairs:
        aims.append(ic_pair[0])
        acs.append(ic_pair[1])
    st.image(aims, acs)
    st.write("Now try it out on yourself!")
    f = st.file_uploader("Upload an Image", type=["png", "jpeg", "jpg"])
    if f is not None:
        if os.path.isfile("temp.png"):
            os.remove("temp.png")
        if os.path.isfile("Temp_Pred.png"):
            os.remove("Temp_Pred.png")
        print("That went fine 1")
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        print("That went fine 2")
        image = cv2.imdecode(file_bytes, 1)
        print("That went fine 3")
        image = cv2.resize(image, (128, 128))
        print("That went fine 4")
        image = bgr_to_rgb(image=image)
        transformed_images = [data_gen.flow(np.array([image])).next()[0] for _ in range(16)]

        for i, image in enumerate(transformed_images):
            image = bgr_to_rgb(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            image = cv2.resize(image, (128, 128))
            cv2.imwrite(f"Data\\temp_images\\captured_aug\\capture_aug_image_{i}.png",
                        img=image)
        transformed_images = []
        for image in os.listdir("Data\\temp_images\\captured_aug"):
            if image.endswith(".png"):
                transformed_images.append(os.path.join("Data\\temp_images\\captured_aug", image))
        st.image(transformed_images)
