import PIL
import keras
import requests
from tqdm import tqdm

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
import os

api_key = "86a71105d612be1bad425258acc1c001"
api_secret = "ec46bf1d93373fad"

RETRIEVE_IMAGES = True
NEW_MODEL = False

IMAGE_SIZE = 128
NUM_IMAGES_PER_CLASS = 200

classes = ["happy", "sad", "angry", "surprised", "disgusted", "fearful", "neutral"]


if RETRIEVE_IMAGES:
    import flickrapi
    import requests

    flickr = flickrapi.FlickrAPI(api_key, api_secret, format="parsed-json")
    print("Retrieving images from Flickr...")
    for c in classes:
        print("Retrieving images for class {}...".format(c))

        try:
            os.mkdir("./images/{}".format(c))
        except:
            pass

        photos = flickr.photos.search(
            text=c, per_page=NUM_IMAGES_PER_CLASS, sort="relevance"
        )
        # urls = []
        with open("./images/{}/{}.txt".format(c, c), "a") as f:
            for i, photo in tqdm(enumerate(photos["photos"]["photo"])):
                url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg".format(
                    photo["farm"], photo["server"], photo["id"], photo["secret"]
                )
                f.write(photo["id"] + "\n")
                image = requests.get(url)
                with open("./images/{}/{}.jpg".format(c, photo["id"]), "wb") as file:
                    file.write(image.content)

NUM_IMAGES = 0
for c in classes:
    with open("./images/{}/{}.txt".format(c, c), "r") as f:
        NUM_IMAGES += len(f.readlines())

X = np.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3))
y = np.zeros((NUM_IMAGES, 1))

iteration = 0
for class_idx, c in enumerate(classes):
    files = os.listdir("./images/{}".format(c))
    for f in files:
        if f[-4:] == ".jpg":
            try:
                img = Image.open("./images/{}/{}".format(c, f))
            except PIL.UnidentifiedImageError:
                continue
            img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            if img.mode == "L":
                img = np.stack((img,) * 3, axis=-1)
            img = np.asarray(img)
            X[iteration] = img
            y[iteration] = class_idx
            iteration += 1

# load model if one exists
try:
    if (NEW_MODEL):
        raise Exception
    model = keras.models.load_model("model.keras")
except:
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
            ),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(IMAGE_SIZE, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(IMAGE_SIZE, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(IMAGE_SIZE, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(IMAGE_SIZE, activation="relu"),
            keras.layers.Dense(7, activation="softmax"),
        ]
    )

model.compile(
    optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train the model
model.fit(X_train, y_train, epochs=25, batch_size=32)
score = model.evaluate(X_test, y_test)
print(score)
# save the model
model.save("model.keras")
