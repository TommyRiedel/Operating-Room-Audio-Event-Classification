"""Feature extraction for t-SNE:

"""
import os

# suppress unnecessary (?) warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNet
import pickle

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

im_size = 224

images_train = []
labels_train = []
images_test = []
labels_test = []
images_vali = []
labels_vali = []
images_aug = []
labels_aug = []

## training data import
# neglecting .DS_Store file (Mac)
data_dir_train = os.listdir("/Users/thomasriedel/spectrograms_tsne/")
if ".DS_Store" in data_dir_train:
    data_dir_train.remove(".DS_Store")

class_labels_train = []

# loads spectrograms from folder
for item in data_dir_train:
    all_classes = os.listdir("/Users/thomasriedel/spectrograms_tsne/" + "/" + item)
    if ".DS_Store" in all_classes:
        all_classes.remove(".DS_Store")

    for room in all_classes:
        class_labels_train.append(
            (
                item,
                str("/Users/thomasriedel/spectrograms_tsne/" + "/" + item) + "/" + room,
            )
        )
        df = pd.DataFrame(data=class_labels_train, columns=["Labels", "images"])

label_count_train = df["Labels"].value_counts()

for i in data_dir_train:
    data_path = os.path.join("/Users/thomasriedel/spectrograms_tsne/", str(i))
    filenames = [i for i in os.listdir(data_path)]
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)
        img = cv2.resize(img, [im_size, im_size])
        images_train.append(img)
        labels_train.append(i)

images_train = np.array(images_train)
images_train = images_train.astype("float32") / 255.0

y_train = df["Labels"].values

# Encodes labels
y_labelencoder = LabelEncoder()
Y_train = y_labelencoder.fit_transform(y_train).T

# images_train, Y_train = shuffle(images_train, y_train)

# 11 classes (small dataset) and img_size = 224 pixels (MobileNet)
NUM_CLASSES = 11
IMG_SIZE = 224


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = MobileNet(include_top=False, input_tensor=x, weights="imagenet")

    # last layer of MobileNet is not trainable
    for layer in model.layers:
        layer.trainable = False

    features = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    model = tf.keras.Model(inputs=inputs, outputs=features, name="MobileNet")
    return model


model = build_model(num_classes=NUM_CLASSES)

sample_count = tf.shape(images_train)[0]
features = model.predict(images_train)
labels = Y_train

# extract features (stored at features.dat) + corresponding labels are stored at labels.dat!
pickle.dump(features, open("features.dat", "wb"))
pickle.dump(labels, open("labels.dat", "wb"))
