import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import DenseNet169
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Training data import
data_dir_train = os.listdir("/Users/thomasriedel/spectrograms/Training/")
if ".DS_Store" in data_dir_train:
    data_dir_train.remove(".DS_Store")

class_labels_train = []

for item in data_dir_train:
    all_classes = os.listdir("/Users/thomasriedel/spectrograms/Training/" + "/" + item)
    if ".DS_Store" in all_classes:
        all_classes.remove(".DS_Store")

    for room in all_classes:
        class_labels_train.append(
            (
                item,
                str("/Users/thomasriedel/spectrograms/Training/" + "/" + item)
                + "/"
                + room,
            )
        )
        df = pd.DataFrame(data=class_labels_train, columns=["Labels", "images"])

label_count_train = df["Labels"].value_counts()

# Spectrogram are resized to the corrsponding model's sizes (EfficientNet = 380 pixel, InceptionV3 = 299 pixels, Rest = 224 pixels)
im_size = 224
# im_size = 299
# im_size = 380

images_train = []
labels_train = []
images_test = []
labels_test = []
images_vali = []
labels_vali = []
images_aug = []
labels_aug = []

for i in data_dir_train:
    data_path = os.path.join("/Users/thomasriedel/spectrograms/Training/", str(i))
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

y_labelencoder = LabelEncoder()
y_train = y_labelencoder.fit_transform(y_train).T

images_train, Y_train = shuffle(images_train, y_train)

## testing data import (similar to training data)
data_dir_test = os.listdir("/Users/thomasriedel/spectrograms/Testing/")
if ".DS_Store" in data_dir_test:
    data_dir_test.remove(".DS_Store")

class_labels_test = []

for item in data_dir_test:
    all_classes = os.listdir("/Users/thomasriedel/spectrograms/Testing/" + "/" + item)
    if ".DS_Store" in all_classes:
        all_classes.remove(".DS_Store")

    for room in all_classes:
        class_labels_test.append(
            (
                item,
                str("/Users/thomasriedel/spectrograms/Testing/" + "/" + item)
                + "/"
                + room,
            )
        )
        df = pd.DataFrame(data=class_labels_test, columns=["Labels", "images"])

label_count_test = df["Labels"].value_counts()

for i in data_dir_test:
    data_path = os.path.join("/Users/thomasriedel/spectrograms/Testing/", str(i))
    filenames = [i for i in os.listdir(data_path)]
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)
        img = cv2.resize(img, [im_size, im_size])
        images_test.append(img)
        labels_test.append(i)

images_test = np.array(images_test)
images_test = images_test.astype("float32") / 255.0

y_test = df["Labels"].values

y_test = y_labelencoder.fit_transform(y_test).T

images_test, Y_test = shuffle(images_test, y_test)

## validation data import (similar to training data)
data_dir_vali = os.listdir("/Users/thomasriedel/spectrograms/Validation/")
if ".DS_Store" in data_dir_vali:
    data_dir_vali.remove(".DS_Store")

class_labels_vali = []

for item in data_dir_vali:
    all_classes = os.listdir(
        "/Users/thomasriedel/spectrograms/Validation/" + "/" + item
    )
    if ".DS_Store" in all_classes:
        all_classes.remove(".DS_Store")

    for room in all_classes:
        class_labels_vali.append(
            (
                item,
                str("/Users/thomasriedel/spectrograms/Validation/" + "/" + item)
                + "/"
                + room,
            )
        )
        df = pd.DataFrame(data=class_labels_vali, columns=["Labels", "images"])

label_count_vali = df["Labels"].value_counts()


for i in data_dir_vali:
    data_path = os.path.join("/Users/thomasriedel/spectrograms/Validation/", str(i))
    filenames = [i for i in os.listdir(data_path)]
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)
        img = cv2.resize(img, [im_size, im_size])
        images_vali.append(img)
        labels_vali.append(i)

images_vali = np.array(images_vali)
images_vali = images_vali.astype("float32") / 255.0

y_vali = df["Labels"].values

y_vali = y_labelencoder.fit_transform(y_vali).T

images_vali, Y_vali = shuffle(images_vali, y_vali)

# Loading/Importing the augmented Spectrograms for testing if they can be classified correctly
data_dir_aug = os.listdir("/Users/thomasriedel/spectrograms_final/AugmentationTest/")
if ".DS_Store" in data_dir_aug:
    data_dir_aug.remove(".DS_Store")

class_labels_aug = []

for item in data_dir_aug:
    all_classes = os.listdir(
        "/Users/thomasriedel/spectrograms_final/AugmentationTest/" + "/" + item
    )
    if ".DS_Store" in all_classes:
        all_classes.remove(".DS_Store")

    for room in all_classes:
        class_labels_aug.append(
            (
                item,
                str(
                    "/Users/thomasriedel/spectrograms_final/AugmentationTest/"
                    + "/"
                    + item
                )
                + "/"
                + room,
            )
        )
        df = pd.DataFrame(data=class_labels_aug, columns=["Labels", "images"])

label_count_aug = df["Labels"].value_counts()


for i in data_dir_aug:
    data_path = os.path.join(
        "/Users/thomasriedel/spectrograms_final/AugmentationTest/", str(i)
    )
    filenames = [i for i in os.listdir(data_path)]
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)
        img = cv2.resize(img, [im_size, im_size])
        images_aug.append(img)
        labels_aug.append(i)

images_aug = np.array(images_aug)
images_aug = images_aug.astype("float32") / 255.0

y_aug = df["Labels"].values

Y_aug = y_labelencoder.fit_transform(y_aug).T

# images_aug, Y_aug = shuffle(images_aug, y_aug)

# Model sizes (EfficientNet = 380 pixel, InceptionV3 = 299 pixels, Rest = 224 pixels)
NUM_CLASSES = 11
NUM_CLASSES = 11
# NUM_CLASSES = 10
# NUM_CLASSES = 9
IMG_SIZE = 224
# IMG_SIZE = 299
# IMG_SIZE = 380


# Creating model based on the pretrained keras models
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    # model = EfficientNetB4(include_top=False, input_tensor=x, weights="imagenet")
    # model = InceptionV3(include_top=False, input_tensor=x, weights="imagenet")
    # model = ResNet50(include_top=False, input_tensor=x, weights="imagenet")
    # model = VGG19(include_top=False, input_tensor=x, weights="imagenet")
    model = MobileNet(include_top=False, input_tensor=x, weights="imagenet")
    # model = DenseNet169(include_top=False, input_tensor=x, weights="imagenet")

    for layer in model.layers:
        layer.trainable = False

    # x = layers.MaxPooling2D(name="max_pool")(model.output)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    # top_dropout_rate = 0.2
    # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(
        NUM_CLASSES,
        kernel_regularizer=regularizers.l2(0.01),
        activation="softmax",
        name="pred",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics="accuracy")
    return model


model = build_model(num_classes=NUM_CLASSES)

# Train the model with following hyperparameters!
hist = model.fit(
    np.array(images_train),
    Y_train,
    batch_size=256,
    epochs=100,
    verbose=2,
    validation_data=[np.array(images_vali), Y_vali],
    callbacks=[EarlyStopping(monitor="val_accuracy", patience=10)],
)


hist_df = pd.DataFrame(hist.history)
hist_json_file = "history/02/MobileNet_bs5_5.json"
with open(hist_json_file, mode="w") as f:
    hist_df.to_json(f)

## Generate Graphs for the accuracies and losses as well as the confusion matrices
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]


def plot_hist(hist):
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "g", label="Training loss")
    plt.plot(epochs, val_loss, "y", label="Validations loss")
    plt.title("Testing and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    plt.plot(epochs, acc, "y", label="Training accuracy")
    plt.plot(epochs, val_acc, "g", label="Validation accuracy")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(hist)

loss, accuracy = model.evaluate(images_test, Y_test)
print("Loss = " + str(loss))
print("Test Accuracy = " + str(accuracy))

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(images_test)

class_names = [
    "mono Coag.",
    "bi Coag.",
    "Cutting",
    "Hämolock",
    "Table up",
    "Table down",
    "tilt table",
    "Table forth",
    "Phone",
    "DaVinci",
    "Idle",
]
cm = confusion_matrix(Y_test, y_pred.argmax(axis=1))
recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-7)
precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-7)
recall = np.mean(recall)
precision = np.mean(precision)
f1_score = (2 * precision * recall) / (precision + recall + 1e-7)
print(recall)
print(precision)
print(f1_score)
cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(
    cmn,
    annot=True,
    fmt=".2f",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap=plt.cm.Blues,
)

plt.xticks(rotation=45)
plt.yticks(rotation=45)
ax.set_title("Confusion Matrix MobileNet", size=18, fontweight="bold")
ax.set_xlabel("Predicted Class", size=13, fontweight="bold")
ax.set_ylabel("True Class", size=13, fontweight="bold")

plt.savefig("final_pictures/02/MobileNet_bs5_5.pdf")
plt.show(block=False)


y_pred_aug = model.predict(images_aug)
class_names_aug = [
    "mono Coag.",
    "bi Coag.",
    "Hämolock",
    "Table up",
    "Table down",
    "tilt table",
    "Table forth",
    "DaVinci",
    "Idle",
]
cm_aug = confusion_matrix(Y_aug, y_pred_aug.argmax(axis=1))
recall_aug = np.diag(cm_aug) / (np.sum(cm_aug, axis=1) + 1e-7)
precision_aug = np.diag(cm_aug) / (np.sum(cm_aug, axis=0) + 1e-7)
recall_aug = np.mean(recall_aug)
precision_aug = np.mean(precision_aug)
f1_score_aug = (2 * precision_aug * recall_aug) / (precision_aug + recall_aug + 1e-7)
print(recall_aug)
print(precision_aug)
print(f1_score_aug)
cmn_aug = cm_aug.astype("float") / cm_aug.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(11, 11))
sns.heatmap(
    cmn_aug,
    annot=True,
    fmt=".2f",
    xticklabels=class_names_aug,
    yticklabels=class_names_aug,
    cmap=plt.cm.Blues,
)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
ax.set_title("Confusion Matrix MobileNet AugmentationTest", size=18, fontweight="bold")
ax.set_xlabel("Predicted Class", size=13, fontweight="bold")
ax.set_ylabel("True Class", size=13, fontweight="bold")
plt.savefig("final_pictures/02/MobileNet_10_SpecAug_Test.pdf")
plt.show(block=False)
