import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers import Dropout, Flatten
#from tensorflow.keras import Model, models
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import DenseNet169

from keras import regularizers
from keras.callbacks import EarlyStopping

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = os.listdir('/Users/thomasriedel/spectrograms/')
if '.DS_Store' in data_dir:
    data_dir.remove('.DS_Store')

class_labels = []

for item in data_dir:
    all_classes = os.listdir('/Users/thomasriedel/spectrograms/' + '/' + item)
    if '.DS_Store' in all_classes:
        all_classes.remove('.DS_Store')

    for room in all_classes:
        class_labels.append((item, str('/Users/thomasriedel/spectrograms/' + '/' + item) + '/' + room))
        df = pd.DataFrame(data=class_labels, columns=['Labels', 'images'])

label_count = df['Labels'].value_counts()

#im_size = 380
im_size = 224
#im_size= 299

images = []
labels = []

for i in data_dir:
    data_path = os.path.join('/Users/thomasriedel/spectrograms/', str(i))
    filenames = [i for i in os.listdir(data_path)]
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)
        img = cv2.resize(img, [im_size, im_size])
        images.append(img)
        labels.append(i)

images = np.array(images)
images = images.astype('float32') / 255.0

y = df['Labels'].values

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y).T

images, Y = shuffle(images, y)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.30)
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.15/(0.15+0.15))



NUM_CLASSES = 11
IMG_SIZE = 224
#IMG_SIZE = 299
#IMG_SIZE = 380

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    #model = EfficientNetB4(include_top=False, input_tensor=x, weights="imagenet")
    #model = InceptionV3(include_top=False, input_tensor=x, weights="imagenet")
    #model = ResNet50(include_top=False, input_tensor=x, weights="imagenet")
    #model = VGG19(include_top=False, input_tensor=x, weights="imagenet")
    model = MobileNet(include_top=False, input_tensor=x, weights="imagenet")
    #model = DenseNet169(include_top=False, input_tensor=x, weights="imagenet")


    for layer in model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, kernel_regularizer=regularizers.l2(0.01), activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNet")
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer, loss=loss, metrics="accuracy"
    )
    return model

model = build_model(num_classes=NUM_CLASSES)

hist = model.fit(np.array(train_x), train_y, epochs=100, verbose=2, validation_data=[np.array(val_x), val_y], callbacks=[EarlyStopping(monitor='val_accuracy', patience=10)])

'''
hist_df = pd.DataFrame(hist.history)
hist_json_file = 'history/MobileNet_5_small.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
'''

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

def plot_hist(hist):
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validations loss')
    plt.title('Testing and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)

preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(test_x)

class_names = ['mono Koag.','bi Koag.','Cutting','Hämolock','Table up','Table down','tilt table','Table forth','Phone','DaVinci', 'Idle']
cm = confusion_matrix(test_y, y_pred.argmax(axis=1))
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(11,  11))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)

plt.xticks(rotation=45)
plt.yticks(rotation=45)
ax.set_title('Confusion Matrix MobileNet', size=18, fontweight="bold")
ax.set_xlabel('Predicted Class', size=13, fontweight="bold")
ax.set_ylabel('True Class', size=13, fontweight="bold")

plt.savefig('final_pictures/MobileNet_5.pdf')
plt.show(block=False)
