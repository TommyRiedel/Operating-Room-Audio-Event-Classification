import pandas as pd
import matplotlib.pyplot as plt


MNet_hist = pd.read_json('history/02/MobileNet_9_small.json')
DNet_hist = pd.read_json('history/02/DenseNet_small.json')
VGG_hist = pd.read_json('history/02/VGG19_small.json')
RNet_hist = pd.read_json('history/02/ResNet_small.json')
Inception_hist = pd.read_json('history/02/Inception_small.json')
ENet_hist = pd.read_json('history/02/EfficientNetB4_small.json')

MNet_val_loss = MNet_hist['val_loss']
MNet_val_acc = MNet_hist['val_accuracy']
MNet_loss = MNet_hist['loss']
MNet_acc = MNet_hist['accuracy']

DNet_val_loss = DNet_hist['val_loss']
DNet_val_acc = DNet_hist['val_accuracy']
DNet_loss = DNet_hist['loss']
DNet_acc = DNet_hist['accuracy']

VGG_val_loss = VGG_hist['val_loss']
VGG_val_acc = VGG_hist['val_accuracy']
VGG_loss = VGG_hist['loss']
VGG_acc = VGG_hist['accuracy']

RNet_val_loss = RNet_hist['val_loss']
RNet_val_acc = RNet_hist['val_accuracy']
RNet_loss = RNet_hist['loss']
RNet_acc = RNet_hist['accuracy']

Inception_val_loss = Inception_hist['val_loss']
Inception_val_acc = Inception_hist['val_accuracy']
Inception_loss = Inception_hist['loss']
Inception_acc = Inception_hist['accuracy']

ENet_val_loss = ENet_hist['val_loss']
ENet_val_acc = ENet_hist['val_accuracy']
ENet_loss = ENet_hist['loss']
ENet_acc = ENet_hist['accuracy']

plt.figure()
epochs_MNet = range(1, len(MNet_val_loss) + 1)
epochs_DNet = range(1, len(DNet_val_loss) + 1)
epochs_VGG = range(1, len(VGG_val_loss) + 1)
epochs_RNet = range(1, len(RNet_val_loss) + 1)
epochs_Inception = range(1, len(Inception_val_loss) + 1)
epochs_ENet = range(1, len(ENet_val_loss) + 1)

plt.rc('font', size=12)
plt.plot(epochs_MNet, MNet_val_loss, 'b', label='MobileNet')
plt.plot(epochs_DNet, DNet_val_loss, 'g', label='DenseNet')
plt.plot(epochs_VGG, VGG_val_loss, 'r', label='VGG19')
plt.plot(epochs_RNet,RNet_val_loss, 'c', label='ResNet50')
plt.plot(epochs_Inception, Inception_val_loss, 'm', label='Validations loss')
plt.plot(epochs_ENet, ENet_val_loss, 'k', label='Validations loss')
plt.title('Comparison of Validation loss', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend(["MobileNet", "DenseNet", "VGG19", "ResNet50", "InceptionV3", "EfficientNetB4"], loc="upper right")
plt.savefig('final_pictures/02/small_val_loss.pdf')
plt.show()


plt.rc('font', size=12)
plt.plot(epochs_MNet, MNet_val_acc, 'b', label='MobileNet')
plt.plot(epochs_DNet, DNet_val_acc, 'g', label='DenseNet')
plt.plot(epochs_VGG, VGG_val_acc, 'r', label='VGG19')
plt.plot(epochs_RNet, RNet_val_acc, 'c', label='ResNet50')
plt.plot(epochs_Inception, Inception_val_acc, 'm', label='InceptionV3')
plt.plot(epochs_ENet, ENet_val_acc, 'k', label='EfficientNetB4')
plt.title("Comparison of Validation accuracy", size=16, fontweight="bold")
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")
plt.legend(["MobileNet", "DenseNet", "VGG19", "ResNet50", "InceptionV3", "EfficientNetB4"], loc="lower right")
plt.savefig('final_pictures/02/small_val_acc.pdf')
plt.show()

plt.rc('font', size=12)
plt.plot(epochs_MNet, MNet_loss, 'b', label='MobileNet')
plt.plot(epochs_DNet, DNet_loss, 'g', label='DenseNet')
plt.plot(epochs_VGG, VGG_loss, 'r', label='VGG19')
plt.plot(epochs_RNet,RNet_loss, 'c', label='ResNet50')
plt.plot(epochs_Inception, Inception_loss, 'm', label='Validations loss')
plt.plot(epochs_ENet, ENet_loss, 'k', label='Validations loss')
plt.title('Comparison of Training loss', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend(["MobileNet", "DenseNet", "VGG19", "ResNet50", "InceptionV3", "EfficientNetB4"], loc="upper right")
plt.savefig('final_pictures/02/small_loss.pdf')
plt.show()

plt.rc('font', size=12)
plt.plot(epochs_MNet, MNet_acc, 'b', label='MobileNet')
plt.plot(epochs_DNet, DNet_acc, 'g', label='DenseNet')
plt.plot(epochs_VGG, VGG_acc, 'r', label='VGG19')
plt.plot(epochs_RNet, RNet_acc, 'c', label='ResNet50')
plt.plot(epochs_Inception, Inception_acc, 'm', label='InceptionV3')
plt.plot(epochs_ENet, ENet_acc, 'k', label='EfficientNetB4')
plt.title("Comparison of Training accuracy", size=16, fontweight="bold")
plt.xlabel("Epochs")
plt.ylabel("Training accuracy")
plt.legend(["MobileNet", "DenseNet", "VGG19", "ResNet50", "InceptionV3", "EfficientNetB4"], loc="lower right")
plt.savefig('final_pictures/02/small_acc.pdf')
plt.show()