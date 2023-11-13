import pandas as pd
import matplotlib.pyplot as plt


MNet_hist1 = pd.read_json('history/MobileNet_1_small.json')
MNet_hist2 = pd.read_json('history/MobileNet_2_small.json')
MNet_hist3 = pd.read_json('history/MobileNet_3_small.json')
MNet_hist4 = pd.read_json('history/MobileNet_4_small.json')
MNet_hist5 = pd.read_json('history/MobileNet_5_small.json')

MNet_val_loss1 = MNet_hist1['val_loss']
MNet_val_acc1 = MNet_hist1['val_accuracy']
MNet_loss1 = MNet_hist1['loss']
MNet_acc1 = MNet_hist1['accuracy']

MNet_val_loss2 = MNet_hist2['val_loss']
MNet_val_acc2 = MNet_hist2['val_accuracy']
MNet_loss2 = MNet_hist2['loss']
MNet_acc2 = MNet_hist2['accuracy']

MNet_val_loss3 = MNet_hist3['val_loss']
MNet_val_acc3 = MNet_hist3['val_accuracy']
MNet_loss3 = MNet_hist3['loss']
MNet_acc3 = MNet_hist3['accuracy']

MNet_val_loss4 = MNet_hist4['val_loss']
MNet_val_acc4 = MNet_hist4['val_accuracy']
MNet_loss4 = MNet_hist4['loss']
MNet_acc4 = MNet_hist4['accuracy']

MNet_val_loss5 = MNet_hist5['val_loss']
MNet_val_acc5 = MNet_hist5['val_accuracy']
MNet_loss5 = MNet_hist5['loss']
MNet_acc5 = MNet_hist5['accuracy']

epochs_MNet1 = range(1, len(MNet_val_loss1) + 1)
epochs_MNet2 = range(1, len(MNet_val_loss2) + 1)
epochs_MNet3 = range(1, len(MNet_val_loss3) + 1)
epochs_MNet4 = range(1, len(MNet_val_loss4) + 1)
epochs_MNet5 = range(1, len(MNet_val_loss5) + 1)

plt.rc('font', size=12)
plt.plot(epochs_MNet1, MNet_val_loss1, 'b', label='lr=1e-3')
plt.plot(epochs_MNet2, MNet_val_loss2, 'g', label='lr=1e-4')
plt.plot(epochs_MNet3, MNet_val_loss3, 'r', label='lr=1e-5')
plt.plot(epochs_MNet4, MNet_val_loss4, 'c', label='lr=5e-2')
plt.plot(epochs_MNet5, MNet_val_loss5, 'm', label='lr=5e-3')
plt.title('Comparison of Validation loss', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend(["lr=1e-3", "lr=1e-4", "lr=1e-5", "lr=5e-2", "lr=5e-3"], loc="upper right")
plt.savefig('final_pictures/small_lr_val_loss.pdf')
plt.show()

plt.rc('font', size=12)
plt.plot(epochs_MNet1, MNet_val_acc1, 'b', label='lr=1e-3')
plt.plot(epochs_MNet2, MNet_val_acc2, 'g', label='lr=1e-4')
plt.plot(epochs_MNet3, MNet_val_acc3, 'r', label='lr=1e-5')
plt.plot(epochs_MNet4, MNet_val_acc4, 'c', label='lr=5e-2')
plt.plot(epochs_MNet5, MNet_val_acc5, 'm', label='lr=5e-3')
plt.title('Comparison of Validation accuracy', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.legend(["lr=1e-3", "lr=1e-4", "lr=1e-5", "lr=5e-2", "lr=5e-3"], loc="upper right")
plt.savefig('final_pictures/small_lr_val_acc.pdf')
plt.show()

plt.rc('font', size=12)
plt.plot(epochs_MNet1, MNet_loss1, 'b', label='lr=1e-3')
plt.plot(epochs_MNet2, MNet_loss2, 'g', label='lr=1e-4')
plt.plot(epochs_MNet3, MNet_loss3, 'r', label='lr=1e-5')
plt.plot(epochs_MNet4, MNet_loss4, 'c', label='lr=5e-2')
plt.plot(epochs_MNet5, MNet_loss5, 'm', label='lr=5e-3')
plt.title('Comparison of Training loss', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend(["lr=1e-3", "lr=1e-4", "lr=1e-5", "lr=5e-2", "lr=5e-3"], loc="upper right")
plt.savefig('final_pictures/small_lr_loss.pdf')
plt.show()

plt.rc('font', size=12)
plt.plot(epochs_MNet1, MNet_acc1, 'b', label='lr=1e-3')
plt.plot(epochs_MNet2, MNet_acc2, 'g', label='lr=1e-4')
plt.plot(epochs_MNet3, MNet_acc3, 'r', label='lr=1e-5')
plt.plot(epochs_MNet4, MNet_acc4, 'c', label='lr=5e-2')
plt.plot(epochs_MNet5, MNet_acc5, 'm', label='lr=5e-3')
plt.title('Comparison of Training accuracy', size=16, fontweight="bold")
plt.xlabel('Epochs')
plt.ylabel('Training accuracy')
plt.legend(["lr=1e-3", "lr=1e-4", "lr=1e-5", "lr=5e-2", "lr=5e-3"], loc="upper right")
plt.savefig('final_pictures/small_lr_acc.pdf')
plt.show()