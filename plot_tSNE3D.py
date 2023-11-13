import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
from matplotlib.lines import Line2D
import seaborn as sns


features_tsne3 = pickle.load(open("t-SNE_3D.dat", "rb"))
palette = sns.color_palette("bright", 11)
labels = pickle.load(open("labels.dat", "rb"))

custom = [Line2D([], [], marker='.', color=palette[0], linestyle='None'),
          Line2D([], [], marker='.', color=palette[1], linestyle='None'),
          Line2D([], [], marker='.', color=palette[2], linestyle='None'),
          Line2D([], [], marker='.', color=palette[3], linestyle='None'),
          Line2D([], [], marker='.', color=palette[4], linestyle='None'),
          Line2D([], [], marker='.', color=palette[5], linestyle='None'),
          Line2D([], [], marker='.', color=palette[6], linestyle='None'),
          Line2D([], [], marker='.', color=palette[7], linestyle='None'),
          Line2D([], [], marker='.', color=palette[8], linestyle='None'),
          Line2D([], [], marker='.', color=palette[9], linestyle='None'),
          Line2D([], [], marker='.', color=palette[9], linestyle='None')]

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    xs=features_tsne3[:,0],
    ys=features_tsne3[:,1],
    zs=features_tsne3[:,2],
    c=labels.T,
    cmap='tab10'
)
ax.set_xlabel('tsne3_1')
ax.set_ylabel('tsne3_2')
ax.set_zlabel('tsne3_3')
plt.legend(custom, ['Coag. mono', 'Coag. bi', 'Cutting', 'Hämolock', 'Table up','Table down', 'Table tilt','Table forth', 'Phone', 'DaVinci', 'Idle'])
plt.savefig('/Users/thomasriedel/Documents/SA_Programming/final_pictures/tsne3.pdf')
plt.show()
