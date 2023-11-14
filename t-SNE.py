""" t-SNE:
Dimensionality reduction on a set of features using PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic Neighbor Embedding)
The resulting lower dimensional representation is visualized in 2D and/or 3D scatter plots 
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mpl.use("macosx")
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(rc={"figure.figsize": (11.7, 8.27)})
palette = sns.color_palette("bright", 11)

# Load the features (features.dat) and the corresponding labels (labels.dat)
features = pickle.load(open("features.dat", "rb"))
labels = pickle.load(open("labels.dat", "rb"))

# First reduce the dimensionality of the features to 50 via PCA
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)

# To those transformed features apply t-SNE to generate both 2D and 3D representations!
tsne3 = TSNE(n_components=3, perplexity=80, random_state=42, n_iter=1000)
features_tsne3 = tsne3.fit_transform(features_pca)

tsne2 = TSNE(n_components=2, perplexity=80, random_state=42, n_iter=1000)
features_tsne2 = tsne2.fit_transform(features_pca)

custom = [
    Line2D([], [], marker=".", color=palette[0], linestyle="None"),
    Line2D([], [], marker=".", color=palette[1], linestyle="None"),
    Line2D([], [], marker=".", color=palette[2], linestyle="None"),
    Line2D([], [], marker=".", color=palette[3], linestyle="None"),
    Line2D([], [], marker=".", color=palette[4], linestyle="None"),
    Line2D([], [], marker=".", color=palette[5], linestyle="None"),
    Line2D([], [], marker=".", color=palette[6], linestyle="None"),
    Line2D([], [], marker=".", color=palette[7], linestyle="None"),
    Line2D([], [], marker=".", color=palette[8], linestyle="None"),
    Line2D([], [], marker=".", color=palette[9], linestyle="None"),
    Line2D([], [], marker=".", color=palette[9], linestyle="None"),
]


sns.set_style("whitegrid", {"axes.grid": False})
ax = sns.scatterplot(
    features_tsne2[:, 0], features_tsne2[:, 1], hue=labels.T, palette=palette
)
# Legend with the different classes (11)
plt.legend(
    custom,
    [
        "Coag. mono",
        "Coag. bi",
        "Cutting",
        "Haemolock",
        "Table up",
        "Table down",
        "Table tilt",
        "Table forth",
        "Phone",
        "DaVinci",
        "Idle",
    ],
)
plt.savefig("/Users/thomasriedel/Documents/SA_Programming/final_pictures/tsne2.pdf")
plt.show()

pickle.dump(features_tsne3, open("t-SNE_3D.dat", "wb"))
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(projection="3d")
ax.scatter(
    xs=features_tsne3[:, 0],
    ys=features_tsne3[:, 1],
    zs=features_tsne3[:, 2],
    c=labels.T,
    cmap="tab10",
)
ax.set_xlabel("tsne3_1")
ax.set_ylabel("tsne3_2")
ax.set_zlabel("tsne3_3")
plt.legend(
    custom,
    [
        "Coag. mono",
        "Coag. bi",
        "Cutting",
        "Haemolock",
        "Table up",
        "Table down",
        "Table tilt",
        "Table forth",
        "Phone",
        "DaVinci",
        "Idle",
    ],
)
plt.savefig("/Users/thomasriedel/Documents/SA_Programming/final_pictures/tsne3.pdf")
plt.show()
