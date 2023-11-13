import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('macosx')
import pickle

fig,ax = plt.subplots()
ax.plot([1,2,3],[10,-10,30])


pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
figx.show()

a=1