import os
import os.path
import argparse
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

# X, Y = load_h5py(args.data)
X, y = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_5.h5')
print(X)
print(y)

# tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
# tsne_results = tsne.fit_transform(X)
# x_data = tsne_results[:, 0]
# y_data = tsne_results[:, 1]
#
# plt.scatter(x_data, y_data, c=y, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.savefig('plots/data1_plot')
#
# plt.show()
#
# plt.close()

