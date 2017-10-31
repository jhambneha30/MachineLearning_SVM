import numpy as np
from sklearn import svm
import h5py
import matplotlib.pyplot as plt

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X, y = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_1.h5')
print(X)
# i is the row number while x is the element.
def my_rbf_kernel(X,Y):
	gamma = 1
	K = np.zeros((X.shape[0], Y.shape[0]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			K[i, j] = np.exp(-gamma * np.power(np.linalg.norm(x-y), 2))
# np.exp((gamma * np.power(np.linalg.norm(x - y), 2)))
	return K


clf = svm.SVC(kernel=my_rbf_kernel)
clf.fit(X, y)
print("SVM trained!")
 # plot the line, the points, and the nearest vectors to the plane
# figure number
fignum = 1
plt.figure(fignum, figsize=(7, 6))
plt.clf()

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#             facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')
print("--------------------------------------", clf.support_vectors_)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')
print("Making colored plot!")
plt.axis('tight')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
            levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
plt.savefig('KernelPlots/data1_kernel_svs_plot')
plt.show()

