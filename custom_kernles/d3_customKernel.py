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

X, y = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_3.h5')

# i is the row number while x is the element.
def my_linear_kernel(X,Y):
	K = np.dot(X,np.transpose(Y))
	return K


clf = svm.SVC(kernel=my_linear_kernel, decision_function_shape='ovr', gamma=1)
clf.fit(X, y)
# dec = clf.decision_function([[1]])
# print("new shape: ", dec.shape[1])
print("SVM trained!")

 # plot the line, the points, and the nearest vectors to the plane
# figure number
fignum = 1
plt.figure(fignum, figsize=(4, 3))
plt.clf()

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#             facecolors='none', zorder=10, edgecolors='k')
# plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
#             edgecolors='k')
print("Making colored plot!")
# plt.axis('tight')
# x_min = -2
# x_max = 2
# y_min = -2
# y_max = 2
#
# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(XX.shape)
# plt.figure(fignum, figsize=(5, 4))
# plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
# plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#             levels=[-.5, 0, .5])
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
#
# plt.xticks(())
# plt.yticks(())
# plt.savefig('plots/data3_kernel_plot')
# plt.show()

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

print("x_min:", x_min, "--  x_max:", x_max)
print("y_min:", y_min, "--  y_max:", y_max)

xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print("xx shape:", np.shape(xx), "--  yy shape:", np.shape(yy))
plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

plt.title('SVC with linear kernel')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
plt.savefig('data3_kernel_plot')
plt.show()

