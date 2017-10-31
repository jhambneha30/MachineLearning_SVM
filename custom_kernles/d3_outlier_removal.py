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


def remove_outliers_util(temp_x):
    elements = np.array(temp_x)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    # tmp = mean - (2 * sd)

    print("Mean is: ", mean)
    final_list = list()
    for x in temp_x:
        dist = np.linalg.norm(x - mean)
        print("dist: ", dist)
        if dist < 1.8:
            final_list.append(x)

    # final_list = np.asarray(final_list)

    # final_list = [x for x in temp_x if (x > mean - 2 * sd)]
    # final_list = [x for x in final_list if (x < mean + 2 * sd)]
    print("------------------final list:", final_list)
    return final_list



def remove_outliers(X, y):
    unique_classes_set = set(y)
    unique_classes_list = list(unique_classes_set)
    print("unique classes set: ", unique_classes_set)
    print("unique classes list: ", unique_classes_list)
    X_final = list()
    y_final = list()

    for label in unique_classes_list:
        temp_x = list()
        for i, lbl in enumerate(y):
            if lbl == label:
                temp_x.append(X[i])
        x_processed = remove_outliers_util(temp_x)
        print("x_processed: ", x_processed)
        y_processed = [label] * len(x_processed)
        print("y_processed: ", y_processed)
        X_final += x_processed
        y_final += y_processed

    X_final = np.asarray(X_final)
    y_final = np.asarray(y_final)
    return X_final, y_final


X_arr, y_arr = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_3.h5')
# print("X_arr initial: ", X_arr)
# print("y_arr initial: ", y_arr)
print("AFTER OUTLIER REMOVAL:")
X, y = remove_outliers(X_arr, y_arr)
print("X no outliers", X)
print("y no outliers", y)

# i is the row number while x is the element.
def my_rbf_kernel(X,Y):
	# gamma = 0.2
	K = np.zeros((X.shape[0], Y.shape[0]))
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			K[i, j] = np.exp(-1 * np.power(np.linalg.norm(x-y), 2))
# np.exp((gamma * np.power(np.linalg.norm(x - y), 2)))
	return K


clf = svm.SVC(kernel=my_rbf_kernel)
clf.fit(X, y)
print("SVM trained!")
 # plot the line, the points, and the nearest vectors to the plane
# figure number
fignum = 1
plt.figure(fignum, figsize=(4, 3))
plt.clf()

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#             facecolors='none', zorder=10, edgecolors='k')

print("Making colored plot!")
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
plt.savefig('plots/data3_kernel_noOutlier')
plt.show()