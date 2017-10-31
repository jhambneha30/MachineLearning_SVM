import numpy as np
from sklearn import svm
import h5py
import matplotlib.pyplot as plt

# Label 0: BLUE:
# ('========min dist from mean is:===========', 0.89629439608921446)
# ('========max dist from mean is:===========', 1.1222675367062755)
# Label 1: BROWN:
# ('========min dist from mean is:===========', 0.39093200928573951)
# ('========max dist from mean is:===========', 0.59848785404693494)

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y


def remove_outliers_util(temp_x, label):
    elements = np.array(temp_x)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    # tmp = mean - (2 * sd)
    minDist = 1000
    maxDist = -1

    print("Mean is: ", mean)
    final_list = list()
    for x in temp_x:
        dist = np.linalg.norm(x - mean)
        minDist = min(minDist, dist)
        maxDist = max(maxDist, dist)
        print("dist: ", dist)
        if label == 0 and 0.83 <= dist <= 1.09:
            final_list.append(x)
        elif label == 1 and 0.33 <= dist <= 0.57:
            final_list.append(x)

    print("========min dist from mean is:===========", minDist)
    print("========max dist from mean is:===========", maxDist)

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
        print("================LABEL:==============", label)
        temp_x = list()
        for i, lbl in enumerate(y):
            if lbl == label:
                temp_x.append(X[i])
        x_processed = remove_outliers_util(temp_x, label)
        print("x_processed: ", x_processed)
        y_processed = [label] * len(x_processed)
        print("y_processed: ", y_processed)
        X_final += x_processed
        y_final += y_processed

    X_final = np.asarray(X_final)
    y_final = np.asarray(y_final)
    return X_final, y_final


X_arr, y_arr = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_1.h5')
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
plt.figure(fignum, figsize=(7, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')
print("Making colored plot!")
plt.axis('tight')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(5, 4))
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
            levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
plt.savefig('plots/data1_kernel_noOutlier')
plt.show()
