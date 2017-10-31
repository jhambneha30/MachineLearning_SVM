import numpy as np
from sklearn import svm
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import itertools
from operator import itemgetter

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X, y = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/part_C_train.h5')

def partition_data(x_data, k, fold_size):
    start = k * fold_size
    end = start + fold_size
    train = np.concatenate((x_data[:start], x_data[end:]))
    test = x_data[start:end]
    # print(test)
    # print(train)
    return train, test

def decision_function(test_point, clf):
    ''' params = model parameters
        support_vectors = support vectors
        num_sv = # of support vectors per class
        dual_coef  = dual coefficients
        b  = intercepts
        cs = list of class names
        X  = feature to predict
    '''
    # support_vectors = clf.support_vectors_
    # num_sv = clf.n_support_
    # dual_coef = clf.dual_coef_
    w = clf.coef_
    b = clf.intercept_
    # cs = clf.classes_

    res = np.dot(w, test_point) + b
    return res



def predict(test_data, classifier_dictionary):
    predictions = list()
    for point in test_data:
        max_value = -99999
        max_label = None
        for lbl_key in classifier_dictionary.keys():
            # print("LABEL is:=========", lbl_key)
            decision_value = decision_function(point, classifier_dictionary.get(lbl_key))
            # print("decision val: ", decision_value)
            if decision_value > max_value:
                max_value = decision_value
                max_label = lbl_key
        # print("decision val: ", max_value)
        # print("label for it: ", max_label)
        predictions.append(max_label)
    return predictions

labels = list()
for l in y:
    i = np.where(l == 1)[0][0]
    labels.append(i)


def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)

    return acc

# -----------------------Comment this code below while plotting confusion matrix---------------------

# # Taking penalty C and max_iters as the parameters on which grid search is to be applied
# accuracy_list = list()
# C = [1, 500]
# max_iter = [-1, 100, 1000]
#
# # define the grid here
# parameters = [(c, mi) for c in C
#                  for mi in max_iter]
#
# # do the grid search with k fold cross validation
# # For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
# graph_x_dt, graph_y_dt, param_comb_list = list(), list(), list()
#
# classifier_dict = {}
# num_folds = 5
# mean_acc = 0
#
#
# for (c, mi) in parameters:
#     mean_acc = 0
#     param_comb = [c, mi]
#     print("param comb is: ", param_comb)
#     for k in xrange(0, num_folds):
#         length = len(X)
#         x_train_set, x_test_set = partition_data(X, k, length / num_folds)
#         y_train_set, y_test_set = partition_data(labels, k, length / num_folds)
#
#         unique_classes_set = set(y_train_set)
#         unique_classes_list = list(unique_classes_set)
#         for label in unique_classes_list:
#             y_temp = list()
#             for cl in y_train_set:
#                 if cl !=label:
#                     y_temp.append(0)
#                 else:
#                     y_temp.append(1)
#             print("++++++++++++C:++++++++++++++", c)
#             classifier_dict[label] = svm.SVC(kernel='linear', C=c, max_iter=mi)
#             classifier_dict[label].fit(x_train_set, y_temp)
#             print("label for which svm is trained: ", label)
#
#         predicted = predict(x_test_set, classifier_dict)
#
#         acc = accuracy_score(y_test_set, predicted)
#         # print("for fold: ", k, " accuracy: ", acc)
#         mean_acc += acc
#
#     mean_acc /= num_folds
#     print(mean_acc)
#     mean_acc /= num_folds
#     accuracy_list.append(mean_acc)
#     print("c and wt:", param_comb, "accuracy:", mean_acc)
#     print("Mean accuracy for 5 fold validation: ", mean_acc)

# ----------------------Comment this code above while plotting confusion matrix---------------------

# ==========================================================================================
# Plot confusion matrix:
def confusion_matrix(test_y, predicted_array, classes):
    classes_len = len(classes)
    cm = list()

    print("classes are: ", classes)
    for cls in classes:
        class_list = [0] * classes_len
        for ind in range(0, len(test_y)):
            if test_y[ind] == cls:
                j = int(predicted_array[ind])
                class_list[j] += 1

        cm.append(class_list)

    print("confusion matrix is: ", cm)
    return cm

def maximum(matrix):
    i = -1
    return max(enumerate(map(itemgetter(i), matrix)), key=itemgetter(1))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = maximum(cm)[1] / 2.
    for i, j in itertools.product(range(len(cm[0])), range(len(cm[1]))):
        plt.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i][j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Divided data into train and test set
x_train_cm, x_test_cm = partition_data(X, 0, len(X)/4)
y_train_cm, y_test_cm = partition_data(labels, 0, len(y)/4)

clf_dict = dict()
unique_classes_set = set(y_train_cm)
unique_classes_list = list(unique_classes_set)
for label in unique_classes_list:
    y_temp = list()
    for c in y_train_cm:
        if c !=label:
            y_temp.append(0)
        else:
            y_temp.append(1)

    clf_dict[label] = svm.SVC(kernel='linear')
    clf_dict[label].fit(x_train_cm, y_temp)
    print("label for which svm is trained: ", label)

predicted = predict(x_test_cm, clf_dict)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_cm, predicted, unique_classes_list)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=unique_classes_list,
                      title='Confusion matrix, without normalization')
plt.savefig('ConfusionMatricesAndROC/part_C_confusionmatrix_linearKernel')
plt.show()












