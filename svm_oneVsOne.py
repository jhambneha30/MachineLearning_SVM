import numpy as np
from sklearn import svm
import h5py
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
import operator

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X, y = load_h5py('/home/nehaj/Desktop/Assignment2_ML/data/data_5.h5')

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


def max_frequency(class_list):
    label_votes_count = dict()
    # print("class list with classes: ", class_list)
    for label in class_list:
        if label not in label_votes_count:
            label_votes_count[label] = 1
        else:
            label_votes_count[label] += 1
    # print("label_votes_count dictionary: ", label_votes_count)
    final_label = max(label_votes_count.iteritems(), key=operator.itemgetter(1))[0]
    # print("final label is: ", final_label)
    return final_label


def predict(test_data, classifier_dictionary):
    # print("-----------------clf keys list-----------:", clf_keys_list)
    predictions = list()

    for point in test_data:
        predicted_class_list = list()
        ind = 0
        for lbls_key in classifier_dictionary.keys():
            # print("LABEL is:=========", lbls_key)
            decision_value = decision_function(point, classifier_dictionary.get(lbls_key))
            # print("decision val: ", decision_value)
            clf_class_list = list()
            l1 = int(lbls_key[0])
            l2 = int(lbls_key[2])
            clf_class_list.append(l1)
            clf_class_list.append(l2)
            # print("label key combination: ", lbls_key)
            # print("========================clf class list from string:============:", clf_class_list)
            if decision_value > 0:
                label_assigned = max(clf_class_list)
            else:
                label_assigned = min(clf_class_list)
            ind += 1
            predicted_class_list.append(label_assigned)
        final_label = max_frequency(predicted_class_list)
        predictions.append(final_label)

    return predictions


def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)

    return acc


def get_relevant_rows(x_data, y_data, lbl1, lbl2):
    x_tmp = list()
    y_tmp = list()
    # print("In relevant rows func, label1 and label2 are: ", label1, ", ", label2)
    for lbl_index in range(len(y_data)):
        y_lbl = y_data[lbl_index]
        if y_lbl == lbl1 or y_lbl == lbl2:
            y_tmp.append(y_lbl)
            x_tmp.append(x_data[lbl_index])

    return x_tmp, y_tmp



classifier_dict = {}
num_folds = 5
# Taking penalty C and max_iters as the parameters on which grid search is to be applied
accuracy_list = list()
C = [1, 500]
max_iter = [-1, 100, 1000]

# define the grid here
parameters = [(c, mi) for c in C
                 for mi in max_iter]

# do the grid search with k fold cross validation
# For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
graph_x_dt, graph_y_dt, param_comb_list = list(), list(), list()
index_dt = 1
for (c, mi) in parameters:
    mean_acc = 0
    param_comb = [c, mi]
    print("param comb is: ", param_comb)
    for k in xrange(0, num_folds):
        length = len(X)
        x_train_set, x_test_set = partition_data(X, k, length / num_folds)
        y_train_set, y_test_set = partition_data(y, k, length / num_folds)

        unique_classes_set = set(y_train_set)
        unique_classes_list = list(unique_classes_set)
        classifier_dict_ovo = dict()
        # classifier_keys_list = list()
        unique_classes_list.sort()
        index = 0
        for i in range(len(unique_classes_list)-1):
            p = i+1
            label1 = int(unique_classes_list[i])
            for j in range(p, len(unique_classes_list)):
                label2 = int(unique_classes_list[j])
                key_label = str(label1) + "," + str(label2)
                x_temp, y_temp = get_relevant_rows(x_train_set, y_train_set, label1, label2)
                classifier_dict_ovo[key_label] = svm.SVC(kernel='linear', C=c, max_iter=mi)
                classifier_dict_ovo[key_label].fit(x_temp, y_temp)
                index += 1
                print("label for which svm is trained: ", key_label)

        predicted = predict(x_test_set, classifier_dict_ovo)

        acc = accuracy_score(y_test_set, predicted)
        # print("for fold: ", k, " accuracy: ", acc)
        mean_acc += acc
    print(mean_acc)
    mean_acc /= num_folds
    accuracy_list.append(mean_acc)
    print("c and wt:", param_comb, "accuracy:", mean_acc)

    # print("Mean accuracy for 5 fold validation: ", mean_acc)










