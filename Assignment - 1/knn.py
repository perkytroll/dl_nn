""" knn.py -
Implementation of K Nearest neighbors using Sklearn
"""
import time
import numpy
import matplotlib.pyplot as mpl
import seaborn as sb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

train_time_cv = 0
test_time_cv = 0


def knn_cv_eval(features, labels, kne_cval):
    """
    This function evaluates classifier performance when the model has been cross - validated
    :param features: training set data
    :param labels: training set labels
    :param kne_cval: cross - validated model's instance
    :return: model instance after cross validation and hyperparameter training has been performed
    """
    print("Cross Validation begins.....")
    hyperparameter_grid = {
        'n_neighbors': range(1, 31),
        'p': [1, 2]
    }
    g_clf = GridSearchCV(kne_cval, hyperparameter_grid, cv=10, verbose=1)
    g_clf.fit(features, labels)
    if g_clf.best_params_['p'] == 1:
        print(
            "Power parameter comes out to be 1, which mean Manhattan distance is the best while calculating distances\n"
            "Optimal value of number of neighbors to check is", g_clf.best_params_['n_neighbors'])
    else:
        print(
            "Power parameter comes out to be 2, which mean Euclidean distance is the best while calculating distances\n"
            "Optimal value of number of neighbors to check is", g_clf.best_params_['n_neighbors'])
    return g_clf


def knn_implementation(features, labels):
    """
    This function performs K Nearest Neighbor classification
    :param features:
    :param labels:
    :return:
    """
    knn_cv = KNeighborsClassifier()
    start_train_time_cv = time.time()
    final_model = knn_cv_eval(features, labels, knn_cv)
    end_train_time_cv = time.time()
    global train_time_cv
    train_time_cv = end_train_time_cv - start_train_time_cv

    start_test_time_cv = time.time()
    prediction = final_model.predict(features)
    end_test_time_cv = time.time()
    global test_time_cv
    test_time_cv = end_test_time_cv - start_test_time_cv

    error_num = sum(prediction != labels)
    error_pc = (error_num / labels.shape[0]) * 100
    print("\nFinal accuracy that the cross validated model of KNN classifier yields is : \n", (100 - error_pc))

    """
    - Calculation of Confusion Matrix on the predictions
    """
    result_recordings = confusion_matrix(labels, prediction)
    print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", result_recordings)

    """
    - Calculation of F1 Score for the given predictions
    """
    print("\nF1 - Score for the given class predictions \n", f1_score(labels, prediction, average=None))


def feature_label_extractor():
    """
    Extracts features and labels from the data
    :return:
    """
    data_frame = numpy.genfromtxt("iris.data", delimiter=',', encoding="utf-8-sig", dtype="U")
    features = numpy.array([list(map(float, i)) for i in data_frame[:, 0:(data_frame.shape[1] - 1)]])
    labels = data_frame[:, data_frame.shape[1] - 1]
    return features, labels


def data_normalization(features):
    """
    Normalize data to scale and take units out of equation
    :param features:
    :return:
    """
    tp = features.T.copy()
    for i in range(tp.shape[0]):
        tp[i] = tp[i] - tp[i].mean()
        tp[i] = tp[i] / tp[i].std()
    features = tp.T
    return features


def data_plot(features, labels):
    """
    Plotting data to visualize
    :param features:
    :param labels:
    :return:
    """
    u_labels = numpy.unique(labels)
    my_cmap = ListedColormap(sb.color_palette('bright', u_labels.size).as_hex())

    color_dict = {}
    for mapping in range(u_labels.size):
        color_dict[u_labels[mapping]] = my_cmap.colors[mapping]
    colors = [color_dict[label] if label in color_dict else 'Does not Exist' for label in labels]

    mpl.scatter(features[:, 0], features[:, 1], c=colors, alpha=0.5)

    mpl.title('2-D Scatter plot for data in the direction of maximum variances', fontsize=12)
    mpl.xlabel('First Dimension of Score Space', fontsize=10)
    mpl.ylabel("Second Dimension of Score Space", fontsize=10)
    mpl.show()


def data_visualisation(features, labels):
    """
    Setting up data in score space to help visualize
    :param features:
    :param labels:
    :return:
    """
    f_mean = numpy.mean(features, axis=0)
    f_mc = features - f_mean
    cov_matrix = (f_mc.T @ f_mc) / (f_mc.shape[1] - 1)

    eigen_values, eigen_vectors = numpy.linalg.eig(cov_matrix)
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, idx]

    score_space = features @ eigen_vectors
    data_plot(score_space, labels)


if __name__ == '__main__':
    X, y = feature_label_extractor()
    X = data_normalization(X)
    data_visualisation(X, y)
    knn_implementation(X, y)
    print("\nTraining Time for the data set is\n", train_time_cv)
    print("\nTesting Time for the data set is\n", test_time_cv)
