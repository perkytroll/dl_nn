import time
import numpy
import matplotlib.pyplot as mpl
import seaborn as sb
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

train_time_cv_lin = 0
test_time_cv_lin = 0
train_time_cv_rbf = 0
test_time_cv_rbf = 0


def svm_implementation(features, labels):
    clf_lin = SVC(kernel='linear', random_state=1)

    print("Cross validation for Linear Kernel begins.....")
    start_train_lin = time.time()
    c_range_lin = [0.1, 1, 10, 100, 1000]
    parameter_grid_lin = dict(C=c_range_lin)
    grid_lin = GridSearchCV(clf_lin, param_grid=parameter_grid_lin, cv=10, verbose=1)

    grid_lin.fit(features, labels)
    end_train_lin = time.time()
    global train_time_cv_lin
    train_time_cv_lin = end_train_lin - start_train_lin

    start_test_lin = time.time()
    predict_lin = grid_lin.predict(features)
    end_test_lin = time.time()
    global test_time_cv_lin
    test_time_cv_lin = end_test_lin - start_test_lin

    error_num_lin = sum(predict_lin != labels)
    error_pc_lin = (error_num_lin / labels.shape[0]) * 100
    print(
        "SVM implementation in sklearn for Linear kernel with cross validation and C value", grid_lin.best_params_['C'],
        "gives a model that yields an accuracy percentage of : \n", (100 - error_pc_lin), "\n")

    """
    RBF Kernel Cross-Validation
    """
    clf_rbf = SVC(kernel='rbf', random_state=1)
    print("Cross validation for RBF Kernel begins.....")
    start_train_rbf = time.time()
    c_range_rbf = [0.1, 1, 10, 100, 1000]
    gamma_rbf = [0.1, 1, 10, 100]
    parameter_grid_rbf = dict(C=c_range_rbf, gamma=gamma_rbf)
    grid_rbf = GridSearchCV(clf_rbf, param_grid=parameter_grid_rbf, cv=10, verbose=1)
    grid_rbf.fit(features, labels)
    end_train_rbf = time.time()
    global train_time_cv_rbf
    train_time_cv_rbf = end_train_rbf - start_train_rbf

    start_test_rbf = time.time()
    predict_rbf = grid_rbf.predict(features)
    end_test_rbf = time.time()
    global test_time_cv_rbf
    test_time_cv_rbf = end_test_rbf - start_test_rbf

    error_num_rbf = sum(predict_rbf != labels)
    error_pc_rbf = (error_num_rbf / labels.shape[0]) * 100
    print(
        "SVM implementation in sklearn for RBF kernel with cross validation and C value", grid_rbf.best_params_['C'],
        ", gamma/spread value of, ", grid_rbf.best_params_['gamma'],
        "gives a model that yields an accuracy percentage of :"
        " \n", (100 - error_pc_rbf), "\n")

    if error_pc_rbf < error_pc_lin:
        result_recordings = confusion_matrix(labels, predict_rbf)
        print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", result_recordings)
        print("\nF1 - Score for the given class predictions \n", f1_score(labels, predict_rbf, average=None))
    else:
        result_recordings = confusion_matrix(labels, predict_lin)
        print("\nCONFUSION MATRIX FOR THE TEST DATA: \n", result_recordings)
        print("\nF1 - Score for the given class predictions \n", f1_score(labels, predict_lin, average=None))


def feature_label_extractor():
    """

    :return:
    """
    data_frame = numpy.genfromtxt("iris.data", delimiter=',', encoding="utf-8-sig", dtype="U")
    features = numpy.array([list(map(float, i)) for i in data_frame[:, 0:(data_frame.shape[1] - 1)]])
    labels = data_frame[:, data_frame.shape[1] - 1]
    return features, labels


def data_normalization(features):
    """

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
    svm_implementation(X, y)
    print("\nTraining time of cross validated model (linear kernel) is \n", train_time_cv_lin,
          "\nTesting time of cross validated model (linear kernel) is \n", test_time_cv_lin)
    print("\nTraining time of cross validated model (rbf kernel) is \n", train_time_cv_rbf,
          "\nTesting time of cross validated model (rbf kernel) is \n", test_time_cv_rbf)