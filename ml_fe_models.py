from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, matthews_corrcoef
from scipy import interp
from itertools import cycle, product


X = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_features.npy")
y = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_features_y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


type_names =["No_Fault", "AG", "BG", "CG", "AB", "BC", "AC", "ABG", "BCG", "ACG", "ABC", "ABCG", "HIFA", "HIFB", "HIFC",
                   "Capacitor_Switch", "Linear_Load_Switch", "Non_Linear_Load_Switch", "Transformer_Switch",
                 "DG_Switch", "Feeder_Switch", "Insulator_Leakage", "Transformer_Inrush"]

loc_names = ["No_Loc", "Loc1", "Loc2", "Loc3", "Loc4", "Loc5", "Loc6", "Loc7", "Loc8", "Loc9", "Loc10", "Loc11", "Loc12", "Loc13", "Loc14"]


plt.rcParams.update({'legend.fontsize': 14,
                    'axes.labelsize': 18, 
                    'axes.titlesize': 18,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18})

def plot_confusion_matrix(cm, target_names, title="Confusion matrix", cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.04)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, ha="right")
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize = 15,
                     weight='bold',
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize = 15,
                     weight='bold',
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label\naccuracy={:0.5f}; misclass={:0.5f}".format(accuracy, misclass))
    plt.show()

    return accuracy


def plot_roc(val_gts, pred_probas, class_names, title):
    # Plot linewidth.
    lw = 2
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(val_gts[:, i], pred_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(val_gts.ravel(), pred_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print()

    # Plot all ROC curves
    plt.figure(1, figsize=(17, 17))
    plt.plot(fpr["micro"], tpr["micro"],
            label="micro-average ROC curve (AUC = {0:0.6f})"
                    "".format(roc_auc["micro"]),
            color="deeppink", linestyle=":", linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label="macro-average ROC curve (AUC = {0:0.6f})"
                    "".format(roc_auc["macro"]),
            color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "grey", "deeppink", "tan", "sienna", "peru", "royalblue", "lightseagreen", "chocolate", "lightgreen", "yellow", "darkgray", "khaki", "plum", "teal", "crimson", "forestgreen", "slategray", "slateblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label="ROC curve of class {0} (AUC = {1:0.6f})"
                "".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    print()


def test_eval(model, model_name):

    print("\nTesting ")
    
    pred_probas = model.predict(X_test)

    y_type = np.argmax(y_test[:,:23], axis = 1)
    y_loc = np.argmax(y_test[:,23:], axis = 1)

    pred_type = np.argmax(pred_probas[:,:23], axis = 1)
    pred_loc = np.argmax(pred_probas[:,23:], axis = 1)

    ###################################################################################################################

    print("\nClassification Report: Fault Type ")
    print(classification_report(y_type, pred_type, target_names = type_names, digits=6))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_type, pred_type))

    print("\nConfusion Matrix: Fault Type ")
    conf_matrix = confusion_matrix(y_type, pred_type)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = type_names, title = model_name + " Fault Type")

    print("\nROC Curve: Fault Type")
    plot_roc(y_test[:,:23], pred_probas[:,:23], class_names = type_names, title = model_name +" Fault Type")

    ###################################################################################################################

    print("\nClassification Report: Fault Location ")
    print(classification_report(y_loc, pred_loc, target_names = loc_names, digits=6))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_loc, pred_loc))

    print("\nConfusion Matrix: Fault Location ")
    conf_matrix = confusion_matrix(y_loc, pred_loc)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = loc_names, title = model_name + " Fault Location")

    print("\nROC Curve: Fault Location")
    plot_roc(y_test[:,23:], pred_probas[:,23:], class_names = loc_names, title = model_name +" Fault Location")


logistic_reg_model = MultiOutputClassifier(LogisticRegression(), n_jobs=-1).fit(X_train, y_train)
test_eval(logistic_reg_model, "Logistic Regression Model")


svc_model = MultiOutputClassifier(SVC(), n_jobs=-1).fit(X_train, y_train)
test_eval(svc_model, "Support Vector Model")


decision_tree_model = DecisionTreeClassifier().fit(X_train, y_train)
test_eval(decision_tree_model, "Decision Tree Model")


random_forest_model = RandomForestClassifier().fit(X_train, y_train)
test_eval(random_forest_model, "Random Forest Model")


ada_boost_model = MultiOutputClassifier(AdaBoostClassifier(), n_jobs=-1).fit(X_train, y_train)
test_eval(ada_boost_model, "Ada-Boost Model")


bagging_model = MultiOutputClassifier(BaggingClassifier(), n_jobs=-1).fit(X_train, y_train)
test_eval(bagging_model, "Bagging Model")


xg_boost_model = MultiOutputClassifier(XGBClassifier(), n_jobs=-1).fit(X_train, y_train)
test_eval(xg_boost_model, "XG Boost Model")


knn_model = KNeighborsClassifier().fit(X_train, y_train)
test_eval(knn_model, "KNN Model")