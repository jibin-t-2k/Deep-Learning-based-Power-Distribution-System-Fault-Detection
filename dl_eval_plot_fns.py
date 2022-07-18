import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle, product

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

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "grey", "indigo", "deeppink", "tan", "sienna", "peru", "royalblue", "lightseagreen", "chocolate", "lightgreen", "yellow", "darkgray", "khaki", "plum", "teal", "crimson", "forestgreen", "slategray", "slateblue"])
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

def train_curves(x_history, model_name):
    loss = x_history["loss"]
    val_loss = x_history["val_loss"]

    type_acc = x_history["type_acc"]
    val_type_acc = x_history["val_type_acc"]

    loc_acc = x_history["loc_acc"]
    val_loc_acc = x_history["val_loc_acc"]

    type_mcc = x_history["type_mcc"]
    val_type_mcc = x_history["val_type_mcc"]

    loc_mcc = x_history["loc_acc"]
    val_loc_mcc = x_history["val_loc_mcc"]

    plt.figure(figsize=(8, 6))
    plt.plot(loss[::2], label="Training Loss")
    plt.plot(val_loss[::2], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Loss")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name + " Loss Curves")
    plt.show()
    print()

    plt.figure(figsize=(8, 6))
    plt.plot(type_acc[::2], label="Training Type Accuracy")
    plt.plot(val_type_acc[::2], label="Validation Type Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Type Accuracy")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name + " Type Accuracy")
    plt.show()
    print()

    plt.figure(figsize=(8, 6))
    plt.plot(loc_acc[::2], label="Training Location Accuracy")
    plt.plot(val_loc_acc[::2], label="Validation Location Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Location Accuracy")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name + " Location Accuracy")
    plt.show()
    print()

    plt.figure(figsize=(8, 6))
    plt.plot(type_mcc[::2], label="Training Type MCC")
    plt.plot(val_type_mcc[::2], label="Validation Type MCC")
    plt.legend(loc="lower right")
    plt.ylabel("Type MCC")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name + " Type MCC")
    plt.show()
    print()

    plt.figure(figsize=(8, 6))
    plt.plot(loc_mcc[::2], label="Training Location MCC")
    plt.plot(val_loc_mcc[::2], label="Validation Location MCC")
    plt.legend(loc="lower right")
    plt.ylabel("Location MCC")
    plt.ylim([min(plt.ylim()),1])
    plt.title(model_name + " Location MCC")
    plt.show()


