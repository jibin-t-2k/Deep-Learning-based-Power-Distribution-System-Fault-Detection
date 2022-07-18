from google.colab import drive
drive.mount('/content/drive')


import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from keras.layers import Layer, Input, Reshape,Rescaling, Flatten, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score
from keras import backend as K

from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef

import gc


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

X = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_features.npy")
y = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_features_y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)

y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


def build_dnn():
    input_sig = Input(shape=(600,))
    sig = input_sig/1355.6187
    sig = Reshape((600,))(sig)

    # sig = Dense(768,activation ="relu")(sig)
    # sig = Dropout(0.2)(sig)
    # sig = Dense(896,activation ="relu")(sig)
    # sig = Dropout(0.2)(sig)
    sig = Dense(1024,activation ="relu")(sig)
    sig = Dropout(0.4)(sig)
    # sig = Dense(768,activation ="relu")(sig)
    # sig = Dropout(0.2)(sig)

    sig = Flatten()(sig)

    typ = Dense(256, activation="relu")(sig)
    typ = Dropout(0.2)(typ)
    typ = Dense(128, activation="relu")(sig)
    # typ = Dense(32, activation="relu")(typ)
    typ = Dropout(0.2)(typ)
    typ_output = Dense(23, activation="softmax", name="type")(typ)

    loc = Dense(256, activation="relu")(sig)
    loc = Dropout(0.2)(loc)
    loc = Dense(128, activation="relu")(sig)
    # loc = Dense(32, activation="relu")(loc)
    loc = Dropout(0.2)(loc)
    loc_output = Dense(15, activation="softmax", name="loc")(loc)

    model = Model(inputs=input_sig, outputs=[typ_output, loc_output])

    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], 
                  optimizer = Adam(learning_rate=0.0004),
                  metrics={"type":[
                                    F1Score(num_classes=23, name='f1_score'),
                                    MatthewsCorrelationCoefficient(num_classes=23, name ="mcc"),
                                    CategoricalAccuracy(name="acc")
                                  ],
                           "loc":[
                                    F1Score(num_classes=15, name='f1_score'),
                                    MatthewsCorrelationCoefficient(num_classes=15, name ="mcc"),
                                    CategoricalAccuracy(name="acc")
                                 ]})

    model._name = "DNN_Model"

    return model


with strategy.scope():
    dnn_model = build_dnn()

dnn_model.summary()

history = dnn_model.fit(X_train,
                        [y_train[:,:23], y_train[:,23:]],
                        epochs = 100,
                        batch_size = 32 * strategy.num_replicas_in_sync,
                        validation_data = (X_test, [y_test[:,:23], y_test[:,23:]]),
                        validation_batch_size = 32 * strategy.num_replicas_in_sync,
                        verbose = 1,
                        callbacks = [ModelCheckpoint("dnn_fault_detr_v1.h5",
                                                        verbose = 1,
                                                        monitor = "val_loss",
                                                        save_best_only = True,
                                                        save_weights_only = True,
                                                        mode = "min")])


np.save("dnn_fault_detr_v1_history.npy", history.history)
dnn_model_history = np.load("dnn_fault_detr_v1_history.npy", allow_pickle="TRUE").item()

dnn_model.load_weights("dnn_fault_detr_v1.h5")

test_metrics = dnn_model.evaluate(X_test, [y_test[:,:23], y_test[:,23:]])
test_metrics

plt.rcParams.update({'legend.fontsize': 14,
                    'axes.labelsize': 18, 
                    'axes.titlesize': 18,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18})

type_names =["No_Fault", "AG", "BG", "CG", "AB", "BC", "AC", "ABG", "BCG", "ACG", "ABC", "ABCG", "HIFA", "HIFB", "HIFC",
                   "Capacitor_Switch", "Linear_Load_Switch", "Non_Linear_Load_Switch", "Transformer_Switch",
                 "DG_Switch", "Feeder_Switch", "Insulator_Leakage", "Transformer_Inrush"]

loc_names = ["No Loc", "Loc 1", "Loc 2", "Loc 3", "Loc 4", "Loc 5", "Loc 6", "Loc 7", "Loc 8", "Loc 9", "Loc 10", "Loc 11", "Loc 12", "Loc 13", "Loc 14"]                    

def test_eval(model, dnn_model_history):

    print("\nTesting ")
    train_curves(dnn_model_history, model._name.replace("_"," "))
    
    pred_probas = model.predict(X_test, verbose = 1)

    y_type = np.argmax(y_test[:,0:23], axis = 1)
    y_loc = np.argmax(y_test[:,23:], axis = 1)

    pred_type = np.argmax(pred_probas[0], axis = 1)
    pred_loc = np.argmax(pred_probas[1], axis = 1)

    ###################################################################################################################

    print("\nClassification Report: Fault Type ")
    print(classification_report(y_type, pred_type, target_names = type_names, digits=6))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_type, pred_type))

    print("\nConfusion Matrix: Fault Type ")
    conf_matrix = confusion_matrix(y_type, pred_type)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = type_names, title = model._name.replace("_"," ") + " Fault Type")

    print("\nROC Curve: Fault Type")
    plot_roc(y_test[:,:23], pred_probas[0], class_names = type_names, title = model._name.replace("_"," ") +" Fault Type")

    ###################################################################################################################

    print("\nClassification Report: Fault Location ")
    print(classification_report(y_loc, pred_loc, target_names = loc_names, digits=6))
    print("Matthews Correlation Coefficient: ", matthews_corrcoef(y_loc, pred_loc))

    print("\nConfusion Matrix: Fault Location ")
    conf_matrix = confusion_matrix(y_loc, pred_loc)
    test_accuracy = plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = loc_names, title = model._name.replace("_"," ") + " Fault Location")

    print("\nROC Curve: Fault Location")
    plot_roc(y_test[:,23:], pred_probas[1], class_names = loc_names, title = model._name.replace("_"," ") +" Fault Location")


#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()

test_eval(dnn_model, dnn_model_history)