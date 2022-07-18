from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import (Layer, Input, Reshape, Rescaling, Flatten, Dense, Dropout, TimeDistributed, Conv1D, 
                          Activation, LayerNormalization, Embedding, MultiHeadAttention, Lambda, Add)
                          
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score
from keras import backend as K

from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from 1d_deep_cnns import ResNet152V2, DenseNet121, InceptionResNetV2, Xception, ResNeXt50, EfficientNetB2

from sklearn.model_selection import train_test_split

import gc

from dl_eval_plot_fns import plot_confusion_matrix, plot_roc, train_curves


try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

print("Number of accelerators: ", strategy.num_replicas_in_sync)
print(tf.__version__)

signals = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals.npy", mmap_mode="r")
signals_gts = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_gts.npy", mmap_mode="r")

X = []
y = []

for signal, signal_gt in tqdm(zip(signals.astype(np.float32), signals_gts), position=0, leave=True):
    if any(signal_gt[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]]): # LG, LL, LLG, LLL, LLLG, HIF, Non_Linear_Load_Switch
        noise_count = 20
    elif any(signal_gt[[15, 21]]):  # Capacitor_Switch, Insulator_Leakage
        noise_count = 10
    elif signal_gt[16] == 1: # Load_Switch
        noise_count = 5
    elif signal_gt[22] == 1: # Transformer_Inrush
        noise_count = 30
    elif signal_gt[0] == 1: # No Fault
        noise_count = 100

    for n in range(noise_count):
        X.append(signal)
        y.append(signal_gt)
        
X = np.array(X)
np.random.seed(7)
for i in tqdm(range(X.shape[0])):
    noise = np.random.uniform(-5.0, 5.0, (12800, 15)).astype(np.float32)
    X[i] = X[i] + noise
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
del X, y, signals, signals_gts
gc.collect()

X = np.load("X.npy", mmap_mode="r")
y = np.load("y.npy", mmap_mode="r")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, shuffle = True, random_state = 77, stratify = y)
print(X_tr.shape, y_tr.shape)
print(X_te.shape, y_te.shape)

tr_shape = X_tr.shape[0]
X_train1 = tf.convert_to_tensor(X_tr[:(tr_shape//8)*1])
X_train2 = tf.convert_to_tensor(X_tr[(tr_shape//8)*1: (tr_shape//8)*2])
X_train3 = tf.convert_to_tensor(X_tr[(tr_shape//8)*2: (tr_shape//8)*3])
X_train4 = tf.convert_to_tensor(X_tr[(tr_shape//8)*3: (tr_shape//8)*4])
X_train5 = tf.convert_to_tensor(X_tr[(tr_shape//8)*4: (tr_shape//8)*5])
X_train6 = tf.convert_to_tensor(X_tr[(tr_shape//8)*5: (tr_shape//8)*6])
X_train7 = tf.convert_to_tensor(X_tr[(tr_shape//8)*6: (tr_shape//8)*7])
X_train8 = tf.convert_to_tensor(X_tr[(tr_shape//8)*7:])
X_train = tf.concat([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8], axis=0)

te_shape = X_te.shape[0]
X_test1 = tf.convert_to_tensor(X_te[: (te_shape//4)*1])
X_test2 = tf.convert_to_tensor(X_te[(te_shape//4)*1:(te_shape//4)*2])
X_test3 = tf.convert_to_tensor(X_te[(te_shape//4)*2:(te_shape//4)*3])
X_test4 = tf.convert_to_tensor(X_te[(te_shape//4)*3:])
X_test = tf.concat([X_test1, X_test2, X_test3, X_test4], axis=0)

y_train = tf.convert_to_tensor(y_tr)
y_test = tf.convert_to_tensor(y_te)


def AttentionEncoder(inputs, num_heads, head_size, dropout, units_dim):
    encode1 = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size , dropout=dropout
    )(encode1, encode1)
    encode2 = Add()([attention_output, encode1])
    encode3 = LayerNormalization(epsilon=1e-6)(encode2)
    for units in [units_dim * 2, units_dim]:
        encode3 = Dense(units=units, activation='relu')(encode3)
        encode3 = Dropout(dropout)(encode3)
    outputs = Add()([encode3, encode2])

    return outputs

def build_cnn_attention():
    input_sig = Input(shape=(12800, 15))
    sig = Rescaling(scale=1.0/6065.3965)(input_sig)
    sig = Reshape((50, 256, 15))(sig)

    fe = Xception(input_shape=(256,15), include_top=False, weights=None, pooling="avg")
    # fe = ResNet152V2(input_shape=(256,15), include_top=False, weights=None, pooling="avg")
    # fe = DenseNet121(input_shape=(256,15), include_top=False, weights=None, pooling="avg")
    # fe = InceptionResNetV2(input_shape=(256,15), include_top=False, weights=None, pooling="avg")
    sig = TimeDistributed(fe)(sig)

    embeddings = Embedding(input_dim=50, output_dim=64)
    position_embed = embeddings(tf.range(start=0, limit=50, delta=1))
    sig = sig + position_embed

    for e in range(2):
        sig = AttentionEncoder(sig, num_heads=4, head_size=128, dropout=0.2, units_dim=64)

    sig = Flatten()(sig)

    typ = Dense(512, activation="relu")(sig)
    typ = Dropout(0.2)(typ)
    typ = Dense(128, activation="relu")(typ)
    typ = Dense(32, activation="relu")(typ)
    typ = Dropout(0.2)(typ)
    typ_output = Dense(23, activation="softmax", name="type")(typ)

    loc = Dense(512, activation="relu")(sig)
    loc = Dropout(0.2)(loc)
    loc = Dense(128, activation="relu")(loc)
    loc = Dense(32, activation="relu")(loc)
    loc = Dropout(0.2)(loc)
    loc_output = Dense(15, activation="softmax", name="loc")(loc)

    model = Model(inputs=input_sig, outputs=[typ_output, loc_output])

    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], 
                  optimizer = Adam(learning_rate=0.001),
                  metrics={"type":[ 
                                    CategoricalAccuracy(name="acc"),
                                    MatthewsCorrelationCoefficient(num_classes=23, name ="mcc"),
                                    F1Score(num_classes=23, name='f1_score')
                                  ],
                           "loc":[
                                    CategoricalAccuracy(name="acc"),
                                    MatthewsCorrelationCoefficient(num_classes=15, name ="mcc"),
                                    F1Score(num_classes=15, name='f1_score')
                                 ]})

    model._name = "Deep_CNN_Attention_Model"

    return model


with strategy.scope():
    cnn_attention_model = build_cnn_attention()

cnn_attention_model.summary()

cnn_attention_history = cnn_attention_model.fit(X_train,
                                                    [y_train[:,:23], y_train[:,23:]],
                                                    epochs = 100,
                                                    batch_size = 32 * strategy.num_replicas_in_sync,
                                                    validation_data = (X_test, [y_test[:,:23], y_test[:,23:]]),
                                                    validation_batch_size = 32 * strategy.num_replicas_in_sync,
                                                    verbose = 1,
                                                    callbacks = [ModelCheckpoint("cnn_attention_fault_detr_v1.h5",
                                                                                    verbose = 1,
                                                                                    monitor = "val_loss",
                                                                                    save_best_only = True,
                                                                                    save_weights_only = True,
                                                                                    mode = "min")])

np.save("cnn_attention_fault_detr_v1_history.npy", cnn_attention_history.history)
cnn_attention_model_history = np.load("cnn_attention_fault_detr_v1_history.npy", allow_pickle="TRUE").item()

cnn_attention_model.load_weights("cnn_attention_fault_detr_v1.h5")

test_metrics = cnn_attention_model.evaluate(X_test, [y_test[:,:23], y_test[:,23:]])
test_metrics

type_names =["No_Fault", "AG", "BG", "CG", "AB", "BC", "AC", "ABG", "BCG", "ACG", "ABC", "ABCG", "HIFA", "HIFB", "HIFC",
                   "Capacitor_Switch", "Linear_Load_Switch", "Non_Linear_Load_Switch", "Transformer_Switch",
                 "DG_Switch", "Feeder_Switch", "Insulator_Leakage", "Transformer_Inrush"]
loc_names = ["No Loc", "Loc 1", "Loc 2", "Loc 3", "Loc 4", "Loc 5", "Loc 6", "Loc 7", "Loc 8", "Loc 9", "Loc 10", "Loc 11", "Loc 12", "Loc 13", "Loc 14"]


plt.rcParams.update({'legend.fontsize': 14,
                    'axes.labelsize': 18, 
                    'axes.titlesize': 18,
                    'xtick.labelsize': 18,
                    'ytick.labelsize': 18})
                    
def test_eval(model, history):

    print("\nTesting ")
    train_curves(history, model._name.replace("_"," "))
    
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

test_eval(cnn_attention_model, cnn_attention_model_history)