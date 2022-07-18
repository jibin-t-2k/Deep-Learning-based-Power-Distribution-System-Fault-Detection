from google.colab import drive
drive.mount('/content/drive')


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import gc
import shutil
from sklearn.model_selection import train_test_split

fault_tags = ["No_Fault", "AG", "BG", "CG", "AB", "BC", "AC", "ABG", "BCG", "ACG", "ABC", "ABCG", "HIFA", "HIFB", "HIFC",
                "Capacitor_Switch", "Linear_Load_Switch", "Non_Linear_Load_Switch", "Transformer_Switch",
                "DG_Switch", "Feeder_Switch", "Insulator_Leakage", "Transformer_Inrush"]


def process_fault_tag(fault_tag):
    tags = fault_tag.split("_")
    typ = [0]*23
    typ[int(tags[1])] = 1
    loc = [0]*15
    loc[int(tags[2])] = 1
    gt = typ + loc
    return gt


def process_data():
    X = []
    y = []
    siz = 12800
    current_headers = ["I1_(1)", "I1_(2)", "I1_(3)", "I2_(1)", "I2_(2)", "I2_(3)", "I3_(1)", "I3_(2)", "I3_(3)", "I4_(1)", "I4_(2)", "I4_(3)", "I5_(1)", "I5_(2)", "I5_(3)"]
 
    print("\nNo Fault")
    fault_bus_path = "/content/Excel_Data/No_Fault/"
    fault_file_name = "0_0_0.xlsx"
    fault_file_path = fault_bus_path + fault_file_name
    fault_signal = pd.read_excel(fault_file_path)
    signal_is = [fault_signal[z].values[:siz] for z in current_headers]
    X.append(np.stack(signal_is , axis=-1))

    fault_tag = "0_0_0"
    gt = process_fault_tag(fault_tag)
    y.append(gt)


    print("\nLG Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/LG/BUS{}/".format(i)
        for j in [1,2,3]:
            fault_file_name = "1_{}_{}.xlsx".format(j,i)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[:siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1))

            fault_tag = "1_{}_{}".format(j, i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nLL Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/LL/BUS{}/".format(i)
        for j in [4,5,6]:
            fault_file_name = "1_{}_{}.xlsx".format(j,i)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[:siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1)) 

            fault_tag = "1_{}_{}".format(j, i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nLLG Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/LLG/BUS{}/".format(i)
        for j in [7,8,9]:
            fault_file_name = "1_{}_{}.xlsx".format(j,i)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[:siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1)) 

            fault_tag = "1_{}_{}".format(j, i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nLLL Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/LLL/BUS{}/".format(i)
        j = 10
        fault_file_name = "1_{}_{}.xlsx".format(j,i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[:siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1)) 

        fault_tag = "1_10_{}".format(i)
        gt = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nLLLG Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/LLLG/BUS{}/".format(i)
        j = 11
        fault_file_name = "1_{}_{}.xlsx".format(j,i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[:siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1)) 

        fault_tag = "1_11_{}".format(i)
        gt = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nHIF Fault")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/HIF/BUS{}/".format(i)
        for j in [12,13,14]:
            fault_file_name = "1_{}_{}.xlsx".format(j,i)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[:siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1)) 

            fault_tag = "1_{}_{}".format(j, i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nCapacitor Switching")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Capacitor_Switching/BUS{}/".format(i)
        for j in ["A", "B", "C"]:
            fault_file_name = "BUS{}_{}.xlsx".format(i,j)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[siz:2*siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1))

            fault_tag = "1_15_{}".format(i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nLinear Load Switching")
    fault_bus_path = "/content/Excel_Data/Load_Switching"
    f_typ = 8
    for i in tqdm(range(1,15), position=0, leave=True):
        for j in ["A", "B", "C"]:
            for k in ["I", "R"]:
                fault_file_name = "BUS{}_{}_{}.xlsx".format(i,j,k)
                fault_file_path = fault_bus_path + "/BUS{}/".format(i) + fault_file_name
                fault_signal = pd.read_excel(fault_file_path)
                signal_is = [fault_signal[z].values[siz:2*siz] for z in current_headers]
                X.append(np.stack(signal_is , axis=-1))

                fault_tag = "1_16_{}".format(i)
                gt = process_fault_tag(fault_tag)
                y.append(gt)


    print("\nNon Linear Load Switching")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Non_Linear_Load_Switching/"
        fault_file_name = "BUS{}.xlsx".format(i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[2*siz:3*siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1))

        fault_tag = "1_17_{}".format(i)
        gt = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nTransformer Switching")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Transformer_Switching/"
        fault_file_name = "BUS{}.xlsx".format(i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[2*siz:3*siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1))

        fault_tag = "1_18_{}".format(i)
        gt3 = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nDG Switching")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/DG_Switching/"
        f_typ = 11
        fault_file_name = "BUS{}.xlsx".format(i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[2*siz:3*siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1))

        fault_tag = "1_19_{}".format(i)
        gt = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nFeeder Switching")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Feeder_Switching/"
        fault_file_name = "BUS{}.xlsx".format(i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[2*siz:3*siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1))

        fault_tag = "1_20_{}".format(i)
        gt = process_fault_tag(fault_tag)
        y.append(gt)


    print("\nInsulator Leakage")
    for i in tqdm(range(1,15), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Insulator_Leakage/BUS{}/".format(i)
        for j in ["A", "B", "C"]:
            fault_file_name = "BUS{}_{}.xlsx".format(i,j)
            fault_file_path = fault_bus_path + fault_file_name
            fault_signal = pd.read_excel(fault_file_path)
            signal_is = [fault_signal[z].values[:siz] for z in current_headers]
            X.append(np.stack(signal_is , axis=-1))

            fault_tag = "1_21_{}".format(i)
            gt = process_fault_tag(fault_tag)
            y.append(gt)


    print("\nTransformer Inrush")
    for i in tqdm(range(1,5), position=0, leave=True):
        fault_bus_path = "/content/Excel_Data/Transformer_Inrush/"
        f_typ = 14
        fault_file_name = "T{}.xlsx".format(i)
        fault_file_path = fault_bus_path + fault_file_name
        fault_signal = pd.read_excel(fault_file_path)
        signal_is = [fault_signal[z].values[:siz] for z in current_headers]
        X.append(np.stack(signal_is , axis=-1))

        fault_tag = "1_22_0"
        gt = process_fault_tag(fault_tag)
        y.append(gt)

    X =  np.array(X, dtype = np.float32)
    y =  np.array(y)

    return X, y


X, y = process_data()

print("No. of No Fault: \t", len(y[y[:,0]==1]))
print("No. of AG Fault: \t", len(y[y[:,1]==1]))
print("No. of BG Fault: \t", len(y[y[:,2]==1]))
print("No. of CG Fault: \t", len(y[y[:,3]==1]))
print("No. of AB Fault: \t", len(y[y[:,4]==1]))
print("No. of BC Fault: \t", len(y[y[:,5]==1]))
print("No. of AC Fault: \t", len(y[y[:,6]==1]))
print("No. of ABG Fault: \t", len(y[y[:,7]==1]))
print("No. of BCG Fault: \t", len(y[y[:,8]==1]))
print("No. of ACG Fault: \t", len(y[y[:,9]==1]))
print("No. of ABC Fault: \t", len(y[y[:,10]==1]))
print("No. of ABCG Fault: \t", len(y[y[:,11]==1]))
print("No. of HIF A Fault: \t", len(y[y[:,12]==1]))
print("No. of HIF B Fault: \t", len(y[y[:,13]==1]))
print("No. of HIF C Fault: \t", len(y[y[:,14]==1]))
print("No. of Capacitor Switching: \t", len(y[y[:,15]==1]))
print("No. of Linear Load Switching: \t", len(y[y[:,16]==1]))
print("No. of Non Linear Load Switching: \t", len(y[y[:,17]==1]))
print("No. of Transformer Switching: \t", len(y[y[:,18]==1]))
print("No. of DG Switching: \t", len(y[y[:,19]==1]))
print("No. of Feeder Switch: \t", len(y[y[:,20]==1]))
print("No. of Insulator Leakage: \t", len(y[y[:,21]==1]))
print("No. of Transformer Inrush: \t", len(y[y[:,22]==1]))
print()
print("No. of No Loc: \t", len(y[y[:,23]==1]))
print("No. of Loc 1: \t", len(y[y[:,24]==1]))
print("No. of Loc 2: \t", len(y[y[:,25]==1]))
print("No. of Loc 3: \t", len(y[y[:,26]==1]))
print("No. of Loc 4: \t", len(y[y[:,27]==1]))
print("No. of Loc 5: \t", len(y[y[:,28]==1]))
print("No. of Loc 6: \t", len(y[y[:,29]==1]))
print("No. of Loc 7: \t", len(y[y[:,30]==1]))
print("No. of Loc 8: \t", len(y[y[:,31]==1]))
print("No. of Loc 9: \t", len(y[y[:,32]==1]))
print("No. of Loc 10: \t", len(y[y[:,33]==1]))
print("No. of Loc 11: \t", len(y[y[:,34]==1]))
print("No. of Loc 12: \t", len(y[y[:,35]==1]))
print("No. of Loc 13: \t", len(y[y[:,36]==1]))
print("No. of Loc 14: \t", len(y[y[:,37]==1]))


np.save("signals.npy", X)
np.save("signals_gts3.npy", y)


signals = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals.npy")
signals_gts = np.load("/content/drive/MyDrive/DS_Fault_Detection/Data/signals_gts1.npy")

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