import glob

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = glob.glob("./data/*")
dataset_path

raw_dframe = pd.read_csv(dataset_path[0])

min_ch_no = raw_dframe['min_ch_no'][0]
max_ch_no = raw_dframe['max_ch_no'][0]

dataset = raw_dframe.copy()
drop_col = ['time', 'e_thick', 'd_thick', 'min_ch_no', 'max_ch_no', 'coil_no', 'grade_no', 'total_pass_no', 'ave_spvr', 'for_learn_ave_spvr']

#削除する列名
for i in range(30):
    col_name = "op_man_flag[" + str(i) + "]"
    drop_col.append(col_name)

#削除する列名
for i in range(50):
    col_name = "add_gain_dl_output[" + str(i) + "]"
    drop_col.append(col_name)
    
dataset.drop(drop_col, axis=1, inplace=True)
#dataset.to_csv('./test.csv')
dataset.head()

train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#削除する列名
pop_col = []
for i in range(min_ch_no, max_ch_no+1):
    col_name = "for_learn_spvr[" + str(i) + "]"
    pop_col.append(col_name)

train_stats = train_dataset.describe()
for i in pop_col:
    train_stats.pop(i)
train_stats = train_stats.transpose()
print(train_stats)

train_labels = []
test_labels = []

for i in pop_col:
    train_labels.append(train_dataset.pop(i))
    test_labels.append(test_dataset.pop(i))

train_labels = pd.DataFrame(train_labels).transpose()
test_labels = pd.DataFrame(test_labels).transpose()


train_labels_dict = {}
test_labels_dict = {}
for i, j in enumerate(pop_col):
    dic_col_name = "spvr_ch" + str(i+min_ch_no)
    train_labels_dict[dic_col_name] = pd.DataFrame(train_labels[j])
    test_labels_dict[dic_col_name] = pd.DataFrame(test_labels[j])
   

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_train_data = normed_train_data.dropna(how='all', axis=1)
normed_test_data = normed_test_data.dropna(how='all', axis=1)

#活性化関数作成
activation_list = []
for i in range(min_ch_no, max_ch_no+1):
    pre_activation_list = []
    for j in range(2):
        acti_name = "activation_ch" + str(i) + "_" + str(j)        
        pre_activation_list.append(layers.Activation('relu', name=acti_name))
    activation_list.append(pre_activation_list)
    
#出力層の活性化関数
acti_out_list = []
for i in range(min_ch_no, max_ch_no+1):
    acti_out_name = "spvr_ch" + str(i)
    acti_out_list.append(layers.Activation('linear', name=acti_out_name))
    
#入力層の作成    
inputs = layers.Input(name='layer_in', shape=(len(normed_train_data.keys()),))

#各CHの中間層(1層目)
layer1_list = []
for i in range(min_ch_no, max_ch_no+1):
    layer1_name = "layer1_ch" + str(i)
    layer1_list.append(layers.Dense(name=layer1_name, units=32))
    
#各CHの中間層(2層目)
layer2_list = []
for i in range(min_ch_no, max_ch_no+1):
    layer2_name = "layer2_ch" + str(i)
    layer2_list.append(layers.Dense(name=layer2_name, units=32))
    
#各CHの出力層
layer_out_list = []
for i in range(min_ch_no, max_ch_no+1):
    layer_out_name = "layer_out_ch" + str(i)
    layer_out_list.append(layers.Dense(name=layer_out_name, units=1))

#forward propagation
x1_list = []
x2_list = []
outputs_list = []
for i in range(max_ch_no-min_ch_no+1):
    x1_list.append(activation_list[i][0](layer1_list[i](inputs)))
    x2_list.append(activation_list[i][1](layer2_list[i](x1_list[i])))
    outputs_list.append(acti_out_list[i](layer_out_list[i](x2_list[i])))
    
model = tf.keras.Model(inputs=inputs, outputs=outputs_list)

model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

model.summary()

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    train_spvr_mae = []
    train_spvr_mse = []
    val_spvr_mae = []
    val_spvr_mse = []
    
    for i in range(min_ch_no, max_ch_no+1):
        col_name = "spvr_ch" + str(i) + "_mae"
        train_spvr_mae.append(col_name)
    for i in range(min_ch_no, max_ch_no+1):
        col_name = "spvr_ch" + str(i) + "_mse"
        train_spvr_mse.append(col_name)
    for i in range(min_ch_no, max_ch_no+1):
        col_name = "val_spvr_ch" + str(i) + "_mae"
        val_spvr_mae.append(col_name)
    for i in range(min_ch_no, max_ch_no+1):
        col_name = "val_spvr_ch" + str(i) + "_mse"
        val_spvr_mse.append(col_name)
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [I-Unit]')
    for i, j in zip(train_spvr_mae, val_spvr_mae):
        plt.plot(hist['epoch'], hist[i], label='Train Error')
        plt.plot(hist['epoch'], hist[j], label = 'Val Error', linestyle='--')
    #plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [I-Unit^2]')
    for i, j in zip(train_spvr_mae, val_spvr_mae):
        plt.plot(hist['epoch'], hist[i], label='Train Error')
        plt.plot(hist['epoch'], hist[j], label = 'Val Error', linestyle='--')
    #plt.ylim([0,20])
    plt.legend()
    plt.show()
    #plot_history(history)
    
  EPOCHS = 500
BATCH_SIZE = 32

# patience は改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(x={'layer_in':normed_train_data}, 
                    y=train_labels_dict,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split = 0.1, 
                    verbose=2, 
                    callbacks=[early_stop])

plot_history(history)

param_list = model.evaluate(normed_test_data, test_labels_dict, verbose=2)
loss = param_list[0]
param_list.pop(0)
loss_list = []
for i in range(max_ch_no-min_ch_no+1):
    loss_list.append(param_list[i])
    
param_list = param_list[max_ch_no-min_ch_no+1:]
param_list = np.array(param_list).reshape(max_ch_no-min_ch_no+1, 2)


print("Loss of this model is {:5.2f} [I-Unit]".format(loss))

k = 0
for i, j in zip(loss_list, param_list):
    mae, mse = j
    print("CH{:d}, Loss:{:5.2f}, mae:{:5.2f}, mse:{:5.2f}".format(k+min_ch_no, i, mae, mse))
    k = k + 1
    
 
test_predictions = model.predict(normed_test_data)

fig_scat = plt.figure(figsize=(48, 16))
plt.subplots_adjust(wspace=0.8, hspace=0.8)
for i in range(max_ch_no-min_ch_no+1):
    dic_name = "spvr_ch" + str(i+min_ch_no)
    ax = fig_scat.add_subplot(5, 10, i+1)
    ax.scatter(test_predictions[i], test_labels_dict[dic_name], s=3, alpha=0.2)
    ax.set_title(dic_name, fontsize=8)
    ax.set_xlabel("pred [I-Unit]", fontsize=8)
    ax.set_ylabel("label [I-Unit]", fontsize=8)
    plt.tick_params(labelsize=5)
    ax.grid()
plt.show()

fig_hist = plt.figure(figsize=(48, 16))
plt.subplots_adjust(wspace=0.8, hspace=0.8)
for i in range(max_ch_no-min_ch_no+1):
    dic_name = "spvr_ch" + str(i+min_ch_no)
    ax = fig_hist.add_subplot(5, 10, i+1)
    error = test_predictions[i] - test_labels_dict[dic_name]
    ax.hist(error, bins=20, histtype='step')
    ax.set_title(dic_name, fontsize=8)
    ax.set_xlabel("pred error [I-Unit]", fontsize=8)
    ax.set_ylabel("counts [I-Unit]", fontsize=8)
    plt.tick_params(labelsize=5)
    ax.grid()
plt.show()

model.save('./saved_model/test_miltiple_regression_multiouts_FuncAPI')



