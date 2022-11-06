import glob

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = glob.glob("./data/*")
dataset_path

raw_dframe = pd.read_csv(dataset_path[0])

min_ch_no = raw_dframe[raw_dframe.columns[3]][0]
max_ch_no = raw_dframe[raw_dframe.columns[4]][0]

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
dataset.to_csv('./test.csv')
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

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = normed_train_data.dropna(how='all', axis=1)
normed_test_data = normed_test_data.dropna(how='all', axis=1)

def build_model():
    model = keras.models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=[len(normed_train_data.keys()),]))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(len(train_labels.keys()), activation='linear'))

    #optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

    return model

model = build_model()

model.summary()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()


#plot_history(history)

model = build_model()
EPOCHS = 500
BATCH_SIZE = 64

# patience は改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

history = model.fit(x=normed_train_data,
                    y=train_labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split = 0.1,
                    verbose=1,
                    callbacks=[early_stop])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG", format(mae))

test_predictions = model.predict(normed_test_data).flatten()

test_labels_flatten = []

for i in test_labels.iloc:
    for j in range(len(i)):
        test_labels_flatten.append(i[j])


plt.figure()
plt.scatter(test_labels_flatten, test_predictions)
plt.xlabel('True Value [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.plot([190, 250], [190, 250])


error = test_predictions - test_labels_flatten
plt.hist(error, bins=100, histtype='step')
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.grid()
