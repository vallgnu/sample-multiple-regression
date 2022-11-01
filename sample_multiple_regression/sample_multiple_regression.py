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

dataset = raw_dframe.copy()
drop_col = ['time', 'e_thick', 'd_thick', 'min_ch_no', 'max_ch_no', 'coil_no', 'grade_no', 'total_pass_no']


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


train_stats = train_dataset.describe()
train_stats.pop("ave_spvr")
train_stats = train_stats.transpose()
print(train_stats)


name = train_dataset.columns[26]
print(name)

train_labels = train_dataset.pop(name)
test_labels = test_dataset.pop(name)
print(train_labels)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = normed_train_data.dropna(how='all', axis=1)
normed_test_data = normed_test_data.dropna(how='all', axis=1)
normed_train_data.describe()


def build_model():
    model = keras.models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=[len(normed_train_data.keys())]))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

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
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


#plot_history(history)



model = build_model()
EPOCHS = 500

# patience は改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.1, verbose=1, callbacks=[early_stop])

plot_history(history)



loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG", format(mae))



test_predictions = model.predict(normed_test_data).flatten()
test_predictions


plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Value [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.plot([190, 250], [190, 250])


error = test_predictions - test_labels
plt.hist(error, bins=100, histtype='step')
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.grid()
