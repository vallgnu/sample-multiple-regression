import pathlib
import glob

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

print(tf.__version__)


dataset_path = glob.glob("")
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
#dataset.to_csv('./test.csv')
dataset.head()


test_dataset = dataset.sample(frac=0.1, random_state=0)
dataset = dataset.drop(test_dataset.index)
val_dataset = dataset.sample(frac=0.1, random_state=0)
train_dataset = dataset.drop(val_dataset.index)


train_stats = train_dataset.describe()
train_stats.pop("ave_spvr")
train_stats = train_stats.transpose()
print(train_stats)


name = train_dataset.columns[26]
#print(name)

train_labels = train_dataset.pop(name)
val_labels = val_dataset.pop(name)
test_labels = test_dataset.pop(name)
print(train_labels)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_val_data = norm(val_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = normed_train_data.dropna(how='all', axis=1)
normed_val_data = normed_val_data.dropna(how='all', axis=1)
normed_test_data = normed_test_data.dropna(how='all', axis=1)


normed_train_data.describe()


x_train_f32 = normed_train_data.astype('float32')
y_train_f32 = train_labels.astype('float32')
x_val_f32 = normed_val_data.astype('float32')
y_val_f32 = val_labels.astype('float32')
x_test_f32 = normed_test_data.astype('float32')
y_test_f32 = test_labels.astype('float32')


train_sliced = tf.data.Dataset.from_tensor_slices((x_train_f32, y_train_f32))
val_sliced = tf.data.Dataset.from_tensor_slices((x_val_f32, y_val_f32))

train_dataset = train_sliced.shuffle(250).batch(50)
val_dataset = val_sliced.batch(50)


activation1 = layers.Activation('relu', name='activation1')
activation2 = layers.Activation('relu', name='activation2')
acti_out = layers.Activation('relu', name='acti_out')

class build_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(build_model, self).__init__(*args, **kwargs)

        self.layer_1 = layers.Dense(name='layer1', units=64)
        self.layer_2 = layers.Dense(name='layer2', units=64)
        self.layer_out = layers.Dense(name='layer_out', units=1)

    def call(self, inputs, training=None):
        x1 = activation1(self.layer_1(inputs))
        x2 = activation2(self.layer_2(x1))
        outputs = acti_out(self.layer_out(x2))
        return outputs

model = build_model(name="multiple_regression")
print(model)


model1 = build_model(name='subclassing_model1')
model1.build(input_shape=(None, 36))
model1.summary()

"""tf.keras.utils.plot_model(model1, show_shapes=True, show_layer_names=True,
                         to_file='model_png')
from IPython.display import Image
Image(retina=False, filename='model.png')"""


import tensorflow.keras.backend as K

# カスタムの評価関数を実装（TensorFlow低水準API利用）
# （tf.keras.metrics.binary_accuracy()の代わり）
def tanh_accuracy(y_true, y_pred):           # y_trueは正解、y_predは予測（出力）
    threshold = K.cast(0.0, y_pred.dtype)              # -1か1かを分ける閾値を作成
    y_pred = K.cast(y_pred >= threshold, y_pred.dtype) # 閾値未満で0、以上で1に変換
    # 2倍して-1.0することで、0／1を-1.0／1.0にスケール変換して正解率を計算
    return K.mean(K.equal(y_true, y_pred * 2 - 1.0), axis=-1)

# カスタムの評価関数クラスを実装（サブクラス化）
# （tf.keras.metrics.BinaryAccuracy()の代わり）
class TanhAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name='tanh_accuracy', dtype=None):
        super(TanhAccuracy, self).__init__(name, dtype)

    # 正解率の状態を更新する際に呼び出される関数をカスタマイズ
    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = tanh_accuracy(y_true, y_pred)
        return super(TanhAccuracy, self).update_state(
            matches, sample_weight=sample_weight)


# 【初中級者向けとの比較用】学習方法を設定する
# model.compile(
#     ……最適化アルゴリズム……,
#     ……損失関数……,
#     [tanh_accuracy])   # 評価関数

# ###【エキスパート向け】最適化アルゴリズムを定義する ###
optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)  # 更新時の学習率

# ###【エキスパート向け】損失関数を定義する ###
criterion = tf.keras.losses.MeanSquaredError()

# ### 【エキスパート向け】評価関数を定義する ###
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = TanhAccuracy(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = TanhAccuracy(name='valid_accuracy')


import tensorflow.keras.backend as K

# ###【エキスパート向け】訓練する（1回分） ###
@tf.function
def train_step(train_X, train_y):
  # 訓練モードに設定
  training = True
  K.set_learning_phase(training)  # tf.keras内部にも伝える

  with tf.GradientTape() as tape: # 勾配をテープに記録
    # フォワードプロパゲーションで出力結果を取得
    #train_X                                   # 入力データ
    pred_y = model(train_X, training=training) # 出力結果
    #train_y                                   # 正解ラベル

    # 出力結果と正解ラベルから損失を計算し、勾配を求める
    loss = criterion(pred_y, train_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得

  # 逆伝播の処理として勾配を計算（自動微分）
  gradient = tape.gradient(loss, model.trainable_weights)

  # 勾配を使ってパラメーター（重みとバイアス）を更新
  optimizer.apply_gradients(zip(gradient, model.trainable_weights)) # 指定されたデータ分の最適化を実施

  # 損失と正解率を算出して保存
  train_loss(loss)
  train_accuracy(train_y, pred_y)

# ###【エキスパート向け】精度検証する（1回分） ###
@tf.function
def valid_step(valid_X, valid_y):
  # 評価モードに設定（※dropoutなどの挙動が評価用になる）
  training = False
  K.set_learning_phase(training)  # tf.keras内部にも伝える

  # フォワードプロパゲーションで出力結果を取得
  #valid_X                                   # 入力データ
  pred_y = model(valid_X, training=training) # 出力結果
  #valid_y                                   # 正解ラベル

  # 出力結果と正解ラベルから損失を計算
  loss = criterion(pred_y, valid_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
  # ※評価時は勾配を計算しない

  # 損失と正解率を算出して保存
  valid_loss(loss)
  valid_accuracy(valid_y, pred_y)


  # 【初中級者向けとの比較用】入力データを指定して学習する
# model.fit(
#     ……訓練データ（入力）……, ……同（ラベル）……,
#     ……精度検証データ……,
#     ……バッチサイズ……,
#     epochs=100,  # エポック数
#     verbose=1)   # 実行状況の出力モード

# ###【エキスパート向け】学習する ###

# 定数（学習／評価時に必要となるもの）
EPOCHS = 100             # エポック数： 100

# 損失の履歴を保存するための変数
train_history = []
valid_history = []

for epoch in range(EPOCHS):
  # エポックのたびに、メトリクスの値をリセット
  train_loss.reset_states()      # 「訓練」時における累計「損失値」
  train_accuracy.reset_states()  # 「訓練」時における累計「正解率」
  valid_loss.reset_states()      # 「評価」時における累計「損失値」
  valid_accuracy.reset_states()  # 「評価」時における累計「正解率」

  for train_X, train_y in train_dataset:
    # 【重要】1ミニバッチ分の「訓練（学習）」を実行
    train_step(train_X, train_y)

  for valid_X, valid_y in val_dataset:
    # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
    valid_step(valid_X, valid_y)

  # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
  n = epoch + 1                          # 処理済みのエポック数
  avg_loss = train_loss.result()         # 訓練用の平均損失値
  avg_acc = train_accuracy.result()      # 訓練用の平均正解率
  avg_val_loss = valid_loss.result()     # 訓練用の平均損失値
  avg_val_acc = valid_accuracy.result()  # 訓練用の平均正解率

  # グラフ描画のために損失の履歴を保存する
  train_history.append(avg_loss)
  valid_history.append(avg_val_loss)

  # 損失や正解率などの情報を表示
  print(f'[Epoch {n:3d}/{EPOCHS:3d}]' \
        f' loss: {avg_loss:.5f}, acc: {avg_acc:.5f}' \
        f' val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

print('Finished Training')
print(model.get_weights())  # 学習後のパラメーターの情報を表示


for train_X, train_y in train_dataset:
    print(train_X)
    print(train_y)


for val_X, val_y in val_dataset:
    print(val_X)
    print(val_y)


test_sliced = tf.data.Dataset.from_tensor_slices((x_test_f32, y_test_f32))

for test_X, test_y in test_sliced:
    print(test_X)
    print(test_y)


test_dataset = test_sliced.batch(1)
#test_dataset = test_sliced

for test_X, test_y in test_dataset:
    print(test_X)
    print(test_y)


record_predict = []
record_correct = []

for test_X, test_y in test_dataset:
    test_pre = model(test_X, training=False)
    record_predict.append(test_pre)
    record_correct.append(test_y)

print(record_predict)


record_predict[0] - record_correct[0]
