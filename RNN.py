import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

PATH = "/Users/munhaneul/PycharmProjects/Machine_learning_handson/data/"
data1 = pd.read_csv(PATH + "2023-4-14_162237.csv")




def generate_timeseries(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offset2) * (freq2 + 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_timeseries(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_val, y_val = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]


input_layer = tf.keras.Input(shape=(None, 1), name='InputLayer')



x1 = tf.keras.layers.SimpleRNN(20, return_sequences=True)(input_layer)
x2 = tf.keras.layers.SimpleRNN(20)(x1)
x3 = tf.keras.layers.Dense(1)(x2)
# 마지막 관심 대상인 layer에는 return_sequence 를 true로 설정하지 않는다.
# x2 = tf.keras.layers.Dense(1, name='OutputLayer')(x1)

func_model = tf.keras.Model(inputs=input_layer, outputs=x3, name='FunctionalModel')

func_model.summary()

func_model.compile(optimizer=tf.optimizers.Adam(),
              loss='mse',
              metrics=[tf.keras.metrics.Accuracy()])

func_model.fit(X_train, y_train, epochs=20)
print(func_model.evaluate(X_test, y_test))

# series = generate_timeseries(1, n_steps + 10)
# X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_val
for step_ahead in range(100):
    y_pred_one = func_model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]
print(Y_pred.shape)
print(Y_pred[0][:,:].shape)
a = Y_pred[0][:,:].reshape((1,100))
print(a)
b = X_val.reshape((1,-1))
print(b)

plt.figure(figsize=(20,10))
plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
plt.plot(a[0], 'b', label='Predict Data')
plt.plot(b[0][0:100], 'r', label='Actual Data')
# plt.title(target_name_2[0], fontsize=18)
plt.xlabel("Validation Data")
# plt.ylabel("km/h")
# plt.legend(loc='upper right')
plt.show()
# plt.savefig("./result/Test_17")
