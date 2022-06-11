import pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

image_base = tf.keras.datasets.fashion_mnist

(train_set, train_lebel), (test_set, test_lebel) = image_base.load_data()

test_set = test_set / 255.0
print(test_set[1].shape)

class_index = model.predict(test_set[:100])
print(class_index)

pred = np.argmax(class_index, axis=1)
accu = np.amax(class_index, axis=1)
data_set = {}
for i in range(10):
    data_set[i] = []

for i in range(100):
    data_set[pred[i]].append(accu[i])
for key in data_set:
    data_set[key] = sum(data_set[key]) / len(data_set[key]) * 100

fig = plt.figure()
plt.xlabel("Type of cloth")
plt.ylabel("Accuracy[%]")
plt.xticks(range(10))
plt.bar(data_set.keys(), data_set.values())
plt.show()
