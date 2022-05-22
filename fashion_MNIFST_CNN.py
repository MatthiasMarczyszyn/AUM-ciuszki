import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import pickle

image_base = tf.keras.datasets.fashion_mnist

(train_set,train_lebel),(test_set,test_lebel) = image_base.load_data()

# normalizetion of data sets (changing pixel values from 0-255 to 0-1)
train_set = train_set/255
test_set = test_set/255

def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(units=24, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = cnn_model()

batch_size = 50
epoch_value = 50
step_per_epoch = 700

history = model.fit(train_set,train_lebel,epochs=30)

plt.figure(1)
plt.plot(history.history["loss"])
plt.legend(["training"])
plt.title("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history["accuracy"])
plt.legend(["training"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.show()

score = model.evaluate(test_set, test_lebel, verbose=0)
print(score[0])
print(score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()