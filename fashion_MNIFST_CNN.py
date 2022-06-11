import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import pickle
import time

image_base = tf.keras.datasets.fashion_mnist

(train_set, train_lebel), (test_set, test_lebel) = image_base.load_data()

train_set = train_set / 255.0
test_set = test_set / 255.0

model_sets = [
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(units=10, activation="softmax"),
    ],
    [
        keras.layers.Conv2D(
            30,
            kernel_size=(3, 3),
            strides=2,
            activation="relu",
            input_shape=(28, 28, 1),
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(
            30, kernel_size=(3, 3), strides=2, activation="relu"
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(units=10, activation="softmax"),
    ],
    [
        keras.layers.Conv2D(
            32,
            (5, 5),
            input_shape=(28, 28, 1),
            activation="relu",
        ),
        keras.layers.Conv2D(32, (5, 5), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (5, 5), activation="relu"),
        keras.layers.Conv2D(64, (5, 5), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(units=10, activation="softmax"),
    ],
]

model = tf.keras.models.Sequential(model_sets[2])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
batch_size = 50
epoch_value = 50
step_per_epoch = 700

start_time = time.time()
history = model.fit(train_set, train_lebel, epochs=30)
print(f"Learning time: {time.time() - start_time}")
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
print(score)

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
