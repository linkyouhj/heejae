import tensorflow as tf
import numpy as np

tf.random.set_seed(123)
tf.keras.backend.set_floatx('float32')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255


learning_rate = 0.01

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'), # (32, 32, 32)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # (32, 32, 32)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), # (16, 16, 32)
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # (16, 16, 64)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # (16, 16, 64)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), # (8, 8, 64)
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(), # (4096)
    tf.keras.layers.Dense(256, activation='relu'), # (256)
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'), # (64)
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax') # (10)
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

predict = model(x_test[:10]).numpy()
answer = y_test[:10, 0]

print("Predicts: ", np.argmax(predict, axis=1))
print("Answers: ", answer)