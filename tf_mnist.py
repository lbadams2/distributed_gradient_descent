import tensorflow as tf
from tensorflow.keras import datasets, layers, models

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

model = tf.keras.models.Sequential([
    layers.Conv2D(8, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(8, (5, 5), activation='relu'),
    layers.MaxPool2D((2, 2), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, batch_size=128, epochs=1)

#tf 0.9936 accuracy batch size 32 5 epochs 89.80s
#tf 0.9924 accuracy batch size 64 5 epochs 84.94s
#tf 0.9906 accuracy batch size 128 5 epochs 79.52s