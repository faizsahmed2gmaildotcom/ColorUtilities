import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, RandomFlip, RandomZoom
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import matplotlib.pyplot as plt

img_size = (256, 256)
batch_size = 32
training_data_path = "training-data"
logs_dir = "logs"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load dataset with categorical labels (one-hot)
img_data = tf.keras.utils.image_dataset_from_directory(
    training_data_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Data augmentation layers (applied to training data only)
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomZoom(0.2)
])


def preprocessValData(img, label):
    img /= 255
    img = tf.image.rgb_to_grayscale(img)
    return img, label


def preprocessTrainingData(img, label):
    img = data_augmentation(img)
    return preprocessValData(img, label)


# Partition data before preprocessing to avoid augmenting val/test
len_batches = len(img_data)
train_size = int(len_batches * 0.7)
val_size = int(len_batches * 0.2)
test_size = len_batches - train_size - val_size

training_data = img_data.take(train_size).map(preprocessTrainingData, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = img_data.skip(train_size).take(val_size).map(preprocessValData, num_parallel_calls=tf.data.AUTOTUNE)
test_data = img_data.skip(train_size + val_size).take(test_size).map(preprocessValData, num_parallel_calls=tf.data.AUTOTUNE)

print(f"Training size: {train_size}, validation size: {val_size}, testing size: {test_size}")
if train_size == 0 or val_size == 0 or test_size == 0:
    raise IndexError("Dataset not large enough!")

# Model definition
kernel_size = (3, 3)

model = Sequential()

model.add(Input(shape=(*img_size, 1)))

model.add(Conv2D(32, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(os.listdir("training-data")), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

model.summary()

# Training with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

history = model.fit(training_data,
                    epochs=100,
                    validation_data=validation_data,
                    callbacks=[tensorboard_callback, early_stopping])

# Testing with correct metrics
cat_accuracy = CategoricalAccuracy()
precision = Precision()  # Defaults to multi-class 'micro' average
recall = Recall()

for batch in test_data.as_numpy_iterator():
    x, y = batch
    y_pred = model.predict(x)
    cat_accuracy.update_state(y, y_pred)
    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)

print(f"Precision: {precision.result().numpy()}\n"
      f"Recall: {recall.result().numpy()}\n"
      f"Accuracy: {cat_accuracy.result().numpy()}")

model_name = input("Save mode in MLMs directory as: ")
model.save(os.path.join("MLMs", model_name + ".keras"))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
