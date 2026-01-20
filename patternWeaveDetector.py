import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import matplotlib.pyplot as plt

rgb_weights = [0.2989, 0.5870, 0.1140]
img_size = (256, 256)
batch_size = 32
img_folder_path = "training-data"
logs_dir = "logs"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class_names = sorted(os.listdir(img_folder_path))  # Renamed for clarity; assumes subdirs are classes

# Load dataset with categorical labels (one-hot)
img_data = tf.keras.utils.image_dataset_from_directory(
    img_folder_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Data augmentation layers (applied to training data only)
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
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

model.add(Conv2D(32, kernel_size, strides=1, padding='same', activation='relu'))  # Increased filters, added padding
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Too many units will cause overfitting
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(3, activation='softmax'))  # 3 classes

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Adjusted for softmax
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])  # Multi-class compatible

model.summary()

# Training with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

history = model.fit(training_data,
                    epochs=50,
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

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()


def preprocessImg(img_path: str):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.rgb_to_grayscale(img) / 255
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    return img


test_dir_path = os.path.join("test-images")
for img_file_name in sorted(os.listdir(test_dir_path)):
    img_name = os.path.splitext(img_file_name)[0]
    test_img_pred: np.ndarray = model.predict(preprocessImg(os.path.join(test_dir_path, img_file_name)))[0]  # Batch of size 1
    print(f"{img_name}: {test_img_pred} -> {class_names[np.argmax(test_img_pred)]}: {np.max(test_img_pred) * 100:3f}% match\n")
