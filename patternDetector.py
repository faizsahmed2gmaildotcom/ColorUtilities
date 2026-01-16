import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import matplotlib.pyplot as plt

img_size = (512, 512)
batch_size = 32
img_folder_path = "training-data"
logs_dir = "logs"

# Optional: Image conversion (skip if all images are already JPEG)
for dir_path, dirs_in_dir, img_paths in os.walk(img_folder_path):
    for img_path in img_paths:
        if img_path.endswith(".jpg"):
            out_path = os.path.join(dir_path, img_path.removesuffix(".jpg") + ".jpeg")
            Image.open(os.path.join(dir_path, img_path)).convert("RGB").save(out_path)
            os.remove(os.path.join(dir_path, img_path))
        elif not img_path.endswith(".jpeg"):
            os.remove(os.path.join(dir_path, img_path))

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

# Partition data before preprocessing to avoid augmenting val/test
len_batches = len(img_data)
train_size = int(len_batches * 0.7)
val_size = int(len_batches * 0.2)
test_size = len_batches - train_size - val_size

training_data = img_data.take(train_size).map(lambda pixels, cls: (data_augmentation(pixels) / 255, cls))  # Augment train only
validation_data = img_data.skip(train_size).take(val_size)  # No augment
test_data = img_data.skip(train_size + val_size).take(test_size)  # No augment

print(f"Training size: {train_size}, validation size: {val_size}, testing size: {test_size}")
if train_size == 0 or val_size == 0 or test_size == 0:
    raise IndexError("Dataset not large enough!")

# Model definition
kernel_size = (3, 3)

model = Sequential()

model.add(Input(shape=(*img_size, 3)))

model.add(Conv2D(32, kernel_size, strides=1, padding='same', activation='relu'))  # Increased filters, added padding
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
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
                    epochs=10,  # Increased; early stopping will halt if needed
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

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
