from config import *
import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.applications as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# Define key parameters
train_dir = 'training-data'
img_size: tuple[int, int] = config["general"]["img_size"]
processed_img_size: tuple[int, int] = config["general"]["cropped_img_size"]
batch_size = 8
epochs = 100
validation_split = 0.2
learning_rate = 0.0001

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def show_augmented_examples(
        dataset,
        augmentation: tf.keras.Sequential,
        n_examples_per_class: int = 4,
        dpi: int = 450
) -> None:
    """
    Displays each original image and its augmented version in a separate figure,
    one pair at a time, progressively as they are processed.

    Parameters:
        dataset: tf.data.Dataset from image_dataset_from_directory
        augmentation: The data augmentation layer (tf.keras.Sequential)
        n_examples_per_class: Max number of examples to show per class
        dpi: Screen DPI (affects perceived size)
    """
    _class_names = dataset.class_names
    seen_counts = {cls_name: 0 for cls_name in _class_names}

    for images_batch, labels_batch in dataset:
        for img_tensor, label_tensor in zip(images_batch, labels_batch):
            class_idx = int(label_tensor.numpy())
            class_name = _class_names[class_idx]

            if seen_counts[class_name] >= n_examples_per_class:
                continue

            seen_counts[class_name] += 1

            # Prepare images
            original_img = img_tensor.numpy()
            if original_img.dtype == tf.float32:
                original_img = original_img.astype(np.uint8)

            # Generate augmentation
            aug_tensor = augmentation(tf.expand_dims(img_tensor, 0))[0]
            aug_img = aug_tensor.numpy()
            if aug_img.dtype == tf.float32:
                aug_img = aug_img.astype(np.uint8)

            # Create new figure for this pair
            fig, axes = plt.subplots(
                1, 2,
                figsize=(8, 4.5),
                dpi=dpi,
                gridspec_kw={'wspace': 0.12}
            )

            fig.suptitle(f"Class: {class_name} — {seen_counts[class_name]}",
                         fontsize=14, fontweight='bold')

            # Original
            axes[0].imshow(original_img, aspect='equal', interpolation='lanczos')
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis('off')

            # Augmented
            axes[1].imshow(aug_img, aspect='equal', interpolation='lanczos')
            axes[1].set_title("Augmented", fontsize=12)
            axes[1].axis('off')

            # Force exact display size for images
            for ax in axes:
                bbox = ax.get_position()
                ax.set_position([
                    bbox.x0, bbox.y0,
                    img_size[1] / dpi / fig.get_figwidth(),
                    img_size[0] / dpi / fig.get_figheight()
                ])

            plt.draw()
            plt.pause(5)  # Let user see this pair

            # Close figure to free memory (important when showing many)
            plt.close(fig)

    print("Finished displaying examples.")


# Load the dataset from the directory, splitting into training and validation sets
train_ds = image_dataset_from_directory(
    train_dir,
    interpolation='lanczos5',
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int'  # Integer labels for multi-class classification
)

val_ds = image_dataset_from_directory(
    train_dir,
    interpolation='lanczos5',
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int'
)

# Retrieve class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Apply data augmentation to improve generalization and accuracy
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.01),
    layers.RandomContrast(0.5),
    layers.RandomCrop(*processed_img_size),
])

# show_augmented_examples(train_ds, data_augmentation)

# Use transfer learning with EfficientNet for high accuracy on image classification
base_model = tfa.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*processed_img_size, 3))
base_model.trainable = True  # Freeze base layers initially; can be set to True for fine-tuning later
resize_layer = layers.Resizing(*processed_img_size, interpolation='bilinear')

# Build the model
model = models.Sequential([
    data_augmentation,
    resize_layer,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.build(input_shape=(None, *processed_img_size, 3))

# Display model summary
model.summary()

# Set up callbacks for model saving, logging, and monitoring
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Unique log directory for each session
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)  # For saving logs
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join('MLMs', 'best_model.keras'),  # Save in Keras format for easier loading
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback, checkpoint_callback]
)

# Plot and show accuracy metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Save the final model
model.save(os.path.join('MLMs', 'final_model.keras'))

# For predictions, load the model as follows:
# loaded_model = tf.keras.models.load_model('best_model.keras' or 'final_model.keras')
# Then, use loaded_model.predict() on preprocessed images (resized to 224x224, normalized)
