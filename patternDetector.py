import os
import numpy as np
from pixelLib import preprocessImage
from PIL import Image
from config import *

processed_img_size: tuple[int, int] = config["general"]["cropped_img_size"]
class_names = sorted(os.listdir("training-data"))  # Renamed for clarity; assumes subdirs are classes

for img_file_name in sorted(os.listdir("test-images")):
    preprocessImage(os.path.join("test-images", img_file_name))

model_names = sorted(os.listdir("MLMs"))
print("Select model:")
for i, name in enumerate(model_names):
    print(f"{i}: {name}")

model_no: int = -1
while (model_no < 0) or (model_no >= len(model_names)):
    try:
        model_no = int(input("Enter a model number: "))
    except ValueError:
        continue

selected_model_path = os.path.join("MLMs", model_names[model_no])

from tensorflow.keras.models import load_model
import tensorflow as tf


def preprocessInputImage(img_path: str):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, processed_img_size)
    # img = tf.image.rgb_to_grayscale(img) / 255
    img = tf.image.random_crop(img, processed_img_size)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    return img


def preprocessInputImageGrok(img_path: str) -> np.ndarray:
    """
    Preprocesses a single fabric image file so it can be directly passed to model.predict().

    The function performs the exact same preprocessing steps used during training:
    - Loads the image
    - Resizes to specified dimensions
    - Converts to RGB (in case of any unusual modes)
    - Normalizes pixel values using EfficientNet preprocessing
    - Adds batch dimension

    Args:
        img_path (str): Path to the input JPEG image file

    Returns:
        np.ndarray: Preprocessed image ready for model.predict()
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load image and convert to RGB
    img = Image.open(img_path).convert('RGB')

    # Resize to the exact dimensions used during training
    img = img.resize(processed_img_size, Image.Resampling.LANCZOS)

    # Convert PIL image to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Apply the same preprocessing as EfficientNetB0 expects
    # This scales pixels to [-1, 1] range (mean & std used in EfficientNet)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    img.close()
    return img_array


model = load_model(selected_model_path)

for img_file_name in sorted(os.listdir("test-images")):
    test_img_pred: np.ndarray = model.predict(preprocessInputImageGrok(os.path.join("test-images", img_file_name)))[0]  # Batch of size 1
    test_img_pred *= 100
    img_name = os.path.splitext(img_file_name)[0]
    print(len(test_img_pred), len(class_names))
    print(f"\r{img_name}: "  # {list(map(lambda p: round(p, 1), test_img_pred.tolist()))}"
          f" -> {class_names[np.argmax(test_img_pred)]}: {np.max(test_img_pred):3f}% match\n")
