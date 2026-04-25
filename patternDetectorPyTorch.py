import os
import torch
from PIL import Image
from MLMTrainerPyTorch import ConvnextModelClassifier, val_transform, train_transform
from config import *

# ────────────────────────────────────────────────
#  Configuration (should match your training script)
# ────────────────────────────────────────────────

# Paths
TEST_DIR = "test-images"
MLM_DIR = "MLMs"
l1_class_names = sorted(os.listdir("training-data/level-1"))
l2_class_names = sorted(os.listdir("training-data/level-2"))
img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
img_size = config["general"]["img_size"]
processed_img_size = config["general"]["cropped_img_size"]

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────────
#  Load model and weights
# ────────────────────────────────────────────────
def loadModel(n_classes: int, model_path: str) -> ConvnextModelClassifier:
    model = ConvnextModelClassifier(n_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ────────────────────────────────────────────────
#  Prediction function (single image)
# ────────────────────────────────────────────────
def predictImage(image_path, model):
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        pred_class = l1_class_names[predicted_idx.item()]
        conf_percent = confidence.item() * 100

        return pred_class, conf_percent, img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


def getModelPath(base_model_name):
    mlm_path = ""
    _num_classes: int = -1
    for mlm in os.listdir(MLM_DIR):
        mlm_contents = mlm.split('_')
        try:
            _num_classes = int(mlm_contents[-1].rstrip('.pt'))
        except ValueError:
            raise FileNotFoundError(f"Model \"{mlm}\" has invalid name")
        if base_model_name == '_'.join(mlm_contents[:-1]):
            mlm_path = os.path.join(MLM_DIR, mlm)

    if mlm_path == "":
        raise FileNotFoundError(f"Model \"{base_model_name}\" does not exist")

    print(f"Model loaded from: {mlm_path}")
    return _num_classes, mlm_path


# ────────────────────────────────────────────────
#  Run inference on all images in test-images/
# ────────────────────────────────────────────────
print(f"Number of classes: {len(l1_class_names)}")
print(f"Class names: {l1_class_names}\n")

image_files = [f for f in os.listdir(TEST_DIR)
               if f.lower().endswith(img_ext)]
print("Running predictions on test images...\n")
l1_model = loadModel(*getModelPath("best_model_grayscale"))

for filename in sorted(image_files):
    test_img_path = os.path.join(TEST_DIR, filename)

    l1_model = loadModel(*getModelPath("best_model_grayscale"))
    pred_class, confidence, pil_img = predictImage(test_img_path, l1_model)

    if pred_class is None:
        print(f"Skipped: {filename}")
        continue

    print(f"Image: {filename:35s}  →  "
          f"Predicted pattern: {pred_class:18s}  "
          f"(confidence: {confidence:5.2f}%)")

    if pred_class in l2_class_names:
        l2_model = loadModel(*getModelPath(pred_class))
        pred_class, confidence, pil_img = predictImage(test_img_path, l2_model)

        print(f"Predicted secondary pattern: {pred_class:18s}  "
              f"(confidence: {confidence:5.2f}%)")

    # Optional: show image with prediction (uncomment if desired)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(pil_img)
    # plt.title(f"{filename}\n{pred_class} ({confidence:.2f}%)")
    # plt.axis('off')
    # plt.show()

print("\nINFERENCE COMPLETE")
