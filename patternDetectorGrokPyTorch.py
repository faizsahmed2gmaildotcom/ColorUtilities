import os
import torch
from PIL import Image
from MLMTrainerPyTorch import ModelClassifier, val_transform

# ────────────────────────────────────────────────
#  Configuration (should match your training script)
# ────────────────────────────────────────────────
from config import *

# Paths
MODEL_PATH = os.path.join("MLMs", "best_model.pt")  # or "final_model.pt"
TEST_DIR = "test-images"
class_names = sorted(os.listdir("training-data"))
img_size = config["general"]["img_size"]
processed_img_size = config["general"]["cropped_img_size"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────
#  Load model and weights
# ────────────────────────────────────────────────
model = ModelClassifier(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"Model loaded from: {MODEL_PATH}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}\n")


# ────────────────────────────────────────────────
#  Prediction function (single image)
# ────────────────────────────────────────────────
def predict_image(image_path):
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        pred_class = class_names[predicted_idx.item()]
        conf_percent = confidence.item() * 100

        return pred_class, conf_percent, img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


# ────────────────────────────────────────────────
#  Run inference on all images in test-images/
# ────────────────────────────────────────────────
print("Running predictions on test images...\n")

image_files = [f for f in os.listdir(TEST_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

for filename in sorted(image_files):
    full_path = os.path.join(TEST_DIR, filename)

    pred_class, confidence, pil_img = predict_image(full_path)

    if pred_class is not None:
        print(f"Image: {filename:35s}  →  "
              f"Predicted pattern: {pred_class:18s}  "
              f"(confidence: {confidence:5.2f}%)")

        # Optional: show image with prediction (uncomment if desired)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(pil_img)
        # plt.title(f"{filename}\n{pred_class} ({confidence:.2f}%)")
        # plt.axis('off')
        # plt.show()
    else:
        print(f"Skipped: {filename}")

print("\nInference complete.")
