import os
import torch
from PIL import Image
from torch.nn import Module
from config import *

# ────────────────────────────────────────────────
#  Configuration (should match your training script)
# ────────────────────────────────────────────────
mode = ["convnext", "resnet", "retinanet"][0]
if mode == "convnext":
    from MLMTrainerPyTorchConvnext import val_transform_pattern, val_transform_weave, pattern_training_dir, weave_training_dir
    from MLMTrainerPyTorchConvnext import ConvnextModelClassifier as ModelClassifier
elif mode == "resnet":
    from MLMTrainerPyTorchResNet import val_transform_pattern, val_transform_weave, pattern_training_dir, weave_training_dir
    from MLMTrainerPyTorchResNet import ResNetModelClassifier as ModelClassifier
elif mode == "retinanet":
    from MLMTrainerPyTorchRetinaNet import val_transform_pattern, val_transform_weave, pattern_training_dir, weave_training_dir
    from MLMTrainerPyTorchRetinaNet import RetinaNetClassifier as ModelClassifier
else:
    raise ImportError(f"Mode {mode} does not exist!")

# Paths
TEST_DIR = "test-images"
MLM_DIR = "MLMs"
pattern_class_names = sorted(os.listdir(pattern_training_dir))
weave_class_names = sorted(os.listdir(weave_training_dir))
img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE.type}")


# ────────────────────────────────────────────────
#  Load model and weights
# ────────────────────────────────────────────────
def loadModel(n_classes: int, model_path: str):
    model = ModelClassifier(n_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ────────────────────────────────────────────────
#  Prediction function (single image)
# ────────────────────────────────────────────────
def predictImage(img_path: str, model: Module, transform, class_names):
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        pred_class = class_names[predicted_idx.item()]
        conf_percent = confidence.item() * 100

        return pred_class, conf_percent, img

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None


def getModelPath(base_model_name, _dir):
    mlm_path = ""
    _num_classes: int = -1
    for mlm in os.listdir(_dir):
        mlm_contents = mlm.split('_')
        try:
            _num_classes = int(mlm_contents[-1].rstrip('.pt'))
        except ValueError:
            raise FileNotFoundError(f"Model \"{mlm}\" has invalid name")
        if base_model_name == ('_'.join(mlm_contents[:-1]) + ".pt"):
            mlm_path = os.path.join(_dir, mlm)
            break

    if mlm_path == "":
        raise FileNotFoundError(f"Model \"{base_model_name}\" does not exist")

    print(f"Model loaded from: {mlm_path}")
    return _num_classes, mlm_path


print(f"Number of classes: {len(pattern_class_names)}")
print(f"Class names: {pattern_class_names}\n")

image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(img_ext)]
pattern_model = None
weave_model = None


def predictPattern(img_path: str) -> str:
    global pattern_model
    if pattern_model is None: pattern_model = loadModel(*getModelPath("pattern_best_model.pt", f"MLMs/{mode}"))
    pred_class, confidence, pil_img = predictImage(img_path, pattern_model, val_transform_pattern, pattern_class_names)
    if pred_class is None:
        print(f"Skipped: {img_path}")
        return ""
    print(f"Image: {img_path:35s}  →  "
          f"Predicted pattern: {pred_class:18s}  "
          f"(confidence: {confidence:5.2f}%)")
    return pred_class


def predictWeave(img_path: str) -> str:
    global weave_model
    if weave_model is None: weave_model = loadModel(*getModelPath("weave_best_model_grayscale.pt", f"MLMs/{mode}"))
    pred_class, confidence, pil_img = predictImage(img_path, weave_model, val_transform_weave, weave_class_names)
    if pred_class is None:
        print(f"Skipped: {img_path}")
        return ""
    print(f"Image: {img_path:35s}  →  "
          f"Predicted weave: {pred_class:18s}  "
          f"(confidence: {confidence:5.2f}%)")
    return pred_class


if __name__ == '__main__':
    # ────────────────────────────────────────────────
    #  Run inference on all images in test-images/
    # ────────────────────────────────────────────────
    print("Running predictions on test images...\n")

    for filename in sorted(image_files):
        test_img_path = os.path.join(TEST_DIR, filename)

        predictPattern(test_img_path)
        predictWeave(test_img_path)

        # if pred_class in l2_class_names:
        #     l2_model = loadModel(*getModelPath(pred_class))
        #     pred_class, confidence, pil_img = predictImage(test_img_path, l2_model)
        #
        #     print(f"Predicted secondary pattern: {pred_class:18s}  "
        #           f"(confidence: {confidence:5.2f}%)")

        print()

    print("\nINFERENCE COMPLETE")
