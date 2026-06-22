import os
import torch
from PIL import Image
from config import *

# ────────────────────────────────────────────────
#  Configuration (should match your training script)
# ────────────────────────────────────────────────
from MLMTrainerPyTorchConvnext import val_transform_pattern, val_transform_weave
from MLMTrainerPyTorchConvnext import ConvnextModelClassifier as ModelClassifier

# Paths
test_data_dir = "test-images"
models_dir = "models"
img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

with open(os.path.join(models_dir, "config.toml"), "rb") as config_file:
    from tomllib import load

    model_cfgs: dict[str, Any] = load(config_file)
    config_file.close()

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE.type}")


# ────────────────────────────────────────────────
#  Load model and weights
# ────────────────────────────────────────────────
def loadModel(model_path: str):
    model = ModelClassifier(model_cfgs[model_path]["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ────────────────────────────────────────────────
#  Prediction function (single image)
# ────────────────────────────────────────────────
def predictImage(img_path: str, model_path: str, transform=val_transform_pattern):
    model = loadModel(model_path)
    class_names = model_cfgs[model_path]["class_names"]
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

    return pred_class, conf_percent


def predictFull(img_path: str, models_path: str, classes=None, confs=None):
    if confs is None:
        confs = []
    if classes is None:
        classes = []

    main_model_path = os.path.join(models_path, 'main.pt')
    if not os.path.exists(main_model_path):
        raise FileNotFoundError(main_model_path + " does not exist!")

    main_class, main_conf = predictImage(img_path, main_model_path)
    classes.append(main_class)
    confs.append(main_conf)

    sub_model_path = os.path.join(models_path, main_class)
    if os.path.exists(sub_model_path):
        # if sub model has sub models
        predictFull(img_path, sub_model_path, classes, confs)
    elif os.path.exists(sub_model_path + '.pt'):
        # if sub model has no sub models
        sub_class, sub_conf = predictImage(img_path, sub_model_path + '.pt')
        classes.append(sub_class)
        confs.append(sub_conf)

    return classes, confs


if __name__ == '__main__':
    models_path = "models/shirting"

    # ────────────────────────────────────────────────
    #  Run inference on all images in test-images
    # ────────────────────────────────────────────────
    print("Running predictions on test images...\n")

    for filename in sorted(os.listdir(test_data_dir)):
        test_img_path = os.path.join(test_data_dir, filename)

        classes, confs = predictFull(test_img_path, models_path)
        print(classes)
        print(confs)

        print()

    print("\nINFERENCE COMPLETE")
