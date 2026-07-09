import os
import torch
from PIL import Image
from config import *

# ────────────────────────────────────────────────
#  Configuration (should match your training script)
# ────────────────────────────────────────────────
from MLMTrainerPyTorchConvnext import val_transform_pattern
from MLMTrainerPyTorchConvnext import ConvnextModelClassifier as ModelClassifier

# Paths
test_data_dir = "test-data"
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
def predictImage(img_path: str, model_path: str, transform=val_transform_pattern, top_n=2):
    model = loadModel(model_path)
    class_names = model_cfgs[model_path]["class_names"]

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=min(top_n, len(class_names)), dim=1)

    pred_classes = [class_names[idx] for idx in top_indices[0].cpu().numpy()]
    conf_percents = [prob * 100 for prob in top_probabilities[0].cpu().numpy()]

    return pred_classes, conf_percents


def predictFull(img_path: str, models_dirpath: str, _classes=None, _confs=None, top_n=3):
    if _confs is None: _confs = []
    if _classes is None: _classes = []

    main_model_path = os.path.join(models_dirpath, 'main.pt')
    if not os.path.exists(main_model_path):
        raise FileNotFoundError(main_model_path + " does not exist!")

    main_classes, main_confs = predictImage(img_path, main_model_path, top_n=top_n)
    _classes.append(main_classes)
    _confs.append(main_confs)
    main_class = main_classes[0]

    sub_model_path = os.path.join(models_dirpath, main_class)
    if os.path.exists(sub_model_path):
        # if submodel has submodels
        predictFull(img_path, sub_model_path, _classes, _confs, top_n=top_n)
    elif os.path.exists(sub_model_path + '.pt'):
        # if submodel has no submodels
        sub_classes, sub_confs = predictImage(img_path, sub_model_path + '.pt', top_n=top_n)
        _classes.append(sub_classes)
        _confs.append(sub_confs)

    return _classes, _confs


if __name__ == '__main__':
    models_path = "models/shirting"

    # ────────────────────────────────────────────────
    #  Run inference on all images in test-data
    # ────────────────────────────────────────────────
    print("Running predictions on test images...\n")
    for filename in sorted(os.listdir(test_data_dir)):
        test_img_path = os.path.join(test_data_dir, filename)
        classes, confs = predictFull(test_img_path, models_path, top_n=2)

        print(filename)
        for i, (tier_classes, tier_confs) in enumerate(zip(classes, confs)):
            formatted_confs = list(map(lambda p: f"{p:.2f}%", tier_confs))
            print(f"\tModel Tier {i + 1}:")
            print(f"\t\tClasses:     {tier_classes}")
            print(f"\t\tConfidences: {formatted_confs}")
        print()
    print("\nINFERENCE COMPLETE")
