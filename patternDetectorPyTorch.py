import os
import torch
from PIL import Image
from config import *

# Paths
test_data_dir = "test-data"
models_dir = "models"
img_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE.type}")


# ────────────────────────────────────────────────
#  Load model and weights
# ────────────────────────────────────────────────
def loadModel(model_path: str):
    from MLMTrainerPyTorchConvnext import ConvnextModelClassifier
    from torch.nn import Module
    checkpoint: dict[str, Any] = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model: Module = ConvnextModelClassifier(len(checkpoint['class_names'])).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


# ────────────────────────────────────────────────
#  Prediction function (single image)
# ────────────────────────────────────────────────
def predictImage(PIL_img: Image.Image, model_path: str, top_n: int):
    model, checkpoint = loadModel(model_path)
    img_tensor = checkpoint['transform'](PIL_img).unsqueeze(0).to(DEVICE)  # add batch dimension

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=min(top_n, len(checkpoint['class_names'])), dim=1)

    pred_classes = [checkpoint['class_names'][idx] for idx in top_indices[0].cpu().numpy()]
    conf_percents = [prob * 100 for prob in top_probabilities[0].cpu().numpy()]

    return pred_classes, conf_percents


def predictFull(PIL_img: Image.Image, models_dirpath: str, _classes=None, _confs=None, top_n=3):
    if _confs is None: _confs = []
    if _classes is None: _classes = []

    main_model_path = os.path.join(models_dirpath, 'main' + config['general']['model_type'] + '.pt')
    if not os.path.exists(main_model_path):
        raise FileNotFoundError(main_model_path + " does not exist!")

    main_classes, main_confs = predictImage(PIL_img, main_model_path, top_n)
    _classes.append(main_classes)
    _confs.append(main_confs)
    main_class = main_classes[0]

    sub_model_path = os.path.join(models_dirpath, main_class)
    if os.path.exists(sub_model_path):
        # if submodel has submodels
        predictFull(PIL_img, sub_model_path, _classes, _confs, top_n)
    elif os.path.exists(sub_model_path + config['general']['model_type'] + '.pt'):
        # if submodel has no submodels
        sub_classes, sub_confs = predictImage(PIL_img, sub_model_path + config['general']['model_type'] + '.pt', top_n)
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
        classes, confs = predictFull(Image.open(test_img_path), models_path, top_n=2)

        print(filename)
        for i, (tier_classes, tier_confs) in enumerate(zip(classes, confs)):
            formatted_confs = list(map(lambda p: f"{p:.2f}%", tier_confs))
            print(f"\tModel Tier {i + 1}:")
            print(f"\t\tClasses:     {tier_classes}")
            print(f"\t\tConfidences: {formatted_confs}")
        print()
    print("\nINFERENCE COMPLETE")
